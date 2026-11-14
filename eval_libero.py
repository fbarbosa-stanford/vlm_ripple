"""LIBERO closed-loop eval on Modal: one GPU policy server (dynamic batching) + many CPU sim workers.

  modal run ripple_vla/eval_libero.py --ckpt /outputs/ripple_vla/vla/last.pt --suite libero_spatial --episodes 10
  modal run ripple_vla/eval_libero.py --ckpt /outputs/smoke_vla/last.pt --tiny --suite libero_spatial --episodes 1 --tasks 1

Conventions (match the HuggingFaceVLA/libero training data, openpi-style):
  - images: agentview + wrist, rotated 180 deg ([::-1, ::-1]) vs raw env render
  - state (8): eef_pos(3) + axis-angle(eef_quat)(3) + gripper_qpos(2)
  - actions (7): delta EEF (6) + gripper, sent to env.step as-is
  - first 10 steps: no-op [0]*6+[-1] so objects settle; replan every `replan` actions of each 50-chunk
"""

import json
import os
import time

import modal

from libero_sim_modal import OPENPI_MAX_STEPS
from libero_sim_modal import image as sim_image
from modal_app import RIPPLE, VOLS, SECRETS, image as policy_image

app = modal.App("ripple-vla-eval")
POLICY_GPU = os.environ.get("RIPPLE_EVAL_GPU", "H200:2")
sim_image = sim_image.add_local_python_source("libero_sim_modal")


@app.cls(image=policy_image, gpu=POLICY_GPU, volumes=VOLS, secrets=SECRETS, timeout=6 * 60 * 60,
         scaledown_window=300)
class Policy:
    ckpt: str = modal.parameter()
    tiny: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/root/ripple_vla")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from data import Collator, vla_prompt
        from loading import build_vla_inference

        d = snapshot_download(RIPPLE, allow_patterns=None if not self.tiny else ["*.json", "*.py", "*.jinja", "tokenizer*"])
        self.model, self.cfg = build_vla_inference(self.ckpt, d, tiny=self.tiny)
        self.collate = Collator(AutoTokenizer.from_pretrained(d))
        self.vla_prompt = vla_prompt
        self.torch = torch

    @modal.batched(max_batch_size=32, wait_ms=30)
    def act(self, obs: list[dict]) -> list:
        import numpy as np

        torch = self.torch
        samples = [{"images": torch.from_numpy(np.stack([o["agentview"], o["wrist"]])).permute(0, 3, 1, 2).float() / 255.0,
                    "prompt": self.vla_prompt(o["task"])} for o in obs]
        b = self.collate(samples)
        state = torch.tensor(np.stack([o["state"] for o in obs]), dtype=torch.float32).cuda()
        with torch.inference_mode():
            a = self.model.sample_actions(b["images"].cuda(), b["input_ids"].cuda(), b["attention_mask"].cuda(), state)
        return [x for x in a.cpu().numpy()]


@app.function(image=sim_image, cpu=2, memory=4096, timeout=2 * 60 * 60, max_containers=200)
def rollout(ckpt: str, tiny: bool, suite: str, task_id: int, ep: int, replan: int = 5, save_video: bool = False):
    import math

    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    def quat2axisangle(q):
        q = np.array(q, dtype=np.float64)
        q[3] = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(1.0 - q[3] * q[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (q[:3] * 2.0 * math.acos(q[3])) / den

    ts = benchmark.get_benchmark_dict()[suite]()
    task = ts.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=ts.get_task_bddl_file_path(task_id), camera_heights=256, camera_widths=256)
    env.seed(ep)
    env.reset()
    obs = env.set_init_state(ts.get_task_init_states(task_id)[ep % 50])
    policy = Policy(ckpt=ckpt, tiny=tiny)
    max_steps = OPENPI_MAX_STEPS[suite]
    frames, plan, success, t, t0, n_calls = [], [], False, 0, time.time(), 0
    while t < max_steps + 10:
        if t < 10:
            act = [0.0] * 6 + [-1.0]
        else:
            if not plan:
                o = {"agentview": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                     "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                     "state": np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                              obs["robot0_gripper_qpos"]]).astype(np.float32),
                     "task": task.language}
                chunk = policy.act.remote(o)
                n_calls += 1
                plan = [np.asarray(x, dtype=np.float64) for x in chunk[:replan]]
            act = plan.pop(0).tolist()
        obs, _, done, _ = env.step(act)
        if save_video:
            frames.append(obs["agentview_image"][::-1, ::-1].copy())
        t += 1
        if done:
            success = True
            break
    env.close()
    out = {"suite": suite, "task_id": task_id, "ep": ep, "task": task.language, "success": success,
           "steps": t, "policy_calls": n_calls, "sec": round(time.time() - t0, 1)}
    if save_video and frames:
        out["frames"] = np.stack(frames[::2])  # 10 fps -> 5 fps preview to keep payload small
    return out


@app.local_entrypoint()
def main(ckpt: str, suite: str = "libero_spatial", episodes: int = 10, tasks: int = 10, tiny: bool = False,
         replan: int = 5, videos: int = 2):
    import numpy as np

    jobs = [(t, e) for t in range(tasks) for e in range(episodes)]
    print(f"[eval] {len(jobs)} rollouts on {suite} ({tasks} tasks x {episodes} eps) ckpt={ckpt}")
    results = []
    for i, r in enumerate(rollout.starmap(
            [(ckpt, tiny, suite, t, e, replan, i < videos) for i, (t, e) in enumerate(jobs)],
            order_outputs=False)):
        frames = r.pop("frames", None)
        if frames is not None:
            os.makedirs("eval_videos", exist_ok=True)
            p = f"eval_videos/{suite}_t{r['task_id']}_e{r['ep']}_{'ok' if r['success'] else 'fail'}.npy"
            np.save(p, frames)
            r["video_npy"] = p
        results.append(r)
        print(json.dumps(r), flush=True)
    by_task = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(r["success"])
    for t, s in sorted(by_task.items()):
        print(f"task {t:2d}: {sum(s)}/{len(s)}")
    sr = sum(r["success"] for r in results) / max(1, len(results))
    print(f"[eval] {suite} success rate: {sr:.1%} over {len(results)} episodes")
    json.dump(results, open(f"eval_{suite}.json", "w"), indent=1)
