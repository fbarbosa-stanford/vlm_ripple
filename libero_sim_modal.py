"""Headless LIBERO (robosuite + MuJoCo) smoke test on Modal.

Run from the parent directory of ripple_vla/:
    modal run ripple_vla/libero_sim_modal.py
    modal run ripple_vla/libero_sim_modal.py --suite libero_object --task-id 3 --steps 100

Rendering is CPU-only via OSMesa (MUJOCO_GL=osmesa). No GPU is requested.
"""

import json
import os
import time

import modal

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------
LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"  # master as of 2026-11-13
LIBERO_DIR = "/opt/LIBERO"
SCRATCH_MOUNT = "/scratch"
VOLUME_NAME = "ripple-vla-scratch"

# Reference max episode lengths from openpi's examples/libero/main.py
# (not derived from the bddl files).
OPENPI_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
SUITES_TO_REPORT = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def write_libero_config() -> None:
    """Pre-create ~/.libero/config.yaml so LIBERO's import does not call input()."""
    import yaml

    root = os.path.join(LIBERO_DIR, "libero", "libero")
    cfg = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "bddl_files"),
        "init_states": os.path.join(root, "init_files"),
        "datasets": os.path.join(LIBERO_DIR, "libero", "datasets"),
        "assets": os.path.join(root, "assets"),
    }
    os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
    with open(os.path.expanduser("~/.libero/config.yaml"), "w") as f:
        yaml.dump(cfg, f)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libgl1",
        "libegl1",
        "libglib2.0-0",
        "libosmesa6",
        "libosmesa6-dev",
        "libgl1-mesa-dri",
    )
    .pip_install("pip==24.0", "setuptools==69.5.1", "wheel")
    .pip_install("torch==2.1.2", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "numpy==1.23.5",
        "mujoco==3.2.3",
        "robosuite==1.4.1",
        "bddl==1.0.1",
        "easydict==1.9",
        "gym==0.25.2",
        "matplotlib==3.7.5",
        "opencv-python-headless==4.8.1.78",
        "h5py==3.11.0",
        "pyyaml",
        "future",
        "cloudpickle==2.1.0",
        "termcolor",
        "pillow",
    )
    .run_commands(
        f"git clone {LIBERO_REPO} {LIBERO_DIR} && cd {LIBERO_DIR} && git checkout {LIBERO_COMMIT}"
    )
    .run_commands(f"cd {LIBERO_DIR} && pip install --no-deps -e .")
    .run_function(write_libero_config)
    .env(
        {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

app = modal.App("ripple-vla-libero-smoke")
scratch = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={SCRATCH_MOUNT: scratch},
    cpu=4,
    memory=8192,
    timeout=1800,
)
def smoke(suite: str = "libero_spatial", task_id: int = 0, steps: int = 50) -> dict:
    import numpy as np
    from PIL import Image

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict[suite]()
    task = task_suite.get_task(task_id)
    bddl_path = task_suite.get_task_bddl_file_path(task_id)
    init_states = task_suite.get_task_init_states(task_id)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(init_states[0])

        noop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        t0 = time.perf_counter()
        for _ in range(steps):
            out = env.step(noop)
            obs = out[0]
        elapsed = time.perf_counter() - t0
        ms_per_step = (elapsed / steps * 1000.0) if steps > 0 else None

        obs_info = {
            k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
            for k, v in sorted(obs.items())
            if isinstance(v, np.ndarray)
        }
        robot_state = {
            k: np.asarray(obs[k]).tolist()
            for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            if k in obs
        }

        os.makedirs(SCRATCH_MOUNT, exist_ok=True)
        png_path = os.path.join(SCRATCH_MOUNT, "libero_smoke.png")
        Image.fromarray(np.asarray(obs["agentview_image"])).save(png_path)
        scratch.commit()
    finally:
        env.close()

    suite_summary = {}
    for name in SUITES_TO_REPORT:
        s = bench_dict[name]()
        suite_summary[name] = {
            "n_tasks": s.get_num_tasks(),
            "openpi_ref_max_steps": OPENPI_MAX_STEPS[name],
        }

    return {
        "suite": suite,
        "task_id": task_id,
        "task_language": task.language,
        "bddl_file": bddl_path,
        "num_init_states": len(init_states),
        "steps_run": steps,
        "ms_per_env_step": ms_per_step,
        "obs_keys_shapes": obs_info,
        "agentview_image_shape": list(np.asarray(obs["agentview_image"]).shape),
        "robot0_eye_in_hand_image_shape": list(
            np.asarray(obs["robot0_eye_in_hand_image"]).shape
        ),
        "robot_state": robot_state,
        "png_path_in_volume": png_path,
        "suites": suite_summary,
    }


@app.local_entrypoint()
def main(suite: str = "libero_spatial", task_id: int = 0, steps: int = 50):
    result = smoke.remote(suite=suite, task_id=task_id, steps=steps)
    print(json.dumps(result, indent=2))
