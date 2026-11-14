"""Datasets + collation.

Prompt format (Ripple chat template, ChatML) with images spliced after a fixed prefix:

    <|endoftext|><|im_start|>user\n [IMAGES] {prompt}<|im_end|>\n<|im_start|>assistant\n {target}<|im_end|>

PREFIX is identical for every sample, so the image insertion point is a constant
(`image_insert_pos(tokenizer)`), which the model config stores.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

PREFIX = "<|endoftext|><|im_start|>user\n"
EOT = "<|im_end|>"
ASSISTANT = f"{EOT}\n<|im_start|>assistant\n"

LIBERO_REPO = "HuggingFaceVLA/libero"   # LeRobot v3.0, 1693 eps, 273k frames, 10 fps, 40 tasks
CAM_KEYS = ("observation.images.image", "observation.images.image2")  # agentview, wrist


def image_insert_pos(tok) -> int:
    return len(tok(PREFIX, add_special_tokens=False).input_ids)


def vla_prompt(task: str) -> str:
    return f"What action should the robot take to {task.strip().rstrip('.').lower()}?"


# --------------------------------------------------------------------------------------
class Collator:
    """Tokenises (prompt, target) pairs, right-pads, builds labels (-100 outside target)."""

    def __init__(self, tok, max_text_len: int = 96):
        self.tok = tok
        self.max_len = max_text_len
        self.pad = tok.pad_token_id
        self.prefix = tok(PREFIX, add_special_tokens=False).input_ids

    def encode(self, prompt: str, target: str | None):
        ids = self.prefix + self.tok(prompt + ASSISTANT, add_special_tokens=False).input_ids
        lab = [-100] * len(ids)
        if target is not None:
            t = self.tok(target + EOT, add_special_tokens=False).input_ids
            ids, lab = ids + t, lab + t
        return ids[: self.max_len], lab[: self.max_len]

    def __call__(self, samples: list[dict]) -> dict:
        enc = [self.encode(s["prompt"], s.get("target")) for s in samples]
        L = max(len(i) for i, _ in enc)
        ids = torch.full((len(enc), L), self.pad, dtype=torch.long)
        lab = torch.full((len(enc), L), -100, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for b, (i, l) in enumerate(enc):
            ids[b, : len(i)] = torch.tensor(i)
            lab[b, : len(l)] = torch.tensor(l)
            att[b, : len(i)] = 1
        out = {"input_ids": ids, "labels": lab, "attention_mask": att,
               "images": torch.stack([s["images"] for s in samples])}
        for k in ("state", "actions", "action_is_pad"):
            if k in samples[0]:
                out[k] = torch.stack([s[k] for s in samples])
        return out


# --------------------------------------------------------------------------------------
class LiberoVLA(Dataset):
    """LeRobot LIBERO frames -> {images (2,3,256,256) in [0,1], state (8), actions (chunk,7), prompt}."""

    def __init__(self, repo_id: str = LIBERO_REPO, chunk: int = 50, root: str | None = None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id, root=root)
        fps = meta.fps
        self.ds = LeRobotDataset(repo_id, root=root,
                                 delta_timestamps={"action": [i / fps for i in range(chunk)]})
        st = meta.stats
        self.stats = {k: {s: np.asarray(st[k][s], dtype=np.float32).tolist() for s in ("mean", "std")}
                      for k in ("action", "observation.state")}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x = self.ds[i]
        imgs = torch.stack([x[k] for k in CAM_KEYS])  # (2,3,H,W) float [0,1]
        return {"images": imgs, "state": x["observation.state"].float(), "actions": x["action"].float(),
                "action_is_pad": x["action_is_pad"].bool(), "prompt": vla_prompt(x["task"])}


class LiberoCaption(Dataset):
    """Cheap in-domain alignment: LIBERO frames captioned with their task string."""

    QUESTIONS = ("What is the robot doing?", "Describe the task shown in these images.",
                 "What task is being performed?")

    def __init__(self, vla_ds: LiberoVLA, stride: int = 10):
        self.base = vla_ds.ds
        self.idx = list(range(0, len(self.base), stride))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = self.base[self.idx[i]]
        imgs = torch.stack([x[k] for k in CAM_KEYS])
        return {"images": imgs, "prompt": random.choice(self.QUESTIONS),
                "target": f"The robot is going to {x['task'].strip().rstrip('.').lower()}."}


class LlavaPretrain(Dataset):
    """liuhaotian/LLaVA-Pretrain (558k BLIP captions). Either `root` (json + extracted images/) or
    `zip_path` (images read straight from images.zip, json next to it; no extraction needed)."""

    JSON = "blip_laion_cc_sbu_558k.json"

    def __init__(self, root: str | None = None, size: int = 224, limit: int | None = None,
                 zip_path: str | None = None):
        self.root, self.size, self.zip_path = root, size, zip_path
        self._zf, self._pid = None, None
        base = os.path.dirname(zip_path) if zip_path else root
        self.items = json.load(open(os.path.join(base, self.JSON)))
        if limit:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def _open(self, rel: str):
        if self.zip_path:
            import zipfile

            if self._zf is None or self._pid != os.getpid():  # one handle per dataloader worker
                self._zf, self._pid = zipfile.ZipFile(self.zip_path), os.getpid()
            import io

            return io.BytesIO(self._zf.read(rel))
        return os.path.join(self.root, "images", rel)

    def __getitem__(self, i):
        from PIL import Image

        it = self.items[i]
        img = Image.open(self._open(it["image"])).convert("RGB")
        img = img.resize((self.size, self.size), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        q = it["conversations"][0]["value"].replace("<image>", "").strip()
        a = it["conversations"][1]["value"].strip()
        return {"images": t[None], "prompt": q, "target": a}


class Synthetic(Dataset):
    """For smoke tests: learnable fake data (actions are a fixed function of state)."""

    def __init__(self, n=256, num_images=2, chunk=50, caption=False):
        g = torch.Generator().manual_seed(0)
        self.n, self.caption = n, caption
        self.imgs = torch.rand(8, num_images, 3, 64, 64, generator=g)
        self.state = torch.randn(n, 8, generator=g)
        W = torch.randn(8, 7, generator=g) * 0.5
        steps = torch.linspace(0, 1, chunk)[:, None]
        self.actions = torch.tanh(self.state @ W)[:, None, :] * (1 + steps)  # (n, chunk, 7)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = i % 8
        if self.caption:
            return {"images": self.imgs[k], "prompt": "Describe the image.", "target": f"image number {k}"}
        return {"images": self.imgs[k], "state": self.state[i], "actions": self.actions[i],
                "action_is_pad": torch.zeros(self.actions.shape[1], dtype=torch.bool),
                "prompt": vla_prompt(f"do task {k}")}
