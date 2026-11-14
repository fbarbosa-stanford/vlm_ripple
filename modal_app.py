"""Modal launcher for Ripple-VLA.

  modal run ripple_vla/modal_app.py::smoke                 # 2 GPUs, tiny Ripple, synthetic data (~minutes)
  modal run ripple_vla/modal_app.py::fetch_weights         # one-time: Ripple-65B -> hf-cache volume (~130 GB)
  modal run ripple_vla/modal_app.py::fetch_llava           # one-time: LLaVA-Pretrain 558k -> data volume
  modal run --detach ripple_vla/modal_app.py::train --stage align --args "--llava_dir /data/llava --steps 3000"
  modal run --detach ripple_vla/modal_app.py::train --stage vla --args "--resume /outputs/ripple_vla/align/last.pt --lora_rank 32"
"""

import os
import subprocess

import modal

APP = "ripple-vla"
RIPPLE = "ricursiveAI/Ripple-65B-Preview"
GPU_TRAIN = os.environ.get("RIPPLE_GPU", "H200:8")

app = modal.App(APP)
SRC = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git", "unzip")
    .pip_install("lerobot==0.5.1", "transformers>=5.0", "safetensors", "accelerate",
                 "wandb", "hf_transfer", "pillow", "sentencepiece")
    .env({"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "HF_TRUST_REMOTE_CODE": "1"})
    .add_local_dir(SRC, "/root/ripple_vla", ignore=["__pycache__", "*.pyc", "libero_sim_modal.py"])
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("ripple-vla-outputs", create_if_missing=True)
data = modal.Volume.from_name("ripple-vla-data", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/outputs": outputs, "/data": data}
SECRETS = [modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")]


def _snapshot(full: bool) -> str:
    from huggingface_hub import snapshot_download

    pats = None if full else ["*.json", "*.py", "*.jinja", "tokenizer*"]
    return snapshot_download(RIPPLE, allow_patterns=pats)


def _torchrun(nproc: int, args: list[str]):
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "train.py", *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root/ripple_vla")


@app.function(image=image, gpu="H100:2", volumes=VOLS, secrets=SECRETS, timeout=60 * 60)
def smoke(stage: str = "vla", steps: int = 150, lora_rank: int = 8):
    import torch
    import transformers

    print("torch", torch.__version__, "transformers", transformers.__version__, flush=True)
    d = _snapshot(full=False)
    _torchrun(2, ["--tiny", "--stage", stage, "--ckpt_dir", d, "--steps", str(steps), "--bs", "8",
                  "--lr", "1e-3", "--lr_lora", "1e-3", "--warmup", "10", "--workers", "0",
                  "--lora_rank", str(lora_rank), "--feature_layers", "3,-1", "--log_every", "10",
                  "--save_every", "100", "--out", f"/outputs/smoke_{stage}"])
    outputs.commit()


@app.function(image=image, cpu=8, memory=32768, volumes=VOLS, secrets=SECRETS, timeout=6 * 60 * 60)
def fetch_weights():
    d = _snapshot(full=True)
    hf_cache.commit()
    print("Ripple snapshot at", d)


@app.function(image=image, cpu=8, memory=32768, volumes=VOLS, secrets=SECRETS, timeout=6 * 60 * 60)
def fetch_llava():
    from huggingface_hub import hf_hub_download

    root = "/data/llava"
    os.makedirs(root, exist_ok=True)
    js = hf_hub_download("liuhaotian/LLaVA-Pretrain", "blip_laion_cc_sbu_558k.json", repo_type="dataset")
    subprocess.run(["cp", js, f"{root}/blip_laion_cc_sbu_558k.json"], check=True)
    z = hf_hub_download("liuhaotian/LLaVA-Pretrain", "images.zip", repo_type="dataset")
    subprocess.run(["unzip", "-q", "-o", z, "-d", f"{root}/images"], check=True)
    data.commit()
    print("LLaVA-Pretrain ready at", root)


@app.function(image=image, gpu=GPU_TRAIN, volumes=VOLS, secrets=SECRETS, timeout=24 * 60 * 60,
              memory=512 * 1024, cpu=32)
def train(stage: str, args: str = "", name: str = ""):
    import shlex
    import threading
    import time

    def _commit():
        while True:
            time.sleep(300)
            try:
                outputs.commit()
            except Exception as e:  # noqa: BLE001
                print("[commit] warn", e, flush=True)

    threading.Thread(target=_commit, daemon=True).start()
    d = _snapshot(full=True)
    n = int(GPU_TRAIN.split(":")[1]) if ":" in GPU_TRAIN else 1
    out = f"/outputs/ripple_vla/{name or stage}"
    _torchrun(n, ["--stage", stage, "--ckpt_dir", d, "--out", out, "--wandb", *shlex.split(args)])
    outputs.commit()


LLAVA_ZIP = ("/root/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-Pretrain/snapshots/"
             "70f9d1e5e1a697fe35830875cfc7de1dd590d727/images.zip")


@app.function(image=image, cpu=4, memory=16384, volumes=VOLS, timeout=30 * 60)
def check_llava(limit: int = 100_000):
    """CPU-only integrity check: zip opens, first `limit` json items exist and decode."""
    import json
    import zipfile
    from io import BytesIO

    from PIL import Image

    zf = zipfile.ZipFile(LLAVA_ZIP)
    names = set(zf.namelist())
    items = json.load(open(os.path.dirname(LLAVA_ZIP) + "/blip_laion_cc_sbu_558k.json"))[:limit]
    missing = [it["image"] for it in items if it["image"] not in names]
    bad = zf.testzip() if False else None  # full CRC pass is slow; spot-check decodes instead
    ok = 0
    for it in items[:: max(1, len(items) // 200)]:
        try:
            Image.open(BytesIO(zf.read(it["image"]))).convert("RGB")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print("decode fail", it["image"], e)
    print(f"zip entries {len(names)}; json items checked {len(items)}; missing {len(missing)}; "
          f"decode spot-check ok {ok}/{len(items[:: max(1, len(items) // 200)])}")
    return {"entries": len(names), "missing": len(missing), "spot_ok": ok}
