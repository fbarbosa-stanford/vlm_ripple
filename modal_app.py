"""Modal launcher for Ripple-VLA.

  modal run ripple_vla/modal_app.py::smoke                 # 2 GPUs, tiny Ripple, synthetic data (~minutes)
  modal run ripple_vla/modal_app.py::fetch_weights         # one-time: Ripple-65B -> hf-cache volume (~130 GB)
  modal run ripple_vla/modal_app.py::fetch_llava           # one-time: LLaVA-Pretrain 558k -> data volume
  modal run --detach ripple_vla/modal_app.py::train --stage align --args "--llava_dir /data/llava --steps 3000"
  modal run --detach ripple_vla/modal_app.py::train --stage vla --args "--resume /outputs/ripple_vla/align/last.pt --lora_rank 32"
  modal run ripple_vla/modal_app.py::eval_vlm                                   # random-projector baseline
  modal run ripple_vla/modal_app.py::eval_vlm --ckpt /outputs/ripple_vla/vlm/last.pt
  modal run ripple_vla/modal_app.py::eval_vlm --tiny --args "--n 32 --tasks shapes,pope"   # pipeline test
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
coco = modal.Volume.from_name("ripple-vla-coco", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/outputs": outputs, "/data": data, "/coco": coco}
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


@app.function(image=image, cpu=8, memory=32768, volumes=VOLS, secrets=SECRETS, timeout=6 * 60 * 60)
def fetch_instruct():
    """LLaVA-Instruct-150K json + COCO train2017.zip (images are read in place from the zip; never extracted,
    since the Modal volume caps the number of files)."""
    import shutil
    import time
    import urllib.request

    from huggingface_hub import hf_hub_download

    os.makedirs("/coco", exist_ok=True)
    js = hf_hub_download("liuhaotian/LLaVA-Instruct-150K", "llava_instruct_150k.json", repo_type="dataset")
    shutil.copy(js, "/coco/llava_instruct_150k.json")
    dst = "/coco/train2017.zip"
    t0, n = time.time(), 0
    with urllib.request.urlopen("http://images.cocodataset.org/zips/train2017.zip", timeout=120) as r, open(dst, "wb") as f:
        while True:
            b = r.read(8 << 20)
            if not b:
                break
            f.write(b)
            n += len(b)
            if (n >> 20) % 1024 < 8:
                print(f"[coco] {n / 2**30:.1f} GiB in {time.time() - t0:.0f}s", flush=True)
    coco.commit()
    print(f"[coco] done: {n / 2**30:.1f} GiB -> {dst}", flush=True)


@app.function(image=image, cpu=4, memory=16384, volumes=VOLS, timeout=30 * 60)
def check_instruct(limit: int = 50_000):
    """CPU-only: zip opens, every image of the (shuffled, seed 0) subset exists and a sample decodes."""
    import json
    import random
    import zipfile
    from io import BytesIO

    from PIL import Image

    zf = zipfile.ZipFile("/coco/train2017.zip")
    names = set(zf.namelist())
    items = json.load(open("/coco/llava_instruct_150k.json"))
    random.Random(0).shuffle(items)
    items = items[:limit]
    missing = [it["image"] for it in items if "train2017/" + it["image"] not in names]
    ok = 0
    for it in items[:: max(1, len(items) // 200)]:
        Image.open(BytesIO(zf.read("train2017/" + it["image"]))).convert("RGB")
        ok += 1
    turns = sum(len(it["conversations"]) // 2 for it in items) / len(items)
    print(f"zip entries {len(names)}; conversations {len(items)}; missing images {len(missing)}; "
          f"decode ok {ok}; avg QA turns {turns:.1f}")
    return {"missing": len(missing), "decode_ok": ok}


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


@app.function(image=image, gpu=os.environ.get("RIPPLE_EVAL_GPU", "H200:2"), volumes=VOLS, secrets=SECRETS,
              timeout=6 * 60 * 60, memory=256 * 1024)
def eval_vlm(ckpt: str = "", args: str = "", name: str = "", tiny: bool = False):
    """VLM eval (see eval_vlm.py). Empty ckpt = random-projector baseline. Results land in
    /outputs/ripple_vla/eval/<name>/ on the ripple-vla-outputs volume and are also returned."""
    import json
    import shlex

    d = _snapshot(full=not tiny)
    out = f"/outputs/ripple_vla/eval/{name or ('random' if not ckpt else os.path.basename(os.path.dirname(ckpt)))}"
    cmd = ["python", "eval_vlm.py", "--ckpt_dir", d, "--out", out, "--llava_zip", LLAVA_ZIP]
    cmd += (["--ckpt", ckpt] if ckpt else []) + (["--tiny"] if tiny else []) + shlex.split(args)
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root/ripple_vla")
    outputs.commit()
    return json.load(open(f"{out}/results.json"))


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
