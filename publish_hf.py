"""Publish the trained adapter (projector + LoRA) to the Hugging Face Hub, from inside Modal.

  modal run ripple_vla/publish_hf.py                       # PRIVATE repo <you>/ripple-vlm-instruct
  modal run ripple_vla/publish_hf.py --public              # public

Uses the Modal secret `huggingface` (HF_TOKEN). Needs a token with write access.
Publishes ONLY the trainable weights (~1.1 GB): the Ripple-65B and SigLIP weights are not redistributed.
"""

import os

import modal

from modal_app import SECRETS, VOLS, image

app = modal.App("ripple-vla-publish")

CARD = """---
license: apache-2.0
base_model:
- ricursiveAI/Ripple-65B-Preview
- google/siglip-so400m-patch14-224
library_name: pytorch
tags:
- vision-language
- adapter
- lora
- ripple
---

# Ripple-VLM (adapter): SigLIP + Ripple-65B-Preview

Adapter weights that give the **text-only** [Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview)
LLM the ability to read images and answer questions about them. **This repo does not contain the base models.**
You need `ricursiveAI/Ripple-65B-Preview` (~130 GB, Apache-2.0) and `google/siglip-so400m-patch14-224`.

Code, training scripts and eval harness: https://github.com/fbarbosa-stanford/vlm_ripple

## What is in here
| File | Contents |
|---|---|
| `adapter.safetensors` | `projector.*` (SigLIP -> Ripple token space, 49.8M params) and `backbone...lora_*` (rank-16 LoRA on all 7 linear layers of all 80 Ripple layers, 232M params), fp32 |
| `config.json` | the `VLAConfig` the weights were trained with (pool=2 -> 64 image tokens per image, lora_rank=16) |
| `results/` | raw eval outputs for this adapter and for the stage-1 (projector-only) model |

The flow-matching action expert that the codebase also defines was **not trained for this release** and is not included.

## How it was trained
1. **Stage 1, alignment:** projector only, Ripple and SigLIP frozen. 3,000 steps, global batch 32, on 100K
   LLaVA-Pretrain (BLIP-558K) image-caption pairs.
2. **Stage 2, visual instruction tuning:** projector + LoRA on Ripple, SigLIP frozen. 50K LLaVA-Instruct-150K
   conversations (random subset), global batch 32, learning rates 5e-5 (projector) / 1e-4 (LoRA).
   The run was **stopped early at step ~910 of 1,500; this is the step-750 checkpoint.**
Both stages ran on 8xH200 with FSDP2. Images are resized to 224x224 (64 tokens each).

## Results (200 items per task, single run, so roughly +-3 points of noise on accuracies)
Each task is scored with the right image and again with another sample's image; `gain` is the difference.

| Task | Stage 1 (projector only) | **This adapter** |
|---|---|---|
| POPE yes/no, accuracy (chance 50%) | 47.0% (gain -0.04 +- 0.04) | **74.5%** (shuffled 59.5%, gain +0.15 +- 0.04) |
| Synthetic shapes, 4-way multiple choice (chance 31%) | 62.5% | **73.5%** |
| COCO caption NLL / token (lower is better) | 3.00 | **2.78** |

Known weaknesses: counting and position are near chance, the model says "yes" too often on POPE (64%), it can
get colours wrong and invents detail in long descriptions, and 64 tokens per image cannot read small text.
Scoring here is by likelihood of fixed answers, so it does not prove open-ended replies are good. Ripple's
text-only quality after the LoRA has **not** been re-tested.

## Usage
```python
# from the GitHub repo (loading.py / data.py)
from loading import build_vla_inference
model, cfg = build_vla_inference("adapter.safetensors", "<local Ripple-65B-Preview snapshot>")
```
`config.json` must sit next to `adapter.safetensors`. Generation: `model.generate(images, input_ids, attention_mask)`,
using the chat format built by `data.Collator`. Needs roughly 140 GB of GPU memory (e.g. 2x H100 80GB).

## Licences and data: please check before reuse
The adapter is released under Apache-2.0 like its base models, but it was trained on third-party data whose terms
may be stricter: LLaVA-Pretrain (BLIP-558K, LAION/CC/SBU images) and LLaVA-Instruct-150K (conversations generated
with GPT-4 over COCO images). My understanding is that LLaVA-Instruct-150K is CC BY-NC 4.0 and subject to OpenAI's
terms of use, so **commercial use may not be permitted**; verify against the dataset cards.
"""


@app.function(image=image, cpu=4, memory=16384, volumes=VOLS, secrets=SECRETS, timeout=60 * 60)
def publish(name: str = "ripple-vlm-instruct", public: bool = False,
            ckpt: str = "/outputs/ripple_vla/instruct/last.pt"):
    import json
    import shutil
    import tempfile

    import torch
    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    VOLS["/outputs"].reload()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.detach().contiguous() for k, v in blob["state_dict"].items()
          if k.startswith("projector.") or ".lora_" in k}
    n_params = sum(v.numel() for v in sd.values())
    out = tempfile.mkdtemp()
    save_file(sd, f"{out}/adapter.safetensors")
    json.dump({"base_model": "ricursiveAI/Ripple-65B-Preview", "vision_encoder": "google/siglip-so400m-patch14-224",
               "step": blob.get("step"), "cfg": blob["cfg"]}, open(f"{out}/config.json", "w"), indent=1)
    open(f"{out}/README.md", "w").write(CARD)
    os.makedirs(f"{out}/results", exist_ok=True)
    for tag, d in (("stage1_vlm", "vlm"), ("instruct_step750", "instruct_750")):
        src = f"/outputs/ripple_vla/eval/{d}/results.json"
        if os.path.exists(src):
            shutil.copy(src, f"{out}/results/{tag}.json")

    api = HfApi()
    who = api.whoami()["name"]
    repo_id = f"{who}/{name}"
    api.create_repo(repo_id, private=not public, exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=repo_id, commit_message="Add Ripple-VLM adapter (projector + LoRA)")
    print(f"published {n_params / 1e6:.1f}M params, step {blob.get('step')} -> https://huggingface.co/{repo_id} "
          f"({'public' if public else 'PRIVATE'})", flush=True)
    return repo_id


@app.local_entrypoint()
def main(name: str = "ripple-vlm-instruct", public: bool = False):
    print(publish.remote(name=name, public=public))
