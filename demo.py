"""Upload an image + type a prompt -> the aligned Ripple VLM replies. Gradio UI served from Modal.

  modal serve ripple_vla/demo.py                 # temporary URL (prints it); Ctrl-C to stop
  modal serve ripple_vla/demo.py  (env RIPPLE_DEMO_CKPT=/outputs/ripple_vla/vlm/last.pt to change checkpoint)

Notes
- Stage-1 model (projector-only on 100K LLaVA captions): it captions well but is NOT instruction-tuned.
  The default prompt matches training; free-form questions are answered like captions, often ignoring the question.
- Cold start loads the 65B across 2 GPUs (~2-4 min); the container stays warm 15 min after the last request.
- `modal serve` is ephemeral. A permanent public endpoint (`modal deploy`) exposes a 2xH200 app to anyone with
  the URL, so add auth before deploying.
"""

import os
import threading

import modal

RIPPLE = "ricursiveAI/Ripple-65B-Preview"
CKPT = os.environ.get("RIPPLE_DEMO_CKPT", "/outputs/ripple_vla/vlm/last.pt")
SRC = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.10.0", "transformers>=5.0", "accelerate", "safetensors", "pillow", "sentencepiece",
                 "hf_transfer", "gradio>=5.0", "fastapi[standard]")
    .env({"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "HF_TRUST_REMOTE_CODE": "1", "PYTHONUNBUFFERED": "1", "RIPPLE_DEMO_CKPT": CKPT})
    .add_local_dir(SRC, "/root/ripple_vla", ignore=["__pycache__", "*.pyc", "*.pt", "*.npy", "demo_out", "eval_videos"])
)

app = modal.App(os.environ.get("RIPPLE_DEMO_APP", "ripple-vlm-demo"))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("ripple-vla-outputs", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/outputs": outputs}


@app.cls(image=image, gpu=os.environ.get("RIPPLE_DEMO_GPU", "H200:2"), volumes=VOLS,
         secrets=[modal.Secret.from_name("huggingface")], timeout=2 * 60 * 60, scaledown_window=900,
         max_containers=1)
@modal.concurrent(max_inputs=8)
class Demo:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/root/ripple_vla")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from data import Collator
        from loading import build_vla_inference

        outputs.reload()
        d = snapshot_download(RIPPLE)
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model, _ = build_vla_inference(CKPT, d)
        self.collate = Collator(self.tok)
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.torch = torch
        self.lock = threading.Lock()  # one generation at a time on the shared GPUs

    def reply(self, pil_image, prompt: str, max_new_tokens: int = 60) -> str:
        import numpy as np
        from PIL import Image

        if pil_image is None:
            return "Upload an image first."
        torch = self.torch
        img = pil_image.convert("RGB").resize((224, 224), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        b = self.collate([{"images": t[None], "prompt": (prompt or "Describe the image concisely.").strip()}])
        imgs, ids, att = b["images"].cuda(), b["input_ids"].cuda(), b["attention_mask"].cuda()
        with self.lock, torch.inference_mode():
            try:  # KV-cached HF generate
                out = self.model.generate(imgs, ids, att, max_new_tokens=int(max_new_tokens), eos_id=self.eos)
                toks = out[0].tolist()
            except Exception as e:  # noqa: BLE001  fall back to the slow no-cache loop
                print(f"[demo] generate() failed ({type(e).__name__}: {e}); using slow fallback", flush=True)
                toks = self.model.caption(imgs, ids, att, max_new_tokens=min(int(max_new_tokens), 40),
                                          eos_ids=(self.eos,))
        toks = toks[: toks.index(self.eos)] if self.eos in toks else toks
        return self.tok.decode(toks, skip_special_tokens=True).strip() or "(empty reply)"

    @modal.asgi_app()
    def ui(self):
        import gradio as gr
        from fastapi import FastAPI

        with gr.Blocks(title="Ripple VLM") as demo:
            gr.Markdown(
                "# Ripple-65B + SigLIP\nStage-1 vision alignment (projector-only, 100K LLaVA captions). "
                "It describes images; it is **not instruction-tuned**, so questions are often answered "
                "like captions.")
            with gr.Row():
                with gr.Column():
                    im = gr.Image(type="pil", label="Image")
                    pr = gr.Textbox(value="Describe the image concisely.", label="Prompt")
                    mt = gr.Slider(8, 120, value=60, step=4, label="Max new tokens")
                    go = gr.Button("Reply", variant="primary")
                out = gr.Textbox(label="Ripple's reply", lines=6)
            go.click(self.reply, [im, pr, mt], out)
            pr.submit(self.reply, [im, pr, mt], out)
        return gr.mount_gradio_app(FastAPI(), demo, path="/")
