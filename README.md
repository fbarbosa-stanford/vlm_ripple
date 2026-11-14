# vlm_ripple

SigLIP vision + [Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) (text-only LLM) + a
flow-matching action head, trained on Modal and evaluated on LIBERO (MuJoCo). Work in progress.

## Pieces
| File | What it does |
|---|---|
| `model.py` | SigLIP projector (64 tokens/image at `--pool 2`), image-token splice into Ripple's embeddings, flow-matching action expert, minimal LoRA |
| `loading.py` | Meta-device build + FSDP2 sharding + rank-local streaming of Ripple weights from safetensors; trainable-only checkpoints; device-mapped inference loader |
| `data.py` | LLaVA-Pretrain (reads `images.zip` in place), LIBERO captions, LIBERO VLA chunks (LeRobot `HuggingFaceVLA/libero`), synthetic data for smoke tests |
| `train.py` | `torchrun` entrypoint for two stages: `align` (caption loss, trains projector) and `vla` (flow-matching loss, trains action expert + LoRA) |
| `modal_app.py` | Modal launcher: `smoke`, `fetch_weights`, `fetch_llava`, `check_llava`, `train` |
| `libero_sim_modal.py` | Headless LIBERO (robosuite + OSMesa) image and smoke test |
| `eval_vlm.py` | Vision-language eval of the projector: synthetic shapes, POPE, COCO caption NLL, held-out LLaVA NLL, MMBench, VQAv2. Every task is scored with the right image and with a shuffled image; the gap is the visual signal |
| `eval_libero.py` | Closed-loop LIBERO eval: one GPU policy server, many CPU sim workers (written, not yet run) |

## Usage (from the parent directory)
```bash
modal run ripple_vla/modal_app.py::smoke --stage vla       # tiny random Ripple, 2 GPUs, synthetic data
modal run ripple_vla/modal_app.py::fetch_weights           # Ripple-65B -> hf-cache volume
RIPPLE_GPU=H200:8 modal run --detach ripple_vla/modal_app.py::train --stage align --name vlm \
  --args="--llava_zip <path to images.zip> --llava_limit 100000 --pool 2 --steps 3000 --bs 4"
```
Needs Modal secrets `huggingface` and `wandb`.

### Evaluating the vision projector
```bash
modal run ripple_vla/modal_app.py::eval_vlm --tiny --args "--n 32 --tasks shapes,pope"     # pipeline test (~minutes)
modal run ripple_vla/modal_app.py::eval_vlm --name random                                  # random-projector baseline, 65B
modal run ripple_vla/modal_app.py::eval_vlm --ckpt /outputs/ripple_vla/vlm/last.pt --name vlm
```
Add `--args "--tasks shapes,pope,coco,llava,mmbench,vqav2 --n 500"` to change tasks or size. Results go to
`/outputs/ripple_vla/eval/<name>/results.json` plus per-item `samples.jsonl` (generated captions included).
For a checkpoint received outside Modal, upload it first: `modal volume put ripple-vla-outputs last.pt /ripple_vla/vlm/last.pt`.

How to read it: `gain` = score with the right image minus score with another sample's image, with a paired
standard error. Text priors score the same either way, so a random projector gives gain ~ 0 on every task;
gain > 2 SE means visual information reaches the LLM. Expect caption NLL (coco, llava) and shapes to move
first; POPE, MMBench and VQAv2 need instruction following and may stay near the baseline until a
visual instruction-tuning stage exists.

## Notes
- Ripple multiplies `inputs_embeds` by `embed_scale` itself, so image tokens are spliced at raw embedding scale.
- FSDP: every rank must run the same number of forwards (no per-rank early stopping in generation).
- Status: tiny-model smoke tests pass for both stages; 65B alignment verified at ~1.5 s/step on 8xH200;
  the `vla` stage and `eval_libero.py` have not been run on the 65B yet.
