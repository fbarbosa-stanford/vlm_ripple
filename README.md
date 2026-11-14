# vlm_ripple: giving a text-only 65B LLM eyes

This repo adds vision to **[Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview)** (a text-only
LLM) by training a small adapter, and contains the scaffolding for a robot-action head that has **not** been trained
yet. Everything runs on [Modal](https://modal.com). It is a hackathon project: it works, it is small-scale, and the
limits are listed honestly below.

**Status in one paragraph.** The vision-language part is trained and evaluated: Ripple can describe images and answer
simple questions about them (POPE yes/no accuracy 47% -> 74.5% after instruction tuning). The robot (VLA) part, a
flow-matching action head on LIBERO, is written and passes tests on a tiny random model, but was **never trained or
evaluated on the 65B model**. There is no trained robot policy.

## What we trained, exactly

Ripple's own 65B weights and the SigLIP weights were **never changed**. We trained two small things on top.

| | Frozen | Trained |
|---|---|---|
| Ripple-65B-Preview | all 65B weights | only the LoRA below (stage 2) |
| SigLIP so400m (224 px, ~400M params) | everything | nothing |
| Projector (49.8M params) | | stages 1 and 2 |
| LoRA, rank 16 on q/k/v/o/gate/up/down of all 80 layers (232M params) | | stage 2 |

**How an image reaches the LLM.** SigLIP turns a 224x224 image (resized, aspect ratio not kept) into a 16x16 grid of
1,152-d patch features. Each 2x2 block is merged (4,608-d), and the **projector** (LayerNorm -> Linear 4608->5120 ->
GELU -> Linear 5120->5120 -> RMS-normalise x a learned gain) maps it into Ripple's 5,120-d token space. Result: **64
image tokens per image**, spliced into the prompt right after `<|endoftext|><|im_start|>user\n`. Ripple multiplies
its input embeddings by `embed_scale` (~71.55) internally, so image tokens are inserted at raw embedding scale and the
projector gain starts at the mean RMS of Ripple's token embeddings.

### Stage 1: alignment (run name `vlm`): projector only
- **Data:** the first **100,000** of the 558,788 LLaVA-Pretrain (BLIP) image-caption pairs. The rest were not used.
- **Task:** given the image and a "describe the image" style instruction, predict the caption. Loss on caption tokens only.
- **Settings:** 3,000 steps, global batch 32 (4/GPU x 8), AdamW (betas 0.9/0.95, no weight decay), lr 1e-3 with 100 warmup
  steps and cosine decay to 1e-4, grad-clip 1.0, bf16 + FSDP2 on 8xH200, ~1.45 s/step (~75 min). Final loss ~2.4.

### Stage 2: visual instruction tuning (run name `instruct`): projector + LoRA
- **Data:** a random **50,000**-conversation subset of LLaVA-Instruct-150K (seed 0), multi-turn visual chat over COCO train2017
  images (avg 2.3 Q&A turns). Loss on the assistant replies only; max 320 text tokens.
- **Settings:** resumed from the stage-1 projector. lr 5e-5 (projector) / 1e-4 (LoRA), 50 warmup steps, cosine decay,
  global batch 32, ~2.6 s/step on 8xH200.
- **It was stopped early:** planned 1,500 steps, stopped at ~910 to save credits. The **step-750 checkpoint** is the one
  evaluated and published. Reply-token loss went 1.72 -> ~1.10 and then flattened.
- Optimizer state is not saved in checkpoints, so resuming restarts AdamW.

### Not trained
- **The robot action head (`vla` stage) and everything under "VLA / LIBERO" below.** Only the tiny-model smoke test ran.
- No LIBERO data went into the released model. Short LIBERO-caption test runs were stopped without keeping a checkpoint.
- Ripple's text-only quality after the LoRA was **not re-tested**.

## Results
Mark Music's harness (`eval_vlm.py`) scores every task twice, with the right image and with another sample's image;
`gain` = difference, so text priors cancel out. 200 items per task, single run, so accuracies carry roughly +-3 points
of noise. Raw outputs are in `results/`.

| Task | Stage 1 (projector only) | Stage 2, step 750 |
|---|---|---|
| POPE yes/no accuracy (chance 50%) | 47.0%; shuffled 51.0%; gain -0.04 +- 0.04 (image unused) | **74.5%**; shuffled 59.5%; gain **+0.15 +- 0.04** |
| Synthetic shapes, 4-way multiple choice (chance 31%) | 62.5% | **73.5%** (colour 100%, shape 100%, count 46%, position 48%) |
| COCO caption NLL per token (lower is better) | 3.00 | **2.78** |

What this does and does not show: after stage 2 the image clearly influences yes/no and multiple-choice answers.
Counting and position are still near chance, the model still says "yes" 64% of the time on POPE (accuracy is well below
strong VLMs), and 64 tokens per image cannot read small text. These tasks score the likelihood of fixed answers; they do
**not** prove open-ended replies are good. On one hand-picked photo (a Diet Coke can) the model named the object,
bottle count and a person correctly, but called the gold can "silver" and invented extra people in a long description.

## Using the trained adapter
- **Hugging Face (private for now):** https://huggingface.co/fbarbosa1/ripple-vlm-instruct holds only the adapter
  (`adapter.safetensors`, 281.8M params, step 750) plus a model card. Base models are not redistributed.
- **Demo (upload an image + a prompt):** a Gradio app served from Modal. Needs ~140 GB of GPU memory; 2xH100 80GB works.
  Cold start (loading 65B) takes a few minutes; it scales to zero when idle.
  ```bash
  RIPPLE_DEMO_CKPT=/outputs/ripple_vla/instruct/last.pt RIPPLE_DEMO_GPU=H100:2 modal serve ripple_vla/demo.py
  ```
  `modal serve` is temporary (stops with the command). Do not `modal deploy` it publicly without adding auth.

## Repo layout
| File | What it does |
|---|---|
| `model.py` | projector, image-token splice, LoRA, the (untrained) flow-matching action expert, loss functions |
| `loading.py` | builds the 65B on the meta device, shards with FSDP2, streams weights from safetensors per rank; trainable-only checkpoints; device-mapped inference loader (also reads the HF safetensors release) |
| `data.py` | LLaVA-Pretrain, LLaVA-Instruct (images read in place from zips), LIBERO datasets, synthetic test data, chat collator |
| `train.py` | `torchrun` entrypoint; stages `align`, `instruct`, `vla` |
| `modal_app.py` | Modal launcher: `train`, `smoke`, `fetch_weights`, `fetch_llava`, `check_llava`, `fetch_instruct`, `check_instruct`, `eval_vlm` |
| `eval_vlm.py` | vision-language eval with the shuffled-image control (Mark Music) |
| `demo.py`, `publish_hf.py` | upload demo; publish the adapter to the Hub |
| `libero_sim_modal.py` | headless LIBERO (robosuite + MuJoCo, OSMesa) image + smoke test; **verified to run** |
| `eval_libero.py` | closed-loop LIBERO eval with batched policy server; **written but never run** |
| `WRITEUP.md` | Mark Music's longer write-up; written mid-project, so its status sections may be out of date; this README is current |

## Reproducing
Run from the parent directory of `ripple_vla/`. Needs Modal secrets `huggingface` and `wandb`.
```bash
modal run ripple_vla/modal_app.py::smoke --stage instruct            # tiny random Ripple, 2 GPUs, seconds-minutes
modal run ripple_vla/modal_app.py::fetch_weights                     # Ripple-65B -> hf-cache volume (~130 GB)
modal run --detach ripple_vla/modal_app.py::fetch_llava              # (see gotcha below: do not extract images)
modal run --detach ripple_vla/modal_app.py::fetch_instruct           # LLaVA-Instruct json + COCO train2017.zip (new volume)

# stage 1 (LLaVA zip is read in place from the hf-cache volume)
RIPPLE_GPU=H200:8 modal run --detach ripple_vla/modal_app.py::train --stage align --name vlm \
  --args="--llava_zip <path to images.zip> --llava_limit 100000 --pool 2 --steps 3000 --warmup 100 --bs 4 --lr 1e-3"
# stage 2
RIPPLE_GPU=H200:8 modal run --detach ripple_vla/modal_app.py::train --stage instruct --name instruct \
  --args="--resume /outputs/ripple_vla/vlm/last.pt --instruct_json /coco/llava_instruct_150k.json \
          --instruct_zip /coco/train2017.zip --instruct_limit 50000 --pool 2 --lora_rank 16 --max_text_len 320 \
          --steps 1500 --warmup 50 --bs 4 --lr 5e-5 --lr_lora 1e-4"
# eval
modal run --detach ripple_vla/modal_app.py::eval_vlm --ckpt /outputs/ripple_vla/instruct/last.pt --name instruct \
  --args "--tasks shapes,pope,coco --n 200"
```
Use `--detach` for anything long: an attached `modal run` dies if your laptop loses its connection.

## VLA / LIBERO (not done)
`train.py --stage vla` trains a ~300M-parameter flow-matching action expert (pi0-style, 50-step action chunks) that
cross-attends to Ripple's hidden states, plus LoRA, on LIBERO demos (`HuggingFaceVLA/libero`, 1,693 episodes);
`eval_libero.py` would then run closed-loop rollouts on Modal. Both pass on a tiny random model only. Expect debugging
before this works on the 65B, and no success rate exists yet.

## Gotchas we hit
- **Modal volumes cap the number of files.** Extracting LLaVA-Pretrain's ~558K images onto a volume failed with "No
  space left on device", so images are read in place from the `.zip` instead (`--llava_zip`, `--instruct_zip`).
- **FSDP needs lock-step.** Every rank must run the same number of forward passes; generation with per-rank early
  stopping on EOS deadlocked NCCL. Training-time caption samples use a fixed token count.
- **`to_empty()` wipes non-persistent buffers** (e.g. SigLIP `position_ids`); `loading.py` snapshots and restores them.
  `fully_shard()` replaces parameters, so the optimizer's parameter list must be rebuilt after sharding.
- Environment: transformers >= 5, torch 2.10, lerobot 0.5.1 (see `modal_app.py`). Ripple needs `trust_remote_code`.

## Licences and data
The code and adapter are Apache-2.0, like the base models. The adapter was trained on third-party data with possibly
stricter terms: LLaVA-Pretrain (BLIP-558K, LAION/CC/SBU images) and LLaVA-Instruct-150K (GPT-4-generated conversations
over COCO images). My understanding is LLaVA-Instruct-150K is CC BY-NC 4.0 and subject to OpenAI's terms, so commercial
use may not be permitted. **Check the dataset cards before reuse.**

## Credits
Vision scaffold, training and demo: fbarbosa-stanford, with Claude. Evaluation harness and write-up: Mark Music.
