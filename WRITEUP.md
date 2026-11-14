# Ripple-VLA: Project Write-up

> **Note added after this document was written:** this write-up was drafted mid-project (for example, section 13
> describes a training run that was still in progress). The **README is the current source of truth** for what was
> trained, the final results and what has not been done.

*Last updated 2026-11-13. Author: Mark Music, building on the initial scaffold by fbarbosa-stanford.*

This document covers the whole project in one place: what we're building and why, how every piece works, what
the backbone model (Ripple-65B) is and isn't good at, how vision gets added to a text-only LLM, and the evaluation
system I built to measure whether that worked. It ends with current status, open questions and next steps.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Goal](#2-goal)
3. [The backbone: Ripple-65B-Preview](#3-the-backbone-ripple-65b-preview)
4. [Architecture](#4-architecture)
5. [How vision is added to a text-only LLM](#5-how-vision-is-added-to-a-text-only-llm)
6. [Training pipeline](#6-training-pipeline)
7. [Infrastructure: loading a 65B model on 8 GPUs](#7-infrastructure-loading-a-65b-model-on-8-gpus)
8. [Data](#8-data)
9. [The VLM evaluation system (new)](#9-the-vlm-evaluation-system-new)
10. [Verification of the eval system](#10-verification-of-the-eval-system)
11. [Random-weights baseline results](#11-random-weights-baseline-results)
12. [Robot evaluation (LIBERO)](#12-robot-evaluation-libero)
13. [Status of the alignment run](#13-status-of-the-alignment-run)
14. [How to run everything](#14-how-to-run-everything)
15. [Code changes made in this round](#15-code-changes-made-in-this-round)
16. [Known issues and gotchas](#16-known-issues-and-gotchas)
17. [Open questions](#17-open-questions)
18. [Next steps](#18-next-steps)
19. [Glossary](#19-glossary)

---

## 1. TL;DR

- **What:** a vision-language-action (VLA) model for robot manipulation. We take **Ripple-65B-Preview**, a
  text-only LLM, give it eyes with a frozen **SigLIP** vision encoder and a learned projector, and add a
  **flow-matching action expert** (pi0-style) that outputs 50-step robot action chunks.
- **Where it runs:** training on Modal with 8×H200 using FSDP2. Evaluation in LIBERO, a MuJoCo robot
  benchmark.
- **Stage 1 (align)** teaches the projector to turn images into "words" Ripple understands. A run named `vlm`
  is training now. Its loss fell from about 4.8 to about 2.5 by step 1.5k of 3k.
- **What I added:** an evaluation harness, [`eval_vlm.py`](eval_vlm.py), that measures whether visual
  information actually reaches the LLM. It scores six tasks twice: once with the correct image and once with
  another sample's image. The difference (the "gain") isolates visual signal from text priors.
- **Verified:** the harness runs end to end on a tiny random model. Its scorer matches the training loss
  exactly (difference 2.5e-7). With a random projector, every gain is within two standard errors of zero, as
  it should be.
- **Next:** run the random baseline on the real 65B model now. When the trained projector arrives, run the
  same eval on it and compare.

---

## 2. Goal

Recent VLAs like pi0 and OpenVLA start from a vision-language model that already sees, then add an action
head. This project asks a different question: **can a strong text-only model, pretrained mostly on code and
reasoning, be turned into a useful robot policy by bolting on vision and an action head?**

That requires two capabilities Ripple doesn't have:

1. **Seeing:** mapping images into Ripple's input space so its internal representations reflect what is in
   the image.
2. **Acting:** reading those representations and producing continuous robot actions.

The project splits these into two training stages. The first is evaluated with standard vision-language
benchmarks, which is what the new eval harness does. The second is evaluated by closed-loop success rate in
LIBERO.

---

## 3. The backbone: Ripple-65B-Preview

Source: [ricursive.com/ripple](https://www.ricursive.com/ripple) and the
[Hugging Face model card](https://huggingface.co/ricursiveAI/Ripple-65B-Preview). All benchmark numbers are
self-reported by Ricursive Intelligence. The model card warns that replication is hard and that harnesses
differ.

### 3.1 What it is

- Made by **Ricursive Intelligence**. It is a general-purpose variant of the proprietary model behind their AI
  chip-design platform.
- **Dense 65B** transformer, not mixture-of-experts. Instruction-tuned. Released as a "Preview".
- **License:** Apache 2.0.
- **Text only.** Pretrained on 12T tokens of mostly English text and code. It has never seen an image.

### 3.2 Architecture details (from `config.json`)

| Property | Value |
|---|---|
| Layers | 80 |
| Hidden size | 5120 |
| FFN size | 40960 |
| Attention heads | 80 query heads, 8 KV heads, head dim 128 |
| Attention pattern | 4 sliding-window layers then 1 global layer, repeating |
| Sliding window | 1024 tokens |
| Max context | about 1M tokens (2M position limit in config) |
| Vocabulary | 100,352 |
| Embedding scale | 71.554, applied inside the model to any input embeddings |
| Other | gated attention, value residuals, FFN clamping at 20, global layers use no positional encoding |

The design moved from an Olmo-3-like layout to one modeled on Gemma 3N. It is "thin and deep".

### 3.3 What it's good at

| Area | Benchmark | Score |
|---|---|---|
| Agentic terminal tasks | Terminal-Bench 2.1 | 80.5 |
| Software engineering | SWE-bench Verified | 76.5 |
| Competition math | AIME 2025 | 98.3 |
| Science QA | GPQA Diamond | 85.2 |
| Knowledge | MMLU-Pro | 84.2 |
| Instruction following | IFBench | 79.2 |

The headline claim is "U.S. open-weight state-of-the-art" on Terminal-Bench. It is not best everywhere. Other
open models lead on SWE-bench Verified, GPQA, MMLU-Pro and IFBench.

### 3.4 What that means for us

- **No visual prior at all.** The projector has to teach image understanding from scratch, not reconnect an
  existing skill. This is harder than standard LLaVA-style training on a model that saw image captions in
  pretraining.
- **Its strengths are not a robot's.** Coding, math and tool use are what it excels at. A robot policy mainly
  needs visual grounding and spatial reasoning.
- **Light post-training.** The release says post-training was "relatively light", with a focus on agentic
  coding. Behavior outside coding may be less polished.
- **Long context goes unused.** Our sequences are about 128 image tokens plus under 96 text tokens.

---

## 4. Architecture

```
            ┌─────────────┐   ┌─────────────┐
 images ──▶ │ SigLIP      │──▶│ Projector   │──┐  64 tokens per image (pool 2)
 (agentview,│ so400m      │   │ pool→LN→MLP │  │
  wrist)    │ (frozen)    │   │ →RMS·gain   │  │
            └─────────────┘   └─────────────┘  │
                                               ▼
 text ──▶ tokenizer ──▶ embed_tokens ──▶ [prefix][IMAGE TOKENS][prompt ...]
                                               │
                                               ▼
                                   ┌───────────────────────┐
                                   │ Ripple-65B (frozen,   │
                                   │ optional LoRA)        │
                                   └───────────────────────┘
                                     │                  │
                       stage align:  ▼                  ▼  stage vla:
                              lm_head → caption   hidden states (chosen layers)
                              cross-entropy                 │
                                                            ▼
                                                 ┌──────────────────────┐
                              state, noisy  ───▶ │ Action expert        │ ──▶ velocity
                              action chunk, t    │ 12 layers, width 1024│     (flow matching)
                                                 │ self + cross attn,   │
                                                 │ adaLN on flow time   │
                                                 └──────────────────────┘
```

All of this lives in [`model.py`](model.py).

### 4.1 Vision encoder

`google/siglip-so400m-patch14-224`, frozen. A 224×224 image becomes a 16×16 grid of 256 patch features, each
1152 wide.

### 4.2 Projector (`VisionProjector`)

1. **Pooling.** With pool 2, each 2×2 block of patches is stacked into one 4608-wide vector. That gives 64
   tokens per image.
2. **MLP.** LayerNorm, then a linear layer to 5120, GELU, and another linear layer to 5120.
3. **Scale matching.** The output is RMS-normalized and multiplied by a learnable per-channel gain. The gain
   starts at the mean RMS of Ripple's word-embedding rows. This matters because Ripple multiplies every input
   embedding by 71.55 internally. Image tokens must enter at raw embedding scale, or they would be 71 times
   too big or too small relative to text.

### 4.3 Sequence layout

The Ripple chat template is ChatML. Image tokens go right after a fixed prefix:

```
<|endoftext|><|im_start|>user\n [IMAGE TOKENS] {prompt}<|im_end|>\n<|im_start|>assistant\n {target}<|im_end|>
```

The prefix is identical for every sample, so the insertion point is a constant stored in the config as
`image_insert_pos`. Padding is on the right, so the default position ids work unchanged.

### 4.4 Action expert (`ActionExpert`)

- **Tokens:** one state token followed by 50 noisy action tokens.
- **Each of the 12 blocks:** self-attention, then cross-attention to the projected Ripple hidden states, then
  an MLP. Flow time modulates the block through adaLN, initialized to zero so each block starts as the
  identity.
- **Context:** hidden states from one or more Ripple layers. The default is the final post-norm layer. Each
  chosen layer gets its own LayerNorm, projection and learned layer embedding.
- **Flow matching, pi0 convention:** t=1 is noise and t=0 is data. During training, t is drawn from
  Beta(1.5, 1) scaled into [0.001, 1]. The noisy input is `x_t = t·ε + (1−t)·a`, and the target velocity is
  `ε − a`. Sampling runs 10 Euler steps from t=1 down to t=0.
- **Normalization:** actions and state are standardized with dataset statistics, which are stored as buffers
  in the checkpoint.

### 4.5 LoRA

LoRA is a minimal, FSDP-friendly implementation of low-rank adapters on the attention and MLP projections.
The init is deterministic and seeded by parameter name, so every GPU rank builds the same full tensor before
slicing its shard. It is off by default. When on, it trains only in the vla stage.

---

## 5. How vision is added to a text-only LLM

This is the LLaVA recipe.

1. **Turn images into fake words.** The projector maps SigLIP features to vectors that live in Ripple's
   word-embedding space.
2. **Splice them into the prompt.** Ripple processes them like any other token.
3. **Train with ordinary next-token loss.** On image-caption pairs, only the caption tokens are scored. The
   gradient flows back through frozen Ripple into the projector, which learns to produce embeddings that make
   Ripple predict the right caption.

### 5.1 Captioning vs. VQA

- **Stage 1 (what we have):** captioning only, with Ripple and SigLIP frozen. The projector learns the
  vocabulary of vision, but the model never practices answering questions about images. This is LLaVA's
  stage 1.
- **Stage 2 (not built yet):** visual instruction tuning. This needs a mixture like LLaVA-Instruct-665k, which
  includes VQAv2, GQA, OK-VQA, OCR and multi-turn visual chat, plus unfreezing part of the LLM, usually via
  LoRA. Frozen-LLM projectors tend to plateau at captioning.

The pieces for stage 2 mostly exist. The collator already supports prompt/answer pairs with loss only on the
answer, and LoRA exists but is only enabled in the vla stage. What's missing is a dataset class and a flag to
enable LoRA with the caption loss.

### 5.2 Why this matters for the robot

In the vla stage, Ripple doesn't generate text. The action expert reads Ripple's hidden states. So the purpose
of the VLM work is to make those hidden states reflect what's in the image. Better visual grounding upstream
should mean a better policy downstream, and the eval harness is how we check the upstream part.

---

## 6. Training pipeline

The entry point is [`train.py`](train.py), launched with `torchrun`.

### 6.1 Stage `align`

- **Trains:** the projector only, about 50M parameters.
- **Loss:** cross-entropy on caption tokens.
- **Data:** LLaVA-Pretrain, read straight from `images.zip`. You can optionally mix in LIBERO frames captioned
  with their task strings. LLaVA has one image per sample and LIBERO has two, so they use separate loaders
  that alternate batch by batch.
- **End-of-run check:** greedy captions one sample at a time. Every rank decodes the same fixed number of
  tokens with no early stop at end-of-sequence, because ranks stopping at different steps would deadlock the
  FSDP collectives.

### 6.2 Stage `vla`

- **Trains:** the projector, the action expert, and LoRA if enabled.
- **Loss:** flow matching on 50-step action chunks, with padded steps masked out.
- **Data:** the LeRobot LIBERO dataset, `HuggingFaceVLA/libero`: 1693 episodes, 273k frames at 10 fps,
  40 tasks.
- **Resumes** from the align checkpoint.

### 6.3 Optimizer

AdamW with betas 0.9 and 0.95. There are separate learning rates for LoRA and everything else. Warmup is
linear, followed by cosine decay down to 10% of the peak. Gradients are clipped at 1.0.

---

## 7. Infrastructure: loading a 65B model on 8 GPUs

[`loading.py`](loading.py) handles the memory problem. In bf16, the 65B weights are about 130 GB.

1. **Build on the meta device.** The backbone is created with no memory behind it.
2. **Shard.** `fully_shard` is applied per decoder layer, plus the embeddings, the LM head, the SigLIP layers,
   the action expert blocks and the root module.
3. **Allocate.** `to_empty(cuda)` allocates only each GPU's shard, about 16 GB each on 8 GPUs.
4. **Stream weights.** Each rank reads only its row slice of each tensor from the safetensors files via
   memory-mapping. No process ever holds the whole model in CPU RAM.
5. **Restore small modules.** SigLIP, the projector, the expert and the normalization buffers were built for
   real before sharding. They are snapshotted and copied back after `to_empty()`. That includes non-persistent
   buffers like SigLIP's position ids, which `to_empty()` would otherwise wipe.

**Checkpoints** store only trainable tensors: projector, expert, LoRA and normalization stats. They are small
and fast to save.

**Inference** uses no FSDP. Hugging Face `device_map="auto"` spreads the backbone across local GPUs, and the
small modules sit on the first GPU. Both the eval harness and the LIBERO policy server use this path.

---

## 8. Data

| Dataset | Used for | Notes |
|---|---|---|
| LLaVA-Pretrain 558k | align | BLIP captions of LAION/CC/SBU images. Read in place from `images.zip`. The `vlm` run uses the first 100k. |
| LIBERO (LeRobot) | align (optional), vla | Agentview and wrist cameras, 8-dim state, 7-dim delta-EEF actions |
| Synthetic | smoke tests | Fake data the model can learn: actions are a fixed function of state |
| POPE, COCO-2017 val, MMBench-EN dev, VQAv2 val | eval only | From the `lmms-lab-encoder` / `lmms-lab` organizations on Hugging Face |

---

## 9. The VLM evaluation system (new)

File: [`eval_vlm.py`](eval_vlm.py). Modal entry point: `modal_app.py::eval_vlm`.

### 9.1 The core question

**Does the model use the image?** A model can do well on vision benchmarks without looking. For example,
"Is there a person in the image?" is usually "yes", and a caption model can guess common words. So raw scores
alone can't tell us whether the projector works.

### 9.2 The shuffled-image control

Every task is scored under two conditions:

- **image:** each question paired with its own image.
- **shuffled:** each question paired with a *different* sample's image. The same set of images and questions
  is used, permuted so that no item keeps its own image.

**gain = score(image) − score(shuffled).** Text priors help both conditions equally, so they cancel. Anything
left over is visual information getting through. The harness also reports a **paired standard error** of the
per-item difference. A gain above two standard errors counts as real visual signal.

Predictions:

- **Random projector:** gain ≈ 0 on every task. This is the baseline.
- **Trained projector:** gain > 0. The size of the gain shows how much visual information gets through.

### 9.3 Tasks

| Task | Data | Metric | Why it's here |
|---|---|---|---|
| `shapes` | Synthetic images of colored circles, squares and triangles. Questions about color, shape, count (1–4) and left/right position. Generated with a fixed seed, no download needed. | Multiple-choice accuracy | Controlled probe with known chance level. Lets us see which visual attributes come through first. |
| `pope` | POPE: COCO images with yes/no object-presence questions in random, popular and adversarial splits | Accuracy, F1, yes-ratio | Standard hallucination benchmark. The shuffled image makes "yes" wrong for most questions. |
| `coco` | COCO-2017 val captions | Caption loss per token (lower is better), plus greedy captions to read | Closest to what align trains. Should move first. |
| `llava` | Held-out LLaVA-Pretrain items beyond index 500k (training used the first 100k) | Caption loss per token | In-distribution held-out check. Needs the zip, which is on the Modal volume. |
| `mmbench` | MMBench-EN dev, 4.3k items, 2–4 options | Multiple-choice accuracy by option likelihood | Broad perception and reasoning coverage |
| `vqav2` | VQAv2 val, streamed | VQA accuracy of short greedy answers | Open-ended. Needs instruction following, so it may stay low until stage 2. |

### 9.4 Scoring methods

- **Likelihood scoring for multiple choice.** For each option, the harness computes the log-probability of
  the option text (plus end-of-turn) as the assistant's reply. It picks the option with the highest *mean*
  log-probability per token, since options differ in length. This needs no instruction following, so it works
  on a projector-only model.
- **Caption loss.** This is the exact quantity the align stage minimizes, computed on held-out data. The
  scorer reuses the training prompt format and tokenizes the target separately, just like the collator.
- **Generation.** Greedy decoding in batches with right padding. Each step writes the new token's embedding
  into a preallocated buffer at that row's current length. There is no KV cache: the full prefix is
  recomputed each step. That's simpler and safe with Ripple's custom attention, and fine for answers of 10–32
  tokens.
- **VQA accuracy:** `min(1, matching human answers / 3)`, after a simplified version of the official answer
  normalization (lowercase, strip punctuation and articles, number words to digits).

### 9.5 Device handling for the 65B model

With `device_map="auto"`, the LM head can land on a different GPU than the embeddings. The scorer computes
hidden states, moves only the needed positions to the LM head's device, and computes logits only at target
positions. This avoids building a full vocabulary-by-sequence logit tensor.

### 9.6 Random-projector baseline

With no `--ckpt`, the loader builds the projector and action expert with fresh random init, seeded for
reproducibility. It uses the same projector gain init as training, so the baseline really is "step 0 of
training". `--pool` has to match the trained run. The friend's run uses pool 2.

### 9.7 Outputs

- `results.json` has per-task metrics for both conditions, the gain, its standard error, and runtime.
- `samples.jsonl` has every item: the prompt, the prediction under both conditions, the gold answer or
  target, and generated captions.
- A markdown summary table is printed at the end, with a "visual signal?" column.

---

## 10. Verification of the eval system

The development machine has no GPU, so everything below ran on CPU with the **tiny random Ripple**. That's
Ripple's real modeling code and tokenizer, shrunk to 5 layers and width 256, with a tiny SigLIP.

1. **Scorer matches the training loss.** For the same three image-caption pairs, `Scorer.loglik` and
   `RippleVLA.caption_loss` agree:

   ```
   caption_loss 11.65935  scorer 11.65935  diff 2.49e-07  tokens [7, 13, 3]
   ```

   The scorer used batch size 2 here, which split the batch and changed the padding. So this also confirms
   the token-position bookkeeping across padding and image insertion.

2. **Checkpoint loading works.** A train.py-format trainable-only checkpoint loads all 69 tensors with none
   missing.
3. **Batched generation runs** and stops at end-of-turn tokens.
4. **Five of six tasks ran end to end locally:** shapes, pope, coco, mmbench and vqav2. The `llava` task
   shares the caption-loss scorer with `coco` but needs the LLaVA zip on the Modal volume, so it has not run
   yet.

**Not yet verified:** running on the real 65B with `device_map` across 2 H200s, and the `llava` task on
Modal. These are the first things to run.

---

## 11. Random-weights baseline results

**Setup:** tiny random Ripple, random projector, 40 items per task, CPU. Everything in this model is random,
including the LLM, so the absolute scores mean nothing. What matters is that the gains are around zero.

| Task | Metric | Image | Shuffled | Gain | Standard error | Visual signal? |
|---|---|---|---|---|---|---|
| shapes | accuracy | 0.350 | 0.350 | +0.000 | 0.000 | no |
| pope | accuracy | 0.450 | 0.625 | −0.175 | 0.101 | no |
| coco | caption loss per token | 11.535 | 11.538 | +0.003 | 0.017 | no |
| mmbench | accuracy | 0.350 | 0.250 | +0.100 | 0.093 | no |
| vqav2 | VQA accuracy | 0.000 | 0.000 | +0.000 | 0.000 | no |

How to read this:

- **Every gain is within two standard errors of zero.** With 40 items the error bars are wide. POPE's −0.175
  is about −1.7 standard errors, which is noise. The real runs use 500 items per task, which shrinks the
  standard error by about 3.5×.
- **Shapes gain is exactly zero.** The tiny random model gives the same answer no matter which image it sees,
  so the image has no effect.
- **COCO caption loss is about 11.5,** close to ln(100k) ≈ 11.5, the value for uniform guessing over the
  vocabulary. That's expected for an untrained LLM.
- **VQAv2 scores zero.** Greedy outputs from a random LLM are gibberish, for example `'()); renalufe());...'`.

**The real baseline is still to run:** real Ripple-65B with a random projector. There, the absolute scores
will reflect Ripple's text priors (for example, POPE will be near the yes-ratio), and the gains should again
be near zero. That gives the reference to compare the trained projector against.

---

## 12. Robot evaluation (LIBERO)

[`eval_libero.py`](eval_libero.py) and [`libero_sim_modal.py`](libero_sim_modal.py). Written, not yet run on
the 65B.

- **Setup:** one GPU policy server batches requests from up to 200 CPU simulator workers, using Modal's
  dynamic batching with up to 32 requests and a 30 ms wait.
- **Conventions match the training data:** both camera images are rotated 180°. The state is EEF position,
  axis-angle orientation and two gripper positions. Actions are delta EEF plus gripper.
- **Episode protocol:** the first 10 steps are no-ops so objects settle. Each policy call returns a 50-step
  chunk, of which the first 5 actions are executed before replanning. Episode limits follow openpi:
  220 / 280 / 300 / 520 steps for spatial / object / goal / libero_10.
- **Rendering:** headless MuJoCo with OSMesa on CPU.

---

## 13. Status of the alignment run

The friend's `vlm` align run uses pool 2, the first 100k LLaVA items, batch size 4 per GPU, 3000 steps, on
8×H200.

| Step | Loss (from W&B) |
|---|---|
| 0 | about 4.8 |
| 250 | about 3.0 |
| 910 | 2.67 |
| 1.5k | about 2.5 |

The curve is still slowly decreasing. Caption loss around 2.5 nats per token suggests the projector carries
real information. A frozen LLM with no visual signal would sit near its text-only caption loss. The eval will
show how much of that is visual, by comparing the trained projector's gain on `coco` and `llava` against the
random-projector baseline.

---

## 14. How to run everything

All Modal commands are run from the parent directory, as in the README.

```bash
# one-time
modal run ripple_vla/modal_app.py::fetch_weights           # Ripple-65B -> hf-cache volume

# smoke test the training loop (tiny model, 2 GPUs)
modal run ripple_vla/modal_app.py::smoke --stage align

# VLM eval: pipeline test, random baseline, trained projector
modal run ripple_vla/modal_app.py::eval_vlm --tiny --args "--n 32 --tasks shapes,pope"
modal run ripple_vla/modal_app.py::eval_vlm --name random
modal run ripple_vla/modal_app.py::eval_vlm --ckpt /outputs/ripple_vla/vlm/last.pt --name vlm

# all tasks including held-out LLaVA, bigger sample
modal run ripple_vla/modal_app.py::eval_vlm --ckpt /outputs/ripple_vla/vlm/last.pt --name vlm_full \
  --args "--tasks shapes,pope,coco,llava,mmbench,vqav2 --n 1000"
```

If the checkpoint arrives as a file instead of on the Modal volume, upload it first:

```bash
modal volume put ripple-vla-outputs last.pt /ripple_vla/vlm/last.pt
```

Run locally on CPU with the tiny model, given a folder holding Ripple's config, code and tokenizer:

```bash
python eval_vlm.py --ckpt_dir path/to/ripple_cfg --tiny --device cpu --tasks shapes,pope --n 32
```

Results go to `/outputs/ripple_vla/eval/<name>/` on the `ripple-vla-outputs` volume. The Modal function also
returns the results.

---

## 15. Code changes made in this round

| File | Change |
|---|---|
| `eval_vlm.py` | **New.** The eval harness: six tasks, likelihood and generation scorers, shuffled-image control, paired standard errors, JSON and JSONL outputs. |
| `loading.py` | `build_vla_inference` now accepts no checkpoint, which builds a random-projector baseline from a given config. It takes a `device` argument so it can run on CPU. It computes the projector gain init from the real embeddings, matching training. The existing LIBERO eval calls it the same way as before. |
| `modal_app.py` | New `eval_vlm` Modal function on H200:2, configurable via `RIPPLE_EVAL_GPU`, plus usage lines in the docstring. |
| `README.md` | New eval row in the file table, plus a section on running and reading the eval. |
| `WRITEUP.md` | This document. |

---

## 16. Known issues and gotchas

- **Embedding scale.** Ripple multiplies input embeddings by 71.55 internally. Image tokens must be at raw
  embedding scale, and the projector's gain init handles this. Any new input path has to respect it too.
- **FSDP generation.** Every rank must run the same number of forward passes, so there's no per-rank early
  stopping.
- **Dataset loading.** `load_dataset(repo, split=...)` downloads *all* splits of some repos. COCO-Caption2017
  would pull 6.6 GB of test images to get the 0.8 GB val split. The harness avoids this by reading only the
  requested split's parquet files.
- **Streaming shuffle buffers are slow.** A 1000-row buffer of images took minutes locally. Only VQAv2 is
  streamed, with a small buffer. The others download once and are cached on the Modal volume.
- **Remote code prompt.** Loading Ripple's tokenizer from a local folder can trigger an interactive
  trust-remote-code prompt, even with the environment variable set. The harness passes the flag explicitly.
- **README paths.** The commands use a `ripple_vla/` path from the parent directory, but the repo folder is
  `vlm_ripple`. Either rename or symlink, or adjust the paths.
- **Inference with LoRA.** LoRA adapters are created on each layer's device under `device_map`. This is
  untested at 65B scale.
- **Generation speed.** With no KV cache, decoding is quadratic in output length. That's fine for short
  answers but not for long-form generation.

---

## 17. Open questions

1. **How much of the align loss drop is visual?** The `coco` and `llava` gains will answer this.
2. **Will a frozen Ripple answer questions about images at all,** or does it need visual instruction tuning
   first? If POPE, MMBench and VQAv2 gains stay near zero while caption gains are large, stage 2 is needed.
3. **Which Ripple layers should feed the action expert?** Features from middle layers are often more spatial
   than the final layer. `--feature_layers 40,-1` is supported but untested.
4. **Is pool 2 too aggressive?** Going from 256 to 64 tokens per image might lose fine spatial detail that
   manipulation needs. The `shapes` position and count probes give a cheap first signal.
5. **Does Ripple's coding and reasoning strength transfer to manipulation at all,** compared with a
   same-size VLM backbone?

---

## 18. Next steps

1. **Run the real random baseline now:**
   `modal run ripple_vla/modal_app.py::eval_vlm --name random`. It doesn't need the trained weights, and it
   also tests the 65B `device_map` path.
2. **When the checkpoint arrives,** run the same command with `--ckpt` and `--name vlm`. Compare gains against
   the baseline.
3. **Evaluate intermediate checkpoints** if the friend saves them every 500 steps. That gives a curve of
   visual gain over training.
4. **If question-answering gains stay flat,** build stage 2: an instruction-tuning dataset class (for example
   LLaVA-Instruct-665k) and LoRA enabled for the caption loss.
5. **Then run the vla stage and LIBERO eval** on the 65B.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| VLA | Vision-language-action model: sees, reads instructions, and outputs robot actions |
| VLM | Vision-language model: sees and reads, outputs text |
| VQA | Visual question answering |
| SigLIP | A contrastively trained vision encoder from Google, used frozen here |
| Projector | The small learned network that maps vision features into the LLM's embedding space |
| Flow matching | A generative training objective that learns a velocity field carrying noise to data |
| Action chunk | A sequence of future actions predicted at once, 50 steps here |
| FSDP2 | PyTorch's fully sharded data parallel, which splits parameters across GPUs |
| LoRA | Low-rank adapters: small trainable matrices added to frozen weight matrices |
| POPE | Polling-based Object Probing Evaluation, a yes/no hallucination benchmark |
| LIBERO | A suite of simulated tabletop manipulation tasks built on MuJoCo and robosuite |
| Gain | In this eval: score with the correct image minus score with a shuffled image |
