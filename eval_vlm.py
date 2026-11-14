"""Vision-language eval for Ripple-VLA's image projector (the `align` stage).

  python eval_vlm.py --ckpt_dir $RIPPLE --ckpt /outputs/ripple_vla/vlm/last.pt --tasks shapes,pope,coco,mmbench,vqav2
  python eval_vlm.py --ckpt_dir $RIPPLE                       # no --ckpt: RANDOM projector baseline
  python eval_vlm.py --ckpt_dir $RIPPLE_CFG_ONLY --tiny --device cpu --tasks shapes --n 16   # pipeline test

The question every task answers: does the model use the image? Each task is scored twice:
  - "image":    every question paired with its own image
  - "shuffled": every question paired with ANOTHER sample's image (same images, same questions)
`gain` = image - shuffled. Text priors (e.g. "is there a person?" -> yes) score equally in both, so a
random projector should give gain ~ 0, and any positive gain is visual information getting through.

Tasks (all scored without any instruction tuning, so they work on a projector-only model):
  shapes   synthetic colour/shape/count/position multiple choice; no download, chance is known
  pope     lmms-lab POPE object-hallucination yes/no (COCO images): acc, F1, yes-ratio
  coco     COCO-2017 val caption NLL per token (lower is better) + a few greedy captions to read
  llava    held-out LLaVA-Pretrain caption NLL (items past --llava_offset; needs --llava_zip)
  mmbench  MMBench-EN dev multiple choice, scored by per-token log-likelihood of each option
  vqav2    VQAv2 val open-ended, greedy short answers, VQA accuracy (needs some instruction following)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from data import ASSISTANT, EOT, PREFIX, image_insert_pos

IMG_SIZE = 224


# --------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------
def pil_to_tensor(img) -> torch.Tensor:
    """PIL image -> (1, 3, 224, 224) float in [0, 1] (one image per sample, like LLaVA-Pretrain)."""
    from PIL import Image

    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float()[None] / 255.0


# --------------------------------------------------------------------------------------
# Tasks. Each returns a list of items; every item has "image" (1,3,H,W) and "prompt", plus:
#   kind "mc":  "options" (list[str]), "answer" (index)      -> accuracy by option log-likelihood
#   kind "nll": "target" (str)                               -> NLL per target token
#   kind "gen": "answers" (list[str] of human answers)       -> VQA accuracy of greedy answer
# --------------------------------------------------------------------------------------
COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (240, 210, 30),
          "purple": (140, 60, 180), "orange": (245, 140, 20)}
SHAPES = ("circle", "square", "triangle")
COUNTS = ("one", "two", "three", "four")


def _draw(draw, shape, cx, cy, r, fill):
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)


def task_shapes(n: int, seed: int, **_):
    """Controlled probe: if the projector carries any visual signal, these should beat chance first."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    items = []
    for i in range(n):
        q = ("color", "shape", "count", "position")[i % 4]
        color, shape = rng.choice(list(COLORS)), rng.choice(SHAPES)
        count = rng.randint(1, 4) if q == "count" else 1
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        d = ImageDraw.Draw(img)
        r = 22
        if q == "position":
            side = rng.choice(("left", "right"))
            cx = rng.randint(r + 5, IMG_SIZE // 2 - r - 10) if side == "left" else \
                rng.randint(IMG_SIZE // 2 + r + 10, IMG_SIZE - r - 5)
            _draw(d, shape, cx, rng.randint(r + 5, IMG_SIZE - r - 5), r, COLORS[color])
        else:
            cells = rng.sample([(x, y) for x in range(3) for y in range(3)], count)  # 3x3 grid: no overlap
            for gx, gy in cells:
                cx, cy = 37 + gx * 75 + rng.randint(-8, 8), 37 + gy * 75 + rng.randint(-8, 8)
                _draw(d, shape, cx, cy, r, COLORS[color])
        if q == "color":
            prompt, opts, ans = f"What color is the {shape} in the image?", list(COLORS), list(COLORS).index(color)
        elif q == "shape":
            prompt, opts, ans = "What shape is shown in the image?", list(SHAPES), SHAPES.index(shape)
        elif q == "count":
            prompt, opts, ans = f"How many {shape}s are in the image?", list(COUNTS), count - 1
        else:
            prompt, opts, ans = f"Is the {shape} on the left or the right side of the image?", ["left", "right"], \
                ("left", "right").index(side)
        items.append({"image": pil_to_tensor(img), "prompt": prompt + " Answer with one word.",
                      "options": opts, "answer": ans, "kind": "mc", "group": q})
    return items


def _hf_rows(repo: str, split: str, n: int, seed: int, stream: bool = False):
    """n rows after a seeded shuffle. Reads only this split's parquet files (load_dataset(repo, split=...)
    would fetch every split, e.g. 6.6 GB of COCO test). Small sets are downloaded once and cached under
    HF_HOME (the hf-cache volume on Modal); stream=True for big ones (VQAv2 val is 7 GB)."""
    from datasets import load_dataset

    files = f"hf://datasets/{repo}/data/{split}-*.parquet"
    ds = load_dataset("parquet", data_files={split: files}, split=split, streaming=stream)
    if not stream:
        ds = ds.shuffle(seed=seed)
        return [ds[i] for i in range(min(n, len(ds)))]
    out = []
    for row in ds.shuffle(seed=seed, buffer_size=max(200, 2 * n)):
        out.append(row)
        if len(out) >= n:
            break
    return out


def task_pope(n: int, seed: int, **_):
    items = []
    for r in _hf_rows("lmms-lab-encoder/POPE", "test", n, seed):
        items.append({"image": pil_to_tensor(r["image"]), "prompt": f"{r['question']} Answer yes or no.",
                      "options": ["Yes", "No"], "answer": 0 if r["answer"].strip().lower() == "yes" else 1,
                      "kind": "mc", "group": r["category"]})
    return items


def task_coco(n: int, seed: int, **_):
    items = []
    for r in _hf_rows("lmms-lab-encoder/COCO-Caption2017", "val", n, seed):
        caps = r["answer"]
        items.append({"image": pil_to_tensor(r["image"]), "prompt": "Describe the image briefly.",
                      "target": caps[seed % len(caps)].strip(), "refs": caps, "kind": "nll"})
    return items


def task_llava(n: int, seed: int, llava_zip: str | None = None, llava_offset: int = 500_000, **_):
    """In-distribution held-out captions. The `vlm` run trained on the first --llava_limit items."""
    assert llava_zip, "llava task needs --llava_zip"
    from data import LlavaPretrain

    ds = LlavaPretrain(zip_path=llava_zip)
    idx = random.Random(seed).sample(range(llava_offset, len(ds)), n)
    items = []
    for i in idx:
        s = ds[i]
        items.append({"image": s["images"], "prompt": s["prompt"], "target": s["target"], "kind": "nll"})
    return items


def task_mmbench(n: int, seed: int, **_):
    items = []
    for r in _hf_rows("lmms-lab/MMBench_EN", "dev", n, seed):
        opts, letters = [], []
        for L in "ABCD":
            v = r.get(L)
            if v is not None and str(v).strip().lower() not in ("", "nan", "none"):
                opts.append(str(v).strip())
                letters.append(L)
        if r["answer"] not in letters:
            continue
        hint = str(r.get("hint") or "").strip()
        q = (hint + "\n" if hint and hint.lower() != "nan" else "") + r["question"].strip()
        items.append({"image": pil_to_tensor(r["image"]), "prompt": q, "options": opts,
                      "answer": letters.index(r["answer"]), "kind": "mc", "group": r.get("l2-category", "")})
    return items


def task_vqav2(n: int, seed: int, **_):
    items = []
    for r in _hf_rows("lmms-lab-encoder/VQAv2", "validation", n, seed, stream=True):
        items.append({"image": pil_to_tensor(r["image"]),
                      "prompt": f"{r['question']}\nAnswer the question using a single word or phrase.",
                      "answers": [a["answer"] for a in r["answers"]], "kind": "gen", "group": r["answer_type"]})
    return items


TASKS = {"shapes": task_shapes, "pope": task_pope, "coco": task_coco, "llava": task_llava,
         "mmbench": task_mmbench, "vqav2": task_vqav2}


# --------------------------------------------------------------------------------------
# VQA answer normalisation (simplified version of the official VQAv2 script)
# --------------------------------------------------------------------------------------
_NUM = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ART = {"a", "an", "the"}


def vqa_norm(s: str) -> str:
    s = s.lower().replace("\n", " ").strip()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^\w\s']", " ", s)
    words = [_NUM.get(w, w) for w in s.split() if w not in _ART]
    return " ".join(words)


def vqa_acc(pred: str, answers: list[str]) -> float:
    p = vqa_norm(pred)
    return min(1.0, sum(vqa_norm(a) == p for a in answers) / 3.0)


# --------------------------------------------------------------------------------------
# Scoring against the model
# --------------------------------------------------------------------------------------
class Scorer:
    def __init__(self, model, tok, bs: int):
        self.m, self.tok, self.bs = model, tok, bs
        self.prefix = tok(PREFIX, add_special_tokens=False).input_ids
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        eos = model.backbone.config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tok.convert_tokens_to_ids(EOT)}
        self.dev = model.lm.embed_tokens.weight.device
        self.head = model.backbone.lm_head
        assert model.projector.fc1.weight.device == self.dev, "projector and embed_tokens must share a device"

    def _ids(self, prompt: str, target: str | None):
        ids = self.prefix + self.tok(prompt + ASSISTANT, add_special_tokens=False).input_ids
        n_prompt = len(ids)
        if target is not None:
            ids = ids + self.tok(target + EOT, add_special_tokens=False).input_ids
        return ids, n_prompt

    def _batch(self, images, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s)
            att[b, : len(s)] = 1
        imgs = torch.stack(images).to(self.dev)
        return self.m.build_inputs(imgs, ids.to(self.dev), att.to(self.dev))

    @torch.inference_mode()
    def loglik(self, rows):
        """rows: list of (image, prompt, target). Returns list of (sum logprob, n target tokens)."""
        out = []
        for i in range(0, len(rows), self.bs):
            chunk = rows[i:i + self.bs]
            enc = [self._ids(p, t) for _, p, t in chunk]
            embeds, mask, n_img = self._batch([r[0] for r in chunk], [e[0] for e in enc])
            [h], _ = self.m.run_backbone(embeds, mask, (-1,))
            # text token j sits at sequence position j + n_img (all targets come after the image) and is
            # predicted from position j + n_img - 1
            bi, pi, tgt, owner = [], [], [], []
            for b, (ids, n_prompt) in enumerate(enc):
                for j in range(n_prompt, len(ids)):
                    bi.append(b), pi.append(j + n_img - 1), tgt.append(ids[j]), owner.append(b)
            hs = h[torch.tensor(bi, device=h.device), torch.tensor(pi, device=h.device)]
            logits = self.head(hs.to(self.head.weight.device)).float()
            lp = logits.log_softmax(-1).gather(-1, torch.tensor(tgt, device=logits.device)[:, None])[:, 0].cpu()
            sums, cnts = [0.0] * len(chunk), [0] * len(chunk)
            for k, b in enumerate(owner):
                sums[b] += float(lp[k])
                cnts[b] += 1
            out.extend(zip(sums, cnts))
        return out

    @torch.inference_mode()
    def generate(self, images, prompts, max_new: int = 16):
        """Batched greedy decoding, right-padded, no KV cache (recomputes the prefix each step)."""
        res = []
        for i in range(0, len(prompts), self.bs):
            ims, ps = images[i:i + self.bs], prompts[i:i + self.bs]
            embeds, mask, _ = self._batch(ims, [self._ids(p, None)[0] for p in ps])
            B, T, D = embeds.shape
            buf = torch.zeros(B, T + max_new, D, dtype=embeds.dtype, device=embeds.device)
            m = torch.zeros(B, T + max_new, dtype=mask.dtype, device=mask.device)
            buf[:, :T], m[:, :T] = embeds, mask
            lens = mask.sum(1).clone()
            done = torch.zeros(B, dtype=torch.bool)
            toks = [[] for _ in range(B)]
            for _ in range(max_new):
                cur = int(lens.max())
                [h], _ = self.m.run_backbone(buf[:, :cur], m[:, :cur], (-1,))
                last = h[torch.arange(B, device=h.device), (lens - 1).to(h.device)]
                nxt = self.head(last.to(self.head.weight.device)).argmax(-1).to(self.dev)
                emb = self.m.lm.embed_tokens(nxt)
                for b in range(B):
                    if done[b]:
                        continue
                    t = int(nxt[b])
                    if t in self.eos:
                        done[b] = True
                        continue
                    toks[b].append(t)
                    buf[b, lens[b]] = emb[b]
                    m[b, lens[b]] = 1
                    lens[b] += 1
                if bool(done.all()):
                    break
            res.extend(self.tok.decode(t, skip_special_tokens=True).strip() for t in toks)
        return res


def derange(n: int, seed: int) -> list[int]:
    """Permutation with no fixed points, so every 'shuffled' item gets a different sample's image."""
    if n < 2:
        return list(range(n))
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    return [perm[(perm.index(i) + 1) % n] for i in range(n)]


def score_task(scorer: Scorer, items: list[dict], images: list[torch.Tensor], n_gen: int = 0):
    """Metrics for one condition (`images[i]` is the image shown with items[i])."""
    kind = items[0]["kind"]
    per_item, metrics = [], {}
    if kind == "mc":
        rows, owner = [], []
        for i, it in enumerate(items):
            for o in it["options"]:
                rows.append((images[i], it["prompt"], o))
                owner.append(i)
        ll = scorer.loglik(rows)
        scores = defaultdict(list)
        for i, (s, c) in zip(owner, ll):
            scores[i].append(s / max(c, 1))  # per-token mean: options differ in length
        groups = defaultdict(list)
        for i, it in enumerate(items):
            pred = int(np.argmax(scores[i]))
            ok = pred == it["answer"]
            groups[it.get("group", "")].append(ok)
            per_item.append({"pred": it["options"][pred], "gold": it["options"][it["answer"]], "correct": ok})
        allok = [p["correct"] for p in per_item]
        metrics["acc"] = float(np.mean(allok))
        metrics["chance"] = float(np.mean([1 / len(it["options"]) for it in items]))
        if len(groups) > 1:
            metrics["acc_by_group"] = {g: round(float(np.mean(v)), 4) for g, v in sorted(groups.items())}
        if items[0]["options"] == ["Yes", "No"]:  # POPE-style extras (positive class = "Yes")
            pred_yes = np.array([p["pred"] == "Yes" for p in per_item])
            gold_yes = np.array([it["answer"] == 0 for it in items])
            tp = float((pred_yes & gold_yes).sum())
            prec = tp / max(pred_yes.sum(), 1)
            rec = tp / max(gold_yes.sum(), 1)
            metrics["f1"] = 2 * prec * rec / max(prec + rec, 1e-9)
            metrics["yes_ratio"] = float(pred_yes.mean())
    elif kind == "nll":
        ll = scorer.loglik([(images[i], it["prompt"], it["target"]) for i, it in enumerate(items)])
        tot, cnt = sum(-s for s, _ in ll), sum(c for _, c in ll)
        metrics["nll_per_token"] = tot / max(cnt, 1)
        metrics["ppl"] = math.exp(min(metrics["nll_per_token"], 50))
        per_item = [{"nll": -s / max(c, 1)} for s, c in ll]
        if n_gen:
            k = min(n_gen, len(items))
            caps = scorer.generate(images[:k], [it["prompt"] for it in items[:k]], max_new=32)
            for i, c in enumerate(caps):
                per_item[i]["generated"] = c
    elif kind == "gen":
        preds = scorer.generate(images, [it["prompt"] for it in items], max_new=10)
        groups = defaultdict(list)
        for it, p in zip(items, preds):
            a = vqa_acc(p, it["answers"])
            groups[it.get("group", "")].append(a)
            per_item.append({"pred": p, "gold": it["answers"][0], "acc": a})
        metrics["vqa_acc"] = float(np.mean([p["acc"] for p in per_item]))
        metrics["acc_by_group"] = {g: round(float(np.mean(v)), 4) for g, v in sorted(groups.items())}
    return metrics, per_item


HEADLINE = {"mc": ("acc", +1), "nll": ("nll_per_token", -1), "gen": ("vqa_acc", +1)}


def item_score(kind: str, p: dict) -> float:
    """Per-item version of the headline metric, oriented so higher = better."""
    return float(p["correct"]) if kind == "mc" else (-p["nll"] if kind == "nll" else p["acc"])


def paired_gain(kind: str, per_img: list[dict], per_shuf: list[dict]) -> tuple[float, float]:
    """Mean and standard error of the per-item (image - shuffled) difference. |gain| < 2 SE is noise.
    (For nll the item mean differs slightly from the token-weighted headline.)"""
    d = np.array([item_score(kind, x) - item_score(kind, y) for x, y in zip(per_img, per_shuf)])
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se


# --------------------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="local Ripple snapshot (weights, or config/code only with --tiny)")
    p.add_argument("--ckpt", default=None, help="trainable-only .pt from train.py; omit for the random baseline")
    p.add_argument("--tiny", action="store_true", help="tiny random Ripple + tiny SigLIP (pipeline test)")
    p.add_argument("--device", default=None, help="device for small modules (default cuda if available)")
    p.add_argument("--tasks", default="shapes,pope,coco,mmbench,vqav2")
    p.add_argument("--n", type=int, default=500, help="items per task")
    p.add_argument("--bs", type=int, default=16, help="sequences per forward")
    p.add_argument("--pool", type=int, default=2, help="random baseline only: must match the trained run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_gen", type=int, default=16, help="greedy captions to save per caption task")
    p.add_argument("--llava_zip", default=None)
    p.add_argument("--llava_offset", type=int, default=500_000)
    p.add_argument("--out", default="eval_vlm_out")
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)  # makes the random projector reproducible
    from transformers import AutoTokenizer

    from loading import build_vla_inference
    from model import VLAConfig

    tok = AutoTokenizer.from_pretrained(a.ckpt_dir, trust_remote_code=True)
    base_cfg = None if a.ckpt else VLAConfig(pool=a.pool, image_insert_pos=image_insert_pos(tok))
    if a.tiny and base_cfg is not None:
        base_cfg = dataclasses.replace(base_cfg, expert_dim=128, expert_layers=2, expert_heads=4)
    t0 = time.time()
    model, cfg = build_vla_inference(a.ckpt, a.ckpt_dir, tiny=a.tiny, cfg=base_cfg, device=a.device)
    assert cfg.image_insert_pos == image_insert_pos(tok), "checkpoint was trained with a different prompt prefix"
    print(f"[eval] model ready in {time.time() - t0:.0f}s (pool={cfg.pool}, ckpt={a.ckpt or 'RANDOM'})", flush=True)
    scorer = Scorer(model, tok, a.bs)

    os.makedirs(a.out, exist_ok=True)
    results = {"ckpt": a.ckpt or "RANDOM", "tiny": a.tiny, "n": a.n, "seed": a.seed, "pool": cfg.pool, "tasks": {}}
    samples = open(os.path.join(a.out, "samples.jsonl"), "w")
    for name in a.tasks.split(","):
        t0 = time.time()
        items = TASKS[name](a.n, a.seed, llava_zip=a.llava_zip, llava_offset=a.llava_offset)
        own = [it["image"] for it in items]
        perm = derange(len(items), a.seed + 1)
        shuf = [own[j] for j in perm]
        kind = items[0]["kind"]
        n_gen = a.n_gen if kind == "nll" else 0
        m_img, per_img = score_task(scorer, items, own, n_gen)
        m_shuf, per_shuf = score_task(scorer, items, shuf, n_gen)
        key, sign = HEADLINE[kind]
        res = {"kind": kind, "n": len(items), "image": m_img, "shuffled": m_shuf, "headline": key,
               "gain": sign * (m_img[key] - m_shuf[key]), "gain_se": paired_gain(kind, per_img, per_shuf)[1],
               "sec": round(time.time() - t0, 1)}
        results["tasks"][name] = res
        for i, it in enumerate(items):
            rec = {"task": name, "i": i, "prompt": it["prompt"], "image": per_img[i], "shuffled": per_shuf[i]}
            for k in ("target", "options", "group"):
                if k in it:
                    rec[k] = it[k]
            samples.write(json.dumps(rec) + "\n")
        samples.flush()
        print(f"[{name}] {key}: image {m_img[key]:.4f} | shuffled {m_shuf[key]:.4f} | gain {res['gain']:+.4f} ± {res['gain_se']:.4f}"
              + (f" | chance {m_img['chance']:.3f}" if "chance" in m_img else "") + f"  ({res['sec']}s)", flush=True)
    samples.close()
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    print("\n| task | metric | image | shuffled | gain | SE | visual signal? |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results["tasks"].items():
        k = r["headline"]
        sig = "yes" if r["gain"] > 2 * r["gain_se"] else "no"
        print(f"| {name} | {k} | {r['image'][k]:.4f} | {r['shuffled'][k]:.4f} | {r['gain']:+.4f} | "
              f"{r['gain_se']:.4f} | {sig} |")
    print(f"\n[eval] wrote {a.out}/results.json and samples.jsonl")


if __name__ == "__main__":
    main()
