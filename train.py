"""torchrun entrypoint for both stages.

  torchrun --nproc_per_node 8 train.py --stage align --ckpt_dir $RIPPLE --llava_dir /data/llava ...
  torchrun --nproc_per_node 8 train.py --stage vla   --ckpt_dir $RIPPLE --resume /out/align/last.pt ...
  torchrun --nproc_per_node 2 train.py --tiny --stage vla --ckpt_dir $RIPPLE_CFG_ONLY --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["align", "vla"], required=True)
    p.add_argument("--ckpt_dir", required=True, help="local Ripple snapshot (config/code/tokenizer[/weights])")
    p.add_argument("--tiny", action="store_true", help="tiny random Ripple + tiny SigLIP + synthetic data")
    p.add_argument("--out", default="/outputs/ripple_vla/run")
    p.add_argument("--resume", default=None, help="trainable-only .pt (e.g. stage-1 projector)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=8, help="per-GPU batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--pool", type=int, default=1)
    p.add_argument("--feature_layers", default="-1", help="comma list into hidden_states, e.g. '40,-1'")
    p.add_argument("--llava_dir", default=None)
    p.add_argument("--llava_zip", default=None, help="path to LLaVA-Pretrain images.zip (read in place)")
    p.add_argument("--llava_limit", type=int, default=None)
    p.add_argument("--libero_caption", action="store_true", help="align: add LIBERO task-caption data")
    p.add_argument("--libero_root", default=None)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="ripple-vla")
    return p.parse_args()


def main():
    a = parse()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(1234 + rank)
    is0 = rank == 0
    log = (lambda *x: print(*x, flush=True)) if is0 else (lambda *x: None)

    from transformers import AutoTokenizer

    from data import Collator, LiberoCaption, LiberoVLA, LlavaPretrain, Synthetic, image_insert_pos
    from loading import build_vla, save_trainable
    from model import VLAConfig

    tok = AutoTokenizer.from_pretrained(a.ckpt_dir)
    cfg = VLAConfig(pool=a.pool, lora_rank=a.lora_rank if a.stage == "vla" else 0,
                    feature_layers=tuple(int(x) for x in a.feature_layers.split(",")),
                    image_insert_pos=image_insert_pos(tok))

    # ---------------- data ----------------
    if a.tiny:
        ds = Synthetic(caption=a.stage == "align", num_images=2)
        cfg = dataclasses.replace(cfg, expert_dim=128, expert_layers=2, expert_heads=4)
    elif a.stage == "vla":
        ds = LiberoVLA(root=a.libero_root, chunk=cfg.chunk)
        st = ds.stats
        cfg = dataclasses.replace(cfg, action_mean=st["action"]["mean"], action_std=st["action"]["std"],
                                  state_mean=st["observation.state"]["mean"],
                                  state_std=st["observation.state"]["std"])
    else:
        parts = []
        if a.llava_dir or a.llava_zip:
            parts.append(LlavaPretrain(a.llava_dir, limit=a.llava_limit, zip_path=a.llava_zip))
        if a.libero_caption:
            parts.append(LiberoCaption(LiberoVLA(root=a.libero_root, chunk=1)))
        assert parts, "align stage needs --llava_dir/--llava_zip and/or --libero_caption"
        ds = parts[0] if len(parts) == 1 else None
        if ds is None:
            # LLaVA has 1 image/sample, LIBERO 2: keep them in separate loaders, alternate batches.
            pass
    loaders = []
    for d in ([ds] if ds is not None else parts):
        sampler = DistributedSampler(d, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        loaders.append(DataLoader(d, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                                  collate_fn=Collator(tok), pin_memory=True, drop_last=True,
                                  persistent_workers=a.workers > 0))
    log(f"[data] {[len(l.dataset) for l in loaders]} samples; world={world} bs/gpu={a.bs} "
        f"accum={a.grad_accum} -> global bs {a.bs * world * a.grad_accum}")

    # ---------------- model ----------------
    t0 = time.time()
    model, trainable = build_vla(cfg, a.stage, tiny=a.tiny, ckpt_dir=a.ckpt_dir, resume=a.resume)
    model.train()
    model.vision.eval()
    n_tr = sum(p.numel() for _, p in trainable)
    log(f"[model] built in {time.time() - t0:.0f}s; trainable params {n_tr / 1e6:.1f}M "
        f"(image_insert_pos={cfg.image_insert_pos})")

    groups = [{"params": [p for n, p in trainable if ".lora_" not in n], "lr": a.lr},
              {"params": [p for n, p in trainable if ".lora_" in n], "lr": a.lr_lora}]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=a.wd, eps=1e-8)

    def lr_mult(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    if is0:
        os.makedirs(a.out, exist_ok=True)
        json.dump({"args": vars(a), "cfg": dataclasses.asdict(cfg)}, open(f"{a.out}/config.json", "w"), indent=1)
        if a.wandb:
            import wandb

            wandb.init(project=a.wandb_project, name=os.path.basename(a.out), config=vars(a))

    iters = [iter(l) for l in loaders]
    epochs = [0] * len(loaders)

    def next_batch(step):
        j = step % len(loaders)
        try:
            b = next(iters[j])
        except StopIteration:
            epochs[j] += 1
            loaders[j].sampler.set_epoch(epochs[j])
            iters[j] = iter(loaders[j])
            b = next(iters[j])
        return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}

    params = [p for _, p in trainable]
    tl, tstart = 0.0, time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_mult(step)
        for micro in range(a.grad_accum):
            batch = next_batch(step * a.grad_accum + micro)
            loss = model(batch, a.stage) / a.grad_accum
            loss.backward()
            tl += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % a.log_every == 0:
            avg = torch.tensor(tl / a.log_every, device="cuda")
            dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            dt = (time.time() - tstart) / a.log_every
            gnv = gn.full_tensor().item() if hasattr(gn, "full_tensor") else float(gn)
            log(f"step {step + 1:6d} loss {avg.item():.4f} gnorm {gnv:.3f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {dt:.2f}s/it")
            if is0 and a.wandb:
                import wandb

                wandb.log({"loss": avg.item(), "grad_norm": gnv, "lr": opt.param_groups[0]["lr"],
                           "sec_per_it": dt}, step=step + 1)
            tl, tstart = 0.0, time.time()
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save_trainable(model, f"{a.out}/last.pt", extra={"step": step + 1, "cfg": dataclasses.asdict(cfg)})
            log(f"[ckpt] saved {a.out}/last.pt @ {step + 1}")
            if is0 and "MODAL_TASK_ID" in os.environ:
                os.system("sync")

    # quick qualitative check
    model.eval()
    b = next_batch(0)
    if a.stage == "align":
        n = 1
        # FSDP: every rank must run the SAME number of forwards, so no early stop on EOS (ranks hold
        # different data and would stop at different steps -> NCCL collective mismatch / hang).
        ids = model.caption(b["images"][:n], b["input_ids"][:n, : _prompt_len(b)], b["attention_mask"][:n, : _prompt_len(b)],
                            max_new_tokens=24, eos_ids=())
        log("[sample caption]", repr(tok.decode(ids).split("<|im_end|>")[0]))
    else:
        act = model.sample_actions(b["images"][:2], b["input_ids"][:2], b["attention_mask"][:2], b["state"][:2])
        err = (act - b["actions"][:2]).abs().mean().item()
        log(f"[sample actions] shape {tuple(act.shape)} mean|err| vs GT {err:.4f}")
    dist.barrier()
    dist.destroy_process_group()


def _prompt_len(b):
    """Length of the prompt part of sample 0 (labels are -100 before the target)."""
    lab = b["labels"][0]
    nz = (lab != -100).nonzero()
    return int(nz[0]) if len(nz) else int(b["attention_mask"][0].sum())


if __name__ == "__main__":
    main()
