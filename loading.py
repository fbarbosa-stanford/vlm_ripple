"""Build RippleVLA, shard with FSDP2, and stream Ripple weights rank-locally from safetensors.

Memory story for the 65B backbone (bf16, ~130 GB):
  1. Backbone is built on the meta device (0 bytes).
  2. fully_shard() per decoder layer / embed / lm_head; then to_empty(cuda) allocates only the
     local shard on each GPU (~130/world GB).
  3. Each rank reads ONLY its row-slice of every tensor via safetensors get_slice (mmap), so no
     process ever materialises the full model in CPU RAM.

Small modules (SigLIP, projector, action expert, norm-stat buffers) are built for real on CPU,
snapshotted, and copied back after to_empty().
"""

from __future__ import annotations

import json
import math
import os
import zlib

import torch
import torch.nn as nn

from model import LoRALinear, RippleVLA, VLAConfig, apply_lora, trainable_parameters

BACKBONE_REPO = "ricursiveAI/Ripple-65B-Preview"
VISION_REPO = "google/siglip-so400m-patch14-224"


# --------------------------------------------------------------------------------------
def tiny_backbone_config(cfg_dir: str):
    """Ripple's real config + modeling code, shrunk to a few-M-param model for smoke tests."""
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=True)
    c.hidden_size, c.intermediate_size = 256, 512
    c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim = 5, 4, 2, 64
    c.layer_types = ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(5)]
    c.sliding_window = 64  # small so the smoke test actually exercises the window mask
    c.embed_scale = math.sqrt(256)
    return c


def tiny_vision():
    from transformers import SiglipVisionConfig, SiglipVisionModel

    return SiglipVisionModel(SiglipVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                                num_attention_heads=2, image_size=224, patch_size=14))


def _embed_rms_from_ckpt(ckpt_dir: str) -> float:
    from safetensors import safe_open

    idx = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(ckpt_dir, idx["model.embed_tokens.weight"]), "pt") as f:
        w = f.get_tensor("model.embed_tokens.weight").float()
    return w.pow(2).mean(-1).sqrt().mean().item()


def _ckpt_name(param_name: str) -> str:
    """backbone.model.layers.0.self_attn.q_proj.base.weight -> model.layers.0.self_attn.q_proj.weight"""
    assert param_name.startswith("backbone.")
    return param_name[len("backbone."):].replace(".base.", ".")


def _lora_init(name: str, shape, dtype) -> torch.Tensor:
    """Deterministic across ranks (seeded by name) so every rank slices the same full tensor."""
    if name.endswith("lora_B"):
        return torch.zeros(shape, dtype=dtype)
    g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
    bound = 1.0 / math.sqrt(shape[1])  # kaiming_uniform(a=sqrt(5)) bound
    return (torch.rand(shape, generator=g, dtype=torch.float32) * 2 - 1).mul_(bound).to(dtype)


# --------------------------------------------------------------------------------------
def build_vla(cfg: VLAConfig, stage: str, *, tiny: bool = False, ckpt_dir: str | None = None,
              vision_repo: str = VISION_REPO, cfg_dir: str | None = None, fsdp: bool = True,
              grad_ckpt: bool = True, resume: str | None = None):
    """Returns (model, trainable_named_params). Call inside an initialised process group if fsdp."""
    from transformers import AutoConfig, AutoModelForCausalLM, SiglipVisionModel

    bf16 = torch.bfloat16
    if tiny:
        bb_cfg = tiny_backbone_config(cfg_dir or ckpt_dir)
        backbone = AutoModelForCausalLM.from_config(bb_cfg, trust_remote_code=True, torch_dtype=bf16)
        vision = tiny_vision().to(bf16)
        target_rms = backbone.model.embed_tokens.weight.float().pow(2).mean(-1).sqrt().mean().item()
    else:
        bb_cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        with torch.device("meta"):
            backbone = AutoModelForCausalLM.from_config(bb_cfg, trust_remote_code=True, torch_dtype=bf16)
        vision = SiglipVisionModel.from_pretrained(vision_repo, torch_dtype=bf16)
        target_rms = _embed_rms_from_ckpt(ckpt_dir)
    vision.requires_grad_(False)

    if cfg.lora_rank > 0:
        apply_lora(backbone.model.layers, cfg)
    if grad_ckpt:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = RippleVLA(backbone, vision, cfg, target_rms)
    trainable = trainable_parameters(model, stage)

    # snapshot everything that is real (non-meta) so we can restore it after to_empty()
    init_sd = {k: v.detach().clone() for k, v in model.state_dict().items() if not v.is_meta}
    # non-persistent buffers (e.g. SigLIP position_ids) are absent from state_dict but are wiped by
    # to_empty(); snapshot them too and put them back afterwards.
    init_bufs = {k: b.detach().clone() for k, b in model.named_buffers() if not b.is_meta}

    if not fsdp:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        return model, trainable

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method

    world = torch.distributed.get_world_size()
    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=torch.float32)
    kw = dict(mesh=mesh, mp_policy=mp)
    for layer in backbone.model.layers:
        fully_shard(layer, **kw)
    fully_shard(backbone.model.embed_tokens, **kw)
    fully_shard(backbone.lm_head, **kw)
    for n, m in vision.named_modules():  # transformers 4/5 differ in nesting; find the encoder block list
        if isinstance(m, nn.ModuleList) and n.endswith("encoder.layers"):
            for layer in m:
                fully_shard(layer, **kw)
    fully_shard(vision, **kw)
    for blk in model.expert.blocks:
        fully_shard(blk, **kw)
    fully_shard(model.expert, **kw)
    fully_shard(model, **kw)
    register_fsdp_forward_method(model, "sample_actions")
    register_fsdp_forward_method(model, "caption")

    model.to_empty(device="cuda")
    _fill_weights(model, init_sd, None if tiny else ckpt_dir)
    with torch.no_grad():
        for k, b in model.named_buffers():
            if k in init_bufs:
                b.copy_(init_bufs[k].to(b.device, b.dtype))
    if resume:
        load_trainable(model, resume)
    # fully_shard() replaced every Parameter with a new sharded one; the pre-shard list is stale.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters after sharding"
    return model, trainable


@torch.no_grad()
def _fill_weights(model: nn.Module, init_sd: dict, ckpt_dir: str | None):
    from torch.distributed.tensor import DTensor

    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    handles, weight_map = {}, {}
    if ckpt_dir:
        from safetensors import safe_open

        weight_map = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))["weight_map"]
        for fn in set(weight_map.values()):
            handles[fn] = safe_open(os.path.join(ckpt_dir, fn), "pt", device="cpu")

    n_loaded = 0
    for name, t in model.state_dict().items():
        local = t.to_local() if isinstance(t, DTensor) else t
        rows = t.shape[0] if t.dim() > 0 else 1
        chunk = math.ceil(rows / world) if isinstance(t, DTensor) else rows
        start = min(rank * chunk, rows) if isinstance(t, DTensor) else 0
        n_local = local.shape[0] if local.dim() > 0 else 1
        if n_local == 0:
            continue

        if name in init_sd:
            src = init_sd[name]
            src = src[start:start + n_local] if isinstance(t, DTensor) else src
        elif ".lora_" in name:
            src = _lora_init(name, tuple(t.shape), torch.float32)[start:start + n_local]
        else:
            key = _ckpt_name(name)
            if key not in weight_map:
                raise KeyError(f"{name} -> {key} not found in checkpoint and not initialised")
            src = handles[weight_map[key]].get_slice(key)[start:start + n_local]
            n_loaded += 1
        local.copy_(src.to(device=local.device, dtype=local.dtype).reshape(local.shape))
    if rank == 0:
        print(f"[load] filled weights; {n_loaded} tensors streamed from checkpoint", flush=True)


# --------------------------------------------------------------------------------------
# Trainable-only checkpoints (projector + expert + LoRA + norm stats): small, fast.
# --------------------------------------------------------------------------------------
def _is_saved(name: str) -> bool:
    return name.startswith(("projector.", "expert.")) or ".lora_" in name or name in (
        "action_mean", "action_std", "state_mean", "state_std")


def save_trainable(model: nn.Module, path: str, extra: dict | None = None):
    from torch.distributed.tensor import DTensor

    sd = {}
    for name, t in model.state_dict().items():
        if _is_saved(name):
            full = t.full_tensor() if isinstance(t, DTensor) else t
            sd[name] = full.detach().cpu()
    if torch.distributed.get_rank() == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": sd, **(extra or {})}, path)
    torch.distributed.barrier()


@torch.no_grad()
def load_trainable(model: nn.Module, path: str):
    from torch.distributed.tensor import DTensor, distribute_tensor

    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    cur = model.state_dict()
    missing = []
    for name, src in sd.items():
        if name not in cur:
            missing.append(name)
            continue
        t = cur[name]
        if isinstance(t, DTensor):
            shard = distribute_tensor(src.to(t.device_mesh.device_type).to(t.dtype), t.device_mesh,
                                      t.placements, src_data_rank=None)
            t.to_local().copy_(shard.to_local())
        else:
            t.copy_(src.to(t.device, t.dtype))
    if torch.distributed.get_rank() == 0:
        print(f"[load] restored {len(sd) - len(missing)} trainable tensors from {path}"
              + (f" (skipped {len(missing)} unknown)" if missing else ""), flush=True)


# --------------------------------------------------------------------------------------
# Inference (no FSDP): backbone spread over local GPUs with accelerate device_map, small
# modules on cuda:0. Good for eval/demo serving with batching across many sims.
# --------------------------------------------------------------------------------------
def build_vla_inference(trained_pt: str, ckpt_dir: str, vision_repo: str = VISION_REPO,
                        tiny: bool = False):
    import dataclasses

    from transformers import AutoModelForCausalLM, SiglipVisionModel

    blob = torch.load(trained_pt, map_location="cpu", weights_only=False)
    cfg = VLAConfig(**{k: (tuple(v) if isinstance(v, list) and k in ("feature_layers", "lora_targets") else v)
                       for k, v in blob["cfg"].items() if k in {f.name for f in dataclasses.fields(VLAConfig)}})
    bf16 = torch.bfloat16
    if tiny:
        backbone = AutoModelForCausalLM.from_config(tiny_backbone_config(ckpt_dir), trust_remote_code=True,
                                                    torch_dtype=bf16).cuda()
        vision = tiny_vision().to(bf16)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(ckpt_dir, trust_remote_code=True, torch_dtype=bf16,
                                                        device_map="auto")
        vision = SiglipVisionModel.from_pretrained(vision_repo, torch_dtype=bf16)
    if cfg.lora_rank > 0:
        apply_lora(backbone.model.layers, cfg)  # LoRA params land on each layer's device
    model = RippleVLA.__new__(RippleVLA)
    nn.Module.__init__(model)
    # build small modules on cuda:0 without moving the (device-mapped) backbone
    tmp = RippleVLA(_Stub(backbone.config.hidden_size), _Stub(vision.config.hidden_size, vision.config), cfg, 1.0)
    model.cfg, model.backbone, model.vision = cfg, backbone, vision.cuda().eval()
    model.projector, model.expert = tmp.projector.cuda(), tmp.expert.cuda()
    for k in ("action_mean", "action_std", "state_mean", "state_std"):
        model.register_buffer(k, getattr(tmp, k).cuda())
    sd = blob["state_dict"]
    cur = dict(model.named_parameters()) | dict(model.named_buffers())
    missing = [n for n in sd if n not in cur]
    with torch.no_grad():
        for n, v in sd.items():
            if n in cur:
                cur[n].copy_(v.to(cur[n].device, cur[n].dtype))
    model.eval()
    print(f"[infer] loaded {len(sd) - len(missing)} trainable tensors; missing={missing[:5]}", flush=True)
    return model, cfg


class _Stub(nn.Module):
    def __init__(self, hidden, config=None):
        super().__init__()

        class C:
            pass

        self.config = config or C()
        self.config.hidden_size = hidden
