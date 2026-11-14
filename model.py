"""Ripple-VLA: SigLIP vision + Ripple-65B backbone + flow-matching action expert.

Sequence fed to the backbone (causal, right-padded):

    [img_0 tokens][img_1 tokens][text tokens ...][pad]

- Stage "align": text = prompt + caption, LM loss on caption tokens only. Trains the projector.
- Stage "vla":   text = task instruction. Backbone hidden states (selected layers) are the
                 context for the action expert, trained with a flow-matching loss on 50-step
                 action chunks (pi0 convention: t=1 is noise, t=0 is data).

Ripple-specific notes (from modeling_ricursive.py):
- `RicursiveModel.forward` multiplies *any* inputs_embeds by config.embed_scale (~71.55), so we
  splice image tokens at the RAW embed_tokens scale; the projector's output RMS is initialised to
  the mean RMS of embed_tokens rows.
- Rotary is computed on the fly (no buffers) and weight names match module names, so meta-device
  init + streaming safetensors load works (see loading.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class VLAConfig:
    # vision
    image_size: int = 224          # SigLIP input resolution
    num_images: int = 2            # agentview + wrist
    pool: int = 1                  # spatial pooling of patch grid before projection
    # action expert
    action_dim: int = 7
    state_dim: int = 8
    chunk: int = 50
    expert_dim: int = 1024
    expert_layers: int = 12
    expert_heads: int = 16
    feature_layers: tuple[int, ...] = (-1,)  # indices into hidden_states tuple; -1 = final (post-norm)
    flow_steps: int = 10
    image_insert_pos: int = 0  # image tokens are spliced in after this many (fixed-prefix) text tokens
    # LoRA on backbone (0 = frozen backbone)
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    # normalisation stats (filled from dataset)
    action_mean: list[float] = field(default_factory=lambda: [0.0] * 7)
    action_std: list[float] = field(default_factory=lambda: [1.0] * 7)
    state_mean: list[float] = field(default_factory=lambda: [0.0] * 8)
    state_std: list[float] = field(default_factory=lambda: [1.0] * 8)


# --------------------------------------------------------------------------------------
# LoRA (minimal, FSDP-friendly: plain parameters on the wrapped module)
# --------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        self.scale = alpha / rank
        self.reset_lora()

    def reset_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t().to(x.dtype)) @ self.lora_B.t().to(x.dtype) * self.scale


def apply_lora(decoder_layers: nn.ModuleList, cfg: VLAConfig) -> None:
    for layer in decoder_layers:
        for parent in layer.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in cfg.lora_targets:
                    setattr(parent, name, LoRALinear(child, cfg.lora_rank, cfg.lora_alpha))


# --------------------------------------------------------------------------------------
# Vision projector
# --------------------------------------------------------------------------------------
class VisionProjector(nn.Module):
    """SigLIP patch grid -> optional pxp pooling (space-to-depth) -> MLP -> LLM width.

    Output is RMS-normalised with a learnable per-channel gain initialised to the mean RMS of the
    LLM's (unscaled) token embeddings, so spliced image tokens start on the same scale as text.
    """

    def __init__(self, vis_dim: int, llm_dim: int, pool: int, target_rms: float):
        super().__init__()
        self.pool = pool
        d_in = vis_dim * pool * pool
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.gain = nn.Parameter(torch.full((llm_dim,), float(target_rms)))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        B, N, D = feats.shape
        s = math.isqrt(N)
        assert s * s == N, f"non-square patch grid {N}"
        p = self.pool
        if p > 1:
            s2 = s // p
            x = feats.view(B, s, s, D)[:, : s2 * p, : s2 * p]
            x = x.reshape(B, s2, p, s2, p, D).permute(0, 1, 3, 2, 4, 5).reshape(B, s2 * s2, p * p * D)
        else:
            x = feats
        x = self.fc2(F.gelu(self.fc1(self.ln(x))))
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
        return x * self.gain.to(x.dtype)


# --------------------------------------------------------------------------------------
# Action expert (flow matching)
# --------------------------------------------------------------------------------------
def sinusoidal(t: torch.Tensor, dim: int, max_period: float = 100.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None] * 1000.0 / max_period
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ExpertBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        # adaLN-style modulation from the flow time
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, ctx, ctx_pad_mask, temb):
        s1, b1, s2, b2, g1, g2 = self.mod(temb)[:, None].chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + (1 + g1) * self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_pad_mask, need_weights=False)[0]
        h = self.n3(x) * (1 + s2) + b2
        x = x + (1 + g2) * self.mlp(h)
        return x


class ActionExpert(nn.Module):
    """Tokens = [state] + [chunk noisy actions]; cross-attends to projected backbone features."""

    def __init__(self, cfg: VLAConfig, llm_dim: int):
        super().__init__()
        d = cfg.expert_dim
        self.cfg = cfg
        n_feat = len(cfg.feature_layers)
        self.ctx_norm = nn.ModuleList([nn.LayerNorm(llm_dim) for _ in range(n_feat)])
        self.ctx_proj = nn.ModuleList([nn.Linear(llm_dim, d) for _ in range(n_feat)])
        self.feat_emb = nn.Parameter(torch.zeros(n_feat, d))
        self.state_in = nn.Linear(cfg.state_dim, d)
        self.action_in = nn.Linear(cfg.action_dim, d)
        self.pos = nn.Parameter(torch.randn(1, cfg.chunk + 1, d) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([ExpertBlock(d, cfg.expert_heads) for _ in range(cfg.expert_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.action_dim)

    def forward(self, feats: list[torch.Tensor], ctx_pad: torch.Tensor, state, x_t, t):
        """feats: list of (B,T,llm_dim) backbone hidden states; ctx_pad: (B,T) True=ignore.
        Context projection lives inside forward so FSDP gathers params for it."""
        pdt = self.out.weight.dtype
        ctx = torch.cat([proj(norm(f.to(pdt))) + self.feat_emb[i].to(pdt)
                         for i, (f, norm, proj) in enumerate(zip(feats, self.ctx_norm, self.ctx_proj))], dim=1)
        ctx_pad = torch.cat([ctx_pad] * len(feats), dim=1)
        temb = self.time_mlp(sinusoidal(t, self.cfg.expert_dim).to(pdt))
        tok = torch.cat([self.state_in(state.to(pdt))[:, None], self.action_in(x_t.to(pdt))], dim=1)
        tok = tok + self.pos.to(pdt) + temb[:, None]
        for blk in self.blocks:
            tok = blk(tok, ctx, ctx_pad, temb)
        return self.out(self.out_norm(tok[:, 1:])).float()  # velocity for the action tokens


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------
class RippleVLA(nn.Module):
    def __init__(self, backbone: nn.Module, vision: nn.Module, cfg: VLAConfig, target_rms: float):
        """backbone: HF *ForCausalLM with `.model` (decoder) and `.lm_head`.
        vision:   SiglipVisionModel (frozen)."""
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.vision = vision
        llm_dim = backbone.config.hidden_size
        vis_dim = vision.config.hidden_size
        self.projector = VisionProjector(vis_dim, llm_dim, cfg.pool, target_rms)
        self.expert = ActionExpert(cfg, llm_dim)
        for k in ("action_mean", "action_std", "state_mean", "state_std"):
            self.register_buffer(k, torch.tensor(getattr(cfg, k), dtype=torch.float32), persistent=True)

    # ---- helpers -------------------------------------------------------------------
    @property
    def lm(self):
        return self.backbone.model

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, K, 3, H, W) float in [0,1] -> (B, K*n_tok, llm_dim)."""
        B, K = images.shape[:2]
        x = images.flatten(0, 1).float()
        if x.shape[-1] != self.cfg.image_size or x.shape[-2] != self.cfg.image_size:
            x = F.interpolate(x, size=(self.cfg.image_size,) * 2, mode="bilinear", align_corners=False, antialias=True)
        x = (x - 0.5) / 0.5  # SigLIP normalisation
        with torch.no_grad():
            feats = self.vision(pixel_values=x.to(self.vision.dtype)).last_hidden_state
        tok = self.projector(feats.to(self.projector.fc1.weight.dtype))
        return tok.reshape(B, -1, tok.shape[-1])

    def build_inputs(self, images, input_ids, text_mask):
        """Layout [text[:p]][images][text[p:]]. Returns inputs_embeds, attention_mask (1=valid), n_img_tokens."""
        img = self.encode_images(images)
        txt = self.lm.embed_tokens(input_ids)
        p = self.cfg.image_insert_pos
        img = img.to(txt.dtype)
        embeds = torch.cat([txt[:, :p], img, txt[:, p:]], dim=1)
        ones = torch.ones(img.shape[:2], dtype=text_mask.dtype, device=text_mask.device)
        mask = torch.cat([text_mask[:, :p], ones, text_mask[:, p:]], dim=1)
        return embeds, mask, img.shape[1]

    def run_backbone(self, embeds, mask, want_layers):
        need_all = any(i != -1 for i in want_layers)
        out = self.lm(inputs_embeds=embeds, attention_mask=mask, use_cache=False,
                      output_hidden_states=need_all, return_dict=True)
        dev = embeds.device  # device-mapped inference: bring features back next to the heads
        if need_all:
            hs = out.hidden_states  # len = n_layers + 1 (last is post-norm)
            return [hs[i].to(dev) for i in want_layers], out.last_hidden_state.to(dev)
        return [out.last_hidden_state.to(dev)], out.last_hidden_state.to(dev)

    # ---- normalisation ---------------------------------------------------------------
    def norm_actions(self, a):
        return (a - self.action_mean) / self.action_std

    def unnorm_actions(self, a):
        return a * self.action_std + self.action_mean

    def norm_state(self, s):
        return (s - self.state_mean) / self.state_std

    # ---- losses ----------------------------------------------------------------------
    def forward(self, batch: dict, stage: str):
        if stage == "align":
            return self.caption_loss(batch)
        if stage == "vla":
            return self.flow_loss(batch)
        raise ValueError(stage)

    def caption_loss(self, batch):
        embeds, mask, n_img = self.build_inputs(batch["images"], batch["input_ids"], batch["attention_mask"])
        _, last = self.run_backbone(embeds, mask, (-1,))
        lab, p = batch["labels"], self.cfg.image_insert_pos   # -100 where no loss
        lab = torch.cat([lab[:, :p], lab.new_full((lab.shape[0], n_img), -100), lab[:, p:]], dim=1)
        tgt = lab[:, 1:]                                        # predict token q from position q-1
        sel = tgt != -100
        logits = self.backbone.lm_head(last[:, :-1][sel])
        return F.cross_entropy(logits.float(), tgt[sel])

    def flow_loss(self, batch):
        embeds, mask, _ = self.build_inputs(batch["images"], batch["input_ids"], batch["attention_mask"])
        feats, _ = self.run_backbone(embeds, mask, self.cfg.feature_layers)

        a = self.norm_actions(batch["actions"].float())           # (B, chunk, A)
        s = self.norm_state(batch["state"].float())
        B = a.shape[0]
        # pi0: t ~ Beta(1.5, 1) mapped to [0.001, 1]; x_t = t*eps + (1-t)*a ; target u = eps - a
        t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(a.device) * 0.999 + 0.001
        eps = torch.randn_like(a)
        x_t = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        v = self.expert(feats, mask == 0, s, x_t, t)
        err = (v - (eps - a)).pow(2)
        if "action_is_pad" in batch:
            keep = (~batch["action_is_pad"]).float()[..., None]
            return (err * keep).sum() / (keep.sum() * a.shape[-1]).clamp_min(1)
        return err.mean()

    @torch.no_grad()
    def sample_actions(self, images, input_ids, attention_mask, state, steps: int | None = None):
        steps = steps or self.cfg.flow_steps
        embeds, mask, _ = self.build_inputs(images, input_ids, attention_mask)
        feats, _ = self.run_backbone(embeds, mask, self.cfg.feature_layers)
        pad = mask == 0
        s = self.norm_state(state.float())
        B = state.shape[0]
        x = torch.randn(B, self.cfg.chunk, self.cfg.action_dim, device=state.device, dtype=torch.float32)
        dt = -1.0 / steps
        t = torch.ones(B, device=state.device)
        for _ in range(steps):
            v = self.expert(feats, pad, s, x, t)
            x = x + dt * v
            t = t + dt
        return self.unnorm_actions(x)

    @torch.no_grad()
    def caption(self, images, input_ids, attention_mask, max_new_tokens=40, eos_ids=()):
        """Greedy caption (sanity check for stage-1 alignment). Batch size 1, no KV cache."""
        embeds, mask, n_img = self.build_inputs(images, input_ids, attention_mask)
        out = []
        for _ in range(max_new_tokens):
            _, last = self.run_backbone(embeds, mask, (-1,))
            nxt = self.backbone.lm_head(last[:, -1]).argmax(-1)
            if int(nxt[0]) in eos_ids:
                break
            out.append(int(nxt[0]))
            embeds = torch.cat([embeds, self.lm.embed_tokens(nxt)[:, None]], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        return out


def trainable_parameters(model: RippleVLA, stage: str):
    """Freeze everything, then unfreeze what the stage trains. Returns list of (name, param)."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for n, p in model.named_parameters():
        train = n.startswith("projector.")
        if stage == "vla":
            train = train or n.startswith("expert.") or ".lora_" in n
        if train:
            p.requires_grad_(True)
            names.append((n, p))
    return names
