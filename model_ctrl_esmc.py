#!/usr/bin/env python3
"""
model_ctrl_esmc.py — the controlled MLM-vs-CLM backbone, anchored to ESM-C.

ANCHOR (single reference): ESM Cambrian (ESM-C), EvolutionaryScale, Dec 2024.
We do NOT reimplement it — we instantiate EvolutionaryScale's own
`esm.layers.transformer_stack.TransformerStack` (pip `esm==3.2.3`, Cambrian Open
License). That inherits, exactly and for free:

  - pre-LayerNorm blocks (nn.LayerNorm), bias-free Linears
  - QK-LayerNorm (qk_layernorm=True, bias-free)
  - RoPE, RotaryEmbedding(d_model // n_heads), base=10000
  - SwiGLU FFN at expansion_ratio=8/3 with swiglu_correction_fn (round up to x256)
  - residual scaling: residue_scaling_factor = sqrt(n_layers / 36) on BOTH branches
  - final nn.LayerNorm(d_model, bias=False)

DELIBERATE DEVIATIONS FROM THE ANCHOR (there are exactly two, both forced):

  1. WIDTH. ESM-C's smallest published size is 300M (d_model=960, n_heads=15,
     n_layers=30). We use d_model=320, n_heads=5, n_layers=30 -> ~42M. We keep
     ESM-C's head_dim (64) and its DEPTH (30 layers) exactly; only width is reduced,
     for compute. Keeping n_layers=30 is deliberate: residue_scaling_factor depends
     ONLY on depth, so 30 layers reproduces ESM-C-300M's value (0.913) exactly. A
     shallower model would silently move it (10 layers -> 0.527, amplifying residual
     branches ~1.9x, outside the regime ESM-C calibrated).

  2. CAUSAL ATTENTION for the CLM arm. ESM-C is masked-only; its MultiHeadAttention
     cannot express causality (its only mask is the SYMMETRIC `seq_id` equality mask).
     So we patch ONE method to AND in a lower-triangular mask. This deviation IS the
     experimental variable.

CONTROLLED-EXPERIMENT INVARIANT (why the patch is written this way):
  Both arms MUST run the *same* code path. So we patch MultiHeadAttention.forward
  globally, once, and switch behaviour with a per-instance `causal` flag. MLM and CLM
  therefore execute byte-identical code; only the flag differs. (Subclassing for the
  causal arm would give the two arms different classes -> different code paths -> an
  asymmetry confound in a controlled experiment.)

  The patched body is copied verbatim from esm==3.2.3 with one added AND. `esm` is
  PINNED in requirements.txt for exactly this reason.
"""
import functools

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.layers.attention import MultiHeadAttention
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack

# ESM-C anchor constants (do not edit — see docstring)
ESMC_HEAD_DIM = 64
ESMC_DEPTH = 30          # == ESM-C 300M; keeps residue_scaling_factor == sqrt(30/36)


def _forward_causal_aware(self, x, seq_id):
    """Verbatim esm==3.2.3 MultiHeadAttention.forward + one causal AND.

    Upstream builds only a symmetric mask (`seq_id` equality), so causality has to be
    injected here. With `causal=False` this is numerically identical to upstream.
    """
    qkv_BLD3 = self.layernorm_qkv(x)
    query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
    query_BLD, key_BLD = (
        self.q_ln(query_BLD).to(query_BLD.dtype),
        self.k_ln(key_BLD).to(query_BLD.dtype),
    )
    query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

    reshaper = functools.partial(
        einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
    )
    query_BHLD, key_BHLD, value_BHLD = map(reshaper, (query_BLD, key_BLD, value_BLD))

    mask_BHLL = None
    if seq_id is not None:
        # True == may attend. Real tokens (id 1) see real tokens; pads (id 0) see pads.
        mask_BHLL = (seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)).unsqueeze(1)
    if getattr(self, "causal", False):
        L = x.shape[-2]
        tri = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()[None, None]
        mask_BHLL = tri if mask_BHLL is None else (mask_BHLL & tri)

    # mask_BHLL=None reproduces upstream's no-bias shortcut branch exactly.
    context_BHLD = F.scaled_dot_product_attention(
        query_BHLD, key_BHLD, value_BHLD, mask_BHLL
    )
    context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
    return self.out_proj(context_BLD)


MultiHeadAttention.forward = _forward_causal_aware


class CtrlESMC(nn.Module):
    """ESM-C architecture at reduced width; `causal` selects the objective's arm.

    The ONLY difference between the MLM and CLM instances is `causal` (the attention
    mask) and how the loss is built downstream. Depth, width, init and every component
    are shared.
    """

    def __init__(self, vocab_size: int, d_model: int = 320, n_heads: int = 5,
                 n_layers: int = ESMC_DEPTH, causal: bool = False):
        super().__init__()
        if d_model // n_heads != ESMC_HEAD_DIM:
            raise ValueError(
                f"head_dim must be {ESMC_HEAD_DIM} to match ESM-C "
                f"(got d_model={d_model}, n_heads={n_heads} -> {d_model // n_heads})"
            )
        self.causal = causal
        self.d_model, self.n_layers = d_model, n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        # v_heads=None, n_layers_geom=0 -> no geometric attention (ESM-C is sequence-only).
        # Everything else left at TransformerStack's defaults, which ARE ESM-C's:
        # scale_residue=True, bias=False, qk_layernorm=True, ffn=swiglu, expansion=8/3.
        self.transformer = TransformerStack(
            d_model, n_heads, None, n_layers, n_layers_geom=0, use_flash_attn=False
        )
        self.head = RegressionHead(d_model, vocab_size)
        for blk in self.transformer.blocks:
            blk.attn.causal = causal

    def forward(self, input_ids, attention_mask, return_hidden: bool = False):
        x = self.embed(input_ids)
        # seq_id: 1=real, 0=pad. Upstream's equality mask then isolates pads from
        # real tokens, which is exactly the padding mask we want.
        post, _pre, hiddens = self.transformer(x, sequence_id=attention_mask.long())
        logits = self.head(post)
        if return_hidden:
            return logits, hiddens
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
