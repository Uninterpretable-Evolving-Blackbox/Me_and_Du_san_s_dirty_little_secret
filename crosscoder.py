#!/usr/bin/env python3
"""
crosscoder.py — BatchTopK model-diffing crosscoder + its validity diagnostics.

ANCHOR (single, citable)
  Lindsey, Templeton, Marcus, Conerly, Batson & Olah, "Sparse Crosscoders for Cross-Layer
  Features and Model Diffing", Transformer Circuits, 25 Oct 2024 — the MODEL-DIFFING variant:
  one shared latent, per-model encoders e^m_j and decoders d^m_j, so a latent's decoder norm
  in each model says whether that model uses it.

MANDATORY CORRECTION (not optional — see below)
  Minder, Dumas, Juang, Chughtai & Nanda, arXiv:2504.02922 (2025). The naive L1 crosscoder's
  decoder-norm diffing metric yields a clean trimodal shared/unique split that is an ARTIFACT
  OF THE LOSS, not of the data. Two named failure modes:
    * Complete Shrinkage  — the sparsity loss drives one model's decoder norm to ~0 even when
      the latent genuinely contributes to that model  -> FALSELY model-specific.
    * Latent Decoupling   — one shared concept split across per-model latents -> also falsely
      model-exclusive (18% of their L1 "chat-only" latents).
  Fixes: BatchTopK (no decoder-norm penalty, hence no shrinkage pressure) + Latent Scaling as
  a diagnostic. Reported >3x fewer false unique attributions.

  We use BatchTopK because the naive method's failure mode is FABRICATING model-specific
  features, which is exactly the claim this experiment would make.

WHY A CROSSCODER AT ALL (the methodological point)
  Every comparison in this project so far trains TWO SEPARATE dictionaries and compares summary
  statistics. ESM-2's SAE and RITA's SAE have no feature correspondence, so "this feature exists
  in one model and not the other" is not even expressible. A crosscoder makes it expressible.

DEVIATIONS FROM THE ANCHOR (logged, per the project's anchoring rule)
  1. BatchTopK, not L1                     -> Minder et al. 2025 (above).
  2. Per-token RMS norm applied upstream   -> the project's A2 instrument fix; also equalises
                                              the two models' scales so RITA's massive
                                              activations cannot dominate the joint loss.
  3. CONCATENATED decoder unit-norm, never per-model. Per-model norming would destroy Delta_norm,
     because the decoder norms ARE the metric. (`sae.py::normalize_decoder` does per-column
     norming — do NOT reuse it here.)
  4. Per-model MSE averaged per DIMENSION then summed, so a 1280-d and a 1536-d model contribute
     equally instead of the wider one carrying ~20% more weight.
  5. Two model FAMILIES, not base-vs-finetune. Precedent for cross-architecture crosscoding:
     Jiralerspong & Bricken, arXiv:2602.11729 (2026). Scope the novelty honestly.
  6. Capacity anchored to the REFERENCE model A: hidden = 8*d_A, k = 0.20*d_A. For ESM-2/RITA
     that is hidden=10240, k=256 — which reproduces the paper's regime for BOTH models at once:
     k/embed = 256/1280 = 20.0% (ESM-2) and 256/1536 = 16.7% (RITA), the two ratios the paper's
     own SAEs ran at, with k/hidden = 256/10240 = 2.50%. For the d=320 controlled pair it gives
     hidden=2560, k=64 (20.0% / 2.50%). One rule, both comparisons, no new free parameter.

  Precise label (no "X-like"): a BatchTopK model-diffing crosscoder (Lindsey et al. 2024) with
  the Minder et al. 2025 sparsity-artifact corrections, trained on per-token-normalised residue
  activations of two residue-tokenised PLMs at matched relative depth.

MPS FOOTGUNS inherited from sae.py — do not "clean up":
  * use `param.copy_(...)`, never `param.data = ...`
  * TopK via `torch.where` masking, never `zeros_like` + in-place `scatter_`
"""
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ======================================================================================
#                                      MODEL
# ======================================================================================

class CrossCoder(nn.Module):
    """BatchTopK model-diffing crosscoder over two models A and B.

    Shared latent f_j(x) = TopK_batch( e^A_j . (x^A - b^A_pre) + e^B_j . (x^B - b^B_pre) + b_enc )
    Per-model reconstruction  x^m_hat = sum_j f_j d^m_j + b^m_dec.

    d_a and d_b MAY DIFFER (per-model encoders/decoders), which is what lets this run on
    ESM-2 (1280) vs RITA (1536).
    """

    def __init__(self, d_a: int, d_b: int, hidden_dim: int, k_sparse: int,
                 k_aux: int = 64, aux_coeff: float = 1 / 32,
                 dead_threshold: int = 1_000_000, theta_ema: float = 0.99,
                 freeze: str = "none", tau: float = 0.8, seed: int = 42):
        """freeze: the E4 null baseline (Korznikov et al. 2026, adapted to crosscoders — they
        explicitly leave crosscoder randomisation as an open problem, so this is ours).

          none                 normal training
          frozen_encoder       encoders random + frozen -> the FEATURES are random, but the
                               decoders (and hence Delta_norm) still learn. THE informative test:
                               does a random feature basis still produce a structural gradient
                               in Delta_norm?
          soft_frozen_decoder  decoders held within cosine tau of random init.

        frozen_decoder is deliberately REFUSED: with both decoders pinned at random init,
        Delta_norm is a deterministic function of the initialisation and cannot move, so the test
        is vacuous (rho ~ 0 by construction, telling us nothing).
        """
        super().__init__()
        if freeze == "frozen_decoder":
            raise ValueError(
                "frozen_decoder is vacuous for a crosscoder: Delta_norm is computed FROM the "
                "decoder norms, so freezing both decoders fixes the metric at its random init. "
                "Use frozen_encoder (random features, learned decoders) or soft_frozen_decoder.")
        if freeze not in ("none", "frozen_encoder", "soft_frozen_decoder"):
            raise ValueError(f"unknown freeze mode {freeze!r}")
        self.freeze_kind = freeze
        self.tau = tau
        self._freeze_seed = seed
        self.d_a, self.d_b = d_a, d_b
        self.hidden_dim = hidden_dim
        self.k_sparse = k_sparse
        self.k_aux = k_aux
        self.aux_coeff = aux_coeff
        self.dead_threshold = dead_threshold
        self.theta_ema = theta_ema

        self.enc_a = nn.Linear(d_a, hidden_dim, bias=False)
        self.enc_b = nn.Linear(d_b, hidden_dim, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))

        self.dec_a = nn.Linear(hidden_dim, d_a, bias=False)
        self.dec_b = nn.Linear(hidden_dim, d_b, bias=False)
        self.b_dec_a = nn.Parameter(torch.zeros(d_a))
        self.b_dec_b = nn.Parameter(torch.zeros(d_b))

        self.register_buffer("b_pre_a", torch.zeros(d_a))
        self.register_buffer("b_pre_b", torch.zeros(d_b))
        # BatchTopK has no fixed per-row k, so inference needs a learned threshold
        # (Bussmann et al. 2024): EMA of the smallest selected pre-activation.
        self.register_buffer("theta", torch.zeros(()))
        self.register_buffer("theta_init", torch.zeros((), dtype=torch.bool))
        self.register_buffer("tokens_since_activation", torch.zeros(hidden_dim, dtype=torch.long))

        self._init_weights()
        self._apply_freeze()

    def _init_weights(self):
        for dec in (self.dec_a, self.dec_b):
            nn.init.xavier_uniform_(dec.weight)
        with torch.no_grad():
            self.normalize_decoders()
            # tied init, per-model (encoder = that model's decoder transpose)
            self.enc_a.weight.copy_(self.dec_a.weight.T)
            self.enc_b.weight.copy_(self.dec_b.weight.T)
        nn.init.zeros_(self.b_enc)

    def _apply_freeze(self):
        if self.freeze_kind == "none":
            return
        g = torch.Generator().manual_seed(self._freeze_seed)
        if self.freeze_kind == "frozen_encoder":
            with torch.no_grad():
                for enc, d in ((self.enc_a, self.d_a), (self.enc_b, self.d_b)):
                    W = torch.randn(self.hidden_dim, d, generator=g)
                    enc.weight.copy_(F.normalize(W, dim=1))
            self.enc_a.weight.requires_grad_(False)
            self.enc_b.weight.requires_grad_(False)
        else:  # soft_frozen_decoder
            with torch.no_grad():
                for dec, d in ((self.dec_a, self.d_a), (self.dec_b, self.d_b)):
                    W = torch.randn(d, self.hidden_dim, generator=g)
                    dec.weight.copy_(F.normalize(W, dim=0))
                self.normalize_decoders()
            self.register_buffer("dec_a_init", self.dec_a.weight.detach().clone())
            self.register_buffer("dec_b_init", self.dec_b.weight.detach().clone())

    @torch.no_grad()
    def _project_decoders_to_cone(self):
        """Keep each model's decoder column within cosine tau of its random init.
        Same construction as sae_variants._project_to_cone, including the float32
        cancellation tolerance (1e-5, NOT machine epsilon — see that file)."""
        for W, W0 in ((self.dec_a, self.dec_a_init), (self.dec_b, self.dec_b_init)):
            V = W.weight
            U = F.normalize(W0, dim=0)
            nrm = V.norm(dim=0, keepdim=True).clamp_min(1e-8)
            Vh = V / nrm
            cos = (Vh * U).sum(0, keepdim=True)
            bad = (cos < self.tau).squeeze(0)
            if not bool(bad.any()):
                continue
            perp = Vh - cos * U
            pn = perp.norm(dim=0, keepdim=True)
            degen = (pn < 1e-5).squeeze(0)
            if bool(degen.any()):
                fb = torch.zeros_like(U[:, degen]); fb[0] = 1.0
                fb = F.normalize(fb - (fb * U[:, degen]).sum(0, keepdim=True) * U[:, degen], dim=0)
                perp[:, degen] = fb
                pn = perp.norm(dim=0, keepdim=True).clamp_min(1e-8)
            p_hat = perp / pn.clamp_min(1e-8)
            w = self.tau * U + math.sqrt(max(0.0, 1 - self.tau ** 2)) * p_hat
            V[:, bad] = (F.normalize(w, dim=0) * nrm)[:, bad]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- decoder normalisation: the one place a naive copy of sae.py breaks the metric ----
    @torch.no_grad()
    def normalize_decoders(self):
        """Unit-norm the CONCATENATED decoder [d^A_j ; d^B_j].

        This pins each latent's total decoder scale (the usual stabiliser) while leaving the
        SPLIT between models free — and that split is exactly what Delta_norm measures.
        Normalising per model would force ||d^A_j|| == ||d^B_j|| == 1 and make Delta_norm
        identically 0.5 for every latent, i.e. destroy the experiment.
        """
        na = self.dec_a.weight.pow(2).sum(dim=0)          # (H,)
        nb = self.dec_b.weight.pow(2).sum(dim=0)
        joint = (na + nb).sqrt().clamp_min(1e-8)
        self.dec_a.weight.div_(joint)
        self.dec_b.weight.div_(joint)

    def _batch_topk(self, z_pre: torch.Tensor) -> torch.Tensor:
        """BatchTopK: keep the k*B largest pre-activations across the WHOLE batch."""
        B, H = z_pre.shape
        n = min(self.k_sparse * B, z_pre.numel())
        flat_vals = torch.topk(z_pre.flatten(), n).values
        thresh = flat_vals[-1].detach()
        z = torch.where((z_pre >= thresh) & (z_pre > 0), z_pre, torch.zeros_like(z_pre))
        if self.training:
            with torch.no_grad():
                pos = flat_vals[flat_vals > 0]
                if pos.numel():
                    m = pos.min()
                    if bool(self.theta_init):
                        self.theta.mul_(self.theta_ema).add_((1 - self.theta_ema) * m)
                    else:
                        self.theta.copy_(m)
                        self.theta_init.fill_(True)
        return z

    def encode(self, xa: torch.Tensor, xb: torch.Tensor,
               use_threshold: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, z_pre).

        use_threshold=None -> follow self.training. At extraction time the JumpReLU threshold
        is what makes z independent of batch composition; using BatchTopK there would make
        every residue's activation depend on which other residues shared its batch, silently
        corrupting L_struct.
        """
        z_pre = self.enc_a(xa - self.b_pre_a) + self.enc_b(xb - self.b_pre_b) + self.b_enc
        thresholded = (not self.training) if use_threshold is None else use_threshold
        if thresholded and bool(self.theta_init):
            z = torch.where(z_pre > self.theta, z_pre, torch.zeros_like(z_pre))
        else:
            z = self._batch_topk(z_pre)
        return z, z_pre

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dec_a(z) + self.b_dec_a, self.dec_b(z) + self.b_dec_b

    def forward(self, xa, xb):
        z, z_pre = self.encode(xa, xb)
        ra, rb = self.decode(z)
        return ra, rb, z, z_pre

    @torch.no_grad()
    def _update_tracking(self, z: torch.Tensor):
        fired = (z > 0).any(dim=0)
        self.tokens_since_activation[fired] = 0
        self.tokens_since_activation[~fired] += z.shape[0]

    def loss(self, xa, xb, ra, rb, z_pre, include_aux=True) -> Dict[str, torch.Tensor]:
        """Per-model MSE averaged per dimension, then summed -> equal weight per model.

        F.mse_loss already averages over every element, so mse(ra,xa) is a per-dimension mean
        for model A regardless of d_a. Summing the two therefore weights the models equally
        rather than in proportion to their width (deviation 4).
        """
        rec_a = F.mse_loss(ra, xa)
        rec_b = F.mse_loss(rb, xb)
        recon = rec_a + rec_b
        aux = torch.zeros((), device=xa.device, dtype=xa.dtype)
        if include_aux and self.training:
            aux = self._aux_loss(xa, xb, ra, rb, z_pre)
        return {"total": recon + self.aux_coeff * aux, "recon": recon,
                "recon_a": rec_a, "recon_b": rec_b, "aux": aux}

    def _aux_loss(self, xa, xb, ra, rb, z_pre):
        """AuxK dead-latent recovery (Gao et al. 2024), applied to the joint residual."""
        dead = self.tokens_since_activation >= self.dead_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return torch.zeros((), device=xa.device, dtype=xa.dtype)
        z_dead = z_pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
        kk = min(self.k_aux, n_dead)
        vals, idx = torch.topk(z_dead, kk, dim=1)
        z_aux = torch.zeros_like(z_pre)
        z_aux.scatter_(1, idx, F.relu(vals))
        ea, eb = self.dec_a(z_aux), self.dec_b(z_aux)
        return F.mse_loss(ea, xa - ra) + F.mse_loss(eb, xb - rb)


# ======================================================================================
#                                     TRAINING
# ======================================================================================

def _batches(n, bs, rng=None):
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_crosscoder(XA: np.ndarray, XB: np.ndarray, *, hidden_dim: int, k_sparse: int,
                     device: str = "cpu", epochs: int = 60, lr: float = 5e-5,
                     batch_size: int = 4096, k_aux: int = 64, seed: int = 42,
                     log_every: int = 10, verbose: bool = True,
                     freeze: str = "none", tau: float = 0.8) -> CrossCoder:
    """Train on TRAIN rows only. XA/XB must be row-aligned (same residue per row)."""
    assert XA.shape[0] == XB.shape[0], "row mismatch — XA and XB must be residue-aligned"
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    cc = CrossCoder(XA.shape[1], XB.shape[1], hidden_dim, k_sparse, k_aux=k_aux,
                    freeze=freeze, tau=tau, seed=seed).to(device)
    with torch.no_grad():
        cc.b_pre_a.copy_(torch.from_numpy(XA.mean(0).astype(np.float32)))
        cc.b_pre_b.copy_(torch.from_numpy(XB.mean(0).astype(np.float32)))
        cc.b_dec_a.copy_(cc.b_pre_a)
        cc.b_dec_b.copy_(cc.b_pre_b)

    opt = torch.optim.Adam(cc.trainable_parameters(), lr=lr)
    n = XA.shape[0]
    cc.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for sel in _batches(n, batch_size, rng):
            xa = torch.from_numpy(XA[sel]).to(device)
            xb = torch.from_numpy(XB[sel]).to(device)
            ra, rb, z, z_pre = cc(xa, xb)
            out = cc.loss(xa, xb, ra, rb, z_pre)
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            opt.step()
            cc.normalize_decoders()
            if cc.freeze_kind == "soft_frozen_decoder":
                cc._project_decoders_to_cone()
            cc._update_tracking(z)
            tot += out["recon"].detach().item(); nb += 1
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            dead = float((cc.tokens_since_activation >= cc.dead_threshold).float().mean()) * 100
            print(f"    epoch {ep:>3}/{epochs}  recon {tot/max(nb,1):.5f}  dead {dead:.1f}%",
                  flush=True)
    cc.eval()
    return cc


@torch.no_grad()
def encode_all(cc: CrossCoder, XA: np.ndarray, XB: np.ndarray, device="cpu",
               batch_size=8192) -> np.ndarray:
    """Extract shared latents for every row, using the JumpReLU threshold (batch-independent)."""
    cc.eval()
    out = np.empty((XA.shape[0], cc.hidden_dim), dtype=np.float32)
    for i in range(0, XA.shape[0], batch_size):
        xa = torch.from_numpy(XA[i:i + batch_size]).to(device)
        xb = torch.from_numpy(XB[i:i + batch_size]).to(device)
        z, _ = cc.encode(xa, xb, use_threshold=True)
        out[i:i + batch_size] = z.cpu().numpy()
    return out


@torch.no_grad()
def val_ev(cc: CrossCoder, XA, XB, device="cpu", batch_size=8192) -> Dict[str, float]:
    """Per-model val explained variance — the project's degeneracy gate (>=0.99 == degenerate)."""
    cc.eval()
    num = {"a": 0.0, "b": 0.0}
    sse = {"a": 0.0, "b": 0.0}
    mu_a = torch.from_numpy(XA.mean(0)).to(device)
    mu_b = torch.from_numpy(XB.mean(0)).to(device)
    for i in range(0, XA.shape[0], batch_size):
        xa = torch.from_numpy(XA[i:i + batch_size]).to(device)
        xb = torch.from_numpy(XB[i:i + batch_size]).to(device)
        z, _ = cc.encode(xa, xb, use_threshold=True)
        ra, rb = cc.decode(z)
        num["a"] += float(((xa - ra) ** 2).sum()); sse["a"] += float(((xa - mu_a) ** 2).sum())
        num["b"] += float(((xb - rb) ** 2).sum()); sse["b"] += float(((xb - mu_b) ** 2).sum())
    return {"val_EV_a": 1.0 - num["a"] / max(sse["a"], 1e-12),
            "val_EV_b": 1.0 - num["b"] / max(sse["b"], 1e-12)}


# ======================================================================================
#                                    DIAGNOSTICS
# ======================================================================================

def delta_norm(cc: CrossCoder) -> Dict[str, np.ndarray]:
    """Relative decoder norm difference (Lindsey et al. 2024, as written in Minder et al. 2025):

        Delta_norm(j) = 0.5 * [ (||d^B_j|| - ||d^A_j||) / max(||d^A_j||, ||d^B_j||) + 1 ]

    ~0 -> A-only, ~1 -> B-only, ~0.5 -> shared.

    Also returns a sqrt(D)-normalised variant. The two models here can differ in width, so a
    raw norm comparison partly reflects dimensionality; if the two variants disagree, the width
    difference is doing the work and the cross-architecture claim weakens. Report both.
    """
    with torch.no_grad():
        na = cc.dec_a.weight.norm(dim=0).cpu().numpy().astype(np.float64)
        nb = cc.dec_b.weight.norm(dim=0).cpu().numpy().astype(np.float64)
    den = np.maximum(np.maximum(na, nb), 1e-12)
    raw = 0.5 * ((nb - na) / den + 1.0)
    sa, sb = na / np.sqrt(cc.d_a), nb / np.sqrt(cc.d_b)
    den_s = np.maximum(np.maximum(sa, sb), 1e-12)
    scaled = 0.5 * ((sb - sa) / den_s + 1.0)
    return {"delta_norm": raw, "delta_norm_sqrtD": scaled, "norm_a": na, "norm_b": nb}


@torch.no_grad()
def latent_scaling(cc: CrossCoder, XA, XB, assign: np.ndarray, device="cpu",
                   batch_size=8192) -> Dict[str, np.ndarray]:
    """Latent Scaling (Minder et al. 2025 §2.3) — the shared/unique VALIDITY diagnostic.

    For latent j assigned to model m* (by Delta_norm), with m' the other model:

        beta^y,m_j = argmin_beta  sum_i || beta * f_j(x_i) * d^{m*}_j  -  y^m(x_i) ||^2

    and the ratio nu^y_j = beta^{y,m'}_j / beta^{y,m*}_j. Near 0 => genuinely m*-specific;
    near 1 => equally present in both, i.e. the "unique" label is spurious.

      nu       (y = the activation h)         overall specificity
      nu_eps   (y = reconstruction error eps) COMPLETE SHRINKAGE   — info sits in m's residual
      nu_r     (y = reconstruction h_tilde)   LATENT DECOUPLING    — info captured by other latents

    The closed-form beta shares the denominator sum_i f_ji^2 ||d_j||^2 across the two models, so
    the RATIO needs only the numerators — which is what we accumulate.

    !! CROSS-ARCHITECTURE LIMITATION !!
    Equation 4 dots the ASSIGNED model's decoder direction d^{m*}_j against the OTHER model's
    activations. That requires d_a == d_b. For ESM-2 (1280) vs RITA (1536) it is undefined, so
    this returns {"applicable": False} and the caller must NOT make shared/unique claims on the
    strength of a diagnostic that was never run. BatchTopK is still the preventive fix; Latent
    Scaling is the confirmatory one, and it is only available for the same-width (MLM/CLM) pair.
    """
    if cc.d_a != cc.d_b:
        return {"applicable": False, "reason":
                f"Latent Scaling needs d_a == d_b (got {cc.d_a} vs {cc.d_b}); "
                "Minder et al. Eq.4 dots the assigned model's decoder into the other "
                "model's activation space."}

    H = cc.hidden_dim
    to_b = torch.from_numpy((assign == 1).astype(np.float32)).to(device).unsqueeze(1)  # (H,1)
    # per-latent assigned direction: rows of (H, d)
    Dstar = cc.dec_a.weight.T * (1 - to_b) + cc.dec_b.weight.T * to_b                  # (H,d)

    keys = [(m, y) for m in ("a", "b") for y in ("h", "err", "rec")]
    num = {k: torch.zeros(H, dtype=torch.float64, device=device) for k in keys}
    s2 = torch.zeros(H, dtype=torch.float64, device=device)

    for i in range(0, XA.shape[0], batch_size):
        xa = torch.from_numpy(XA[i:i + batch_size]).to(device)
        xb = torch.from_numpy(XB[i:i + batch_size]).to(device)
        z, _ = cc.encode(xa, xb, use_threshold=True)
        ra, rb = cc.decode(z)
        zf = z.double()
        s2 += (zf ** 2).sum(0)
        for m, x, r in (("a", xa, ra), ("b", xb, rb)):
            for y, tgt in (("h", x), ("err", x - r), ("rec", r)):
                num[(m, y)] += (zf * (tgt.double() @ Dstar.double().T)).sum(0)

    n = {k: v.cpu().numpy() for k, v in num.items()}

    def ratio(y):
        # numerator for the OTHER model over the numerator for the ASSIGNED model
        star = np.where(assign == 1, n[("b", y)], n[("a", y)])
        other = np.where(assign == 1, n[("a", y)], n[("b", y)])
        out = np.full(H, np.nan)
        ok = np.abs(star) > 1e-12
        out[ok] = other[ok] / star[ok]
        return out

    return {"applicable": True, "nu": ratio("h"), "nu_eps": ratio("err"), "nu_r": ratio("rec"),
            "sum_f2": s2.cpu().numpy()}


def dead_stats(cc: CrossCoder, Z: np.ndarray) -> Dict[str, float]:
    """Dead / always-on counts — part of the same validity gate used in A3."""
    fired = (Z != 0).sum(0)
    n = Z.shape[0]
    return {"pct_dead": float((fired == 0).mean() * 100),
            "pct_always_on_90": float((fired > 0.9 * n).mean() * 100),
            "mean_l0": float((Z != 0).sum(1).mean())}


def latent_covariates(cc: CrossCoder, Z: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-latent nuisance covariates for the E5 partial-correlation guard (R5).

    The rank-gap threat: if one arm's activations occupy far more effective dimensions than the
    other's, the SHARED dictionary reconstructs the easier arm better and shifts decoder mass
    toward it. That drift looks exactly like "objective-specific features". Measured signature on
    ESM-2/RITA: reconstruction cost -0.09 vs -0.31, Delta_norm drifting 0.579 -> 0.667 -> 0.743
    across training. Spearman rho is immune to a UNIFORM shift but not to a GRADED one.

    So before trusting any Delta_norm <-> L_struct relationship we partial out the things a
    capacity imbalance would move:
      activity   mean activation of the latent (how often/hard it fires)
      firing     fraction of residues where it is non-zero
      var_a/b    the variance it explains in each arm: ||d^m_j||^2 * Var(f_j)
      var_total  their sum (overall importance, independent of the split)
    """
    with torch.no_grad():
        na = cc.dec_a.weight.norm(dim=0).cpu().numpy().astype(np.float64)
        nb = cc.dec_b.weight.norm(dim=0).cpu().numpy().astype(np.float64)
    act = Z.mean(0).astype(np.float64)
    fire = (Z != 0).mean(0).astype(np.float64)
    fvar = Z.var(0).astype(np.float64)
    return {"activity": act, "firing_rate": fire, "latent_var": fvar,
            "var_a": (na ** 2) * fvar, "var_b": (nb ** 2) * fvar,
            "var_total": (na ** 2 + nb ** 2) * fvar}
