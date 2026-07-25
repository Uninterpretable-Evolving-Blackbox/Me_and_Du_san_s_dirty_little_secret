#!/usr/bin/env python3
"""
sae_variants.py — SAE architecture variants and the FROZEN-COMPONENT null baselines.

TWO JOBS
  (A6/C7) The L1 control the project has never run. Korznikov, Galichin, Dontsov, Rogov,
          Oseledets & Tutubalina, "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random
          Baselines?" (arXiv:2602.14111, 2026) show that SAEs with randomly-initialised, FROZEN
          components match fully-trained SAEs on interpretability (0.87 vs 0.90), sparse probing
          (0.69 vs 0.72) and causal editing (0.73 vs 0.72) — on BatchTopK and JumpReLU, not
          strawman architectures. Their synthetic result is worse: ~9% of ground-truth features
          recovered at 71% explained variance.

          The project's existing randomised-weights control (C1) randomises the *PLM*. That
          answers a different question ("is the signal in the protein model?"). This answers
          "is the signal in the SAE's LEARNING, or would a random dictionary do?"

  (A7)    Architecture robustness. SAEBench (Karvonen et al. 2025) finds no single winning
          architecture, and that they separate mainly at L0 < 100 — the project runs k=256,
          inside the band where they are hard to distinguish. Untested here, so we test it.

THE THREE NULL BASELINES (exactly as defined in Korznikov et al. §4.1)
  frozen_decoder      decoder vectors randomly initialised and FROZEN for all of training.
                      Their best init: `cov`.
  soft_frozen_decoder decoder randomly initialised, then constrained by projection to stay
                      within cosine similarity tau (=0.8) of its initial value. Motivated by
                      their observation that SAE loss plateaus early while decoder vectors barely
                      move — the "lazy training" hypothesis. Their best init: `iso`.
  frozen_encoder      encoder randomly initialised and FROZEN, so each feature's activation
                      pattern is fixed at init; only threshold/decoder learn. Best init: `iso`.

  Init schemes, also theirs:
    iso  vectors uniform on the unit sphere
    cov  Gaussian with covariance estimated from the real activations, then normalised

ARCHITECTURES
  topk        the project's existing choice (Gao et al. 2024). Pins L0 = k EXACTLY, which is why
              it is the right default for a *comparative* study: L_struct is a co-activation
              statistic, so an L0 mismatch between the two models being compared would be a
              direct confound. L1/Gated/JumpReLU let L0 emerge from the data.
  batchtopk   Bussmann et al. 2024. Selects the top k*B activations across the whole batch, so
              L0 = k only ON AVERAGE. SAEBench ranks it slightly ABOVE TopK at low L0, making it a
              tougher robustness check rather than a friendly one. Inference uses a JumpReLU
              threshold (EMA of the smallest selected pre-activation) so activations do not
              depend on batch composition.

  JumpReLU is deliberately NOT implemented. It needs a straight-through estimator for the
  threshold gradient that we cannot validate against the authors' numbers, and shipping a
  subtly-wrong implementation into a rigor program is worse than not shipping it. If a reviewer
  asks, say that: BatchTopK covers the architecture-robustness question and is ranked above TopK.
"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sae import SparseAutoencoder

ACTIVATIONS = ("topk", "batchtopk")
FREEZE_MODES = ("none", "frozen_decoder", "soft_frozen_decoder", "frozen_encoder")
DEFAULT_INIT = {"frozen_decoder": "cov", "soft_frozen_decoder": "iso", "frozen_encoder": "iso"}


def _random_directions(n_vec: int, dim: int, scheme: str, cov: Optional[np.ndarray],
                       gen: torch.Generator) -> torch.Tensor:
    """(dim, n_vec) random unit columns. `iso` = uniform on the sphere; `cov` = Gaussian with the
    activations' covariance, then normalised (Korznikov et al. Appendix E)."""
    if scheme == "cov":
        if cov is None:
            raise ValueError("cov init requires the activation covariance")
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
        z = torch.randn(dim, n_vec, generator=gen).numpy()
        W = (L @ z).astype(np.float32)
        W = torch.from_numpy(W)
    else:
        W = torch.randn(dim, n_vec, generator=gen)
    return F.normalize(W, dim=0)


class VariantSAE(SparseAutoencoder):
    """TopK SAE plus BatchTopK, plus the three frozen-component null baselines."""

    def __init__(self, input_dim, expansion=8, k_sparse=256, k_aux=64,
                 activation="topk", freeze="none", init_scheme=None, tau=0.8,
                 act_cov=None, seed=42, theta_ema=0.99, **kw):
        super().__init__(input_dim=input_dim, expansion=expansion, k_sparse=k_sparse,
                         k_aux=k_aux, **kw)
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if freeze not in FREEZE_MODES:
            raise ValueError(f"freeze must be one of {FREEZE_MODES}")
        self.activation_kind = activation
        self.freeze_kind = freeze
        self.tau = tau
        self.register_buffer("theta", torch.zeros(()))
        self.register_buffer("theta_init", torch.zeros((), dtype=torch.bool))
        self.theta_ema = theta_ema

        if freeze != "none":
            scheme = init_scheme or DEFAULT_INIT[freeze]
            gen = torch.Generator().manual_seed(seed)
            if freeze in ("frozen_decoder", "soft_frozen_decoder"):
                W = _random_directions(self.hidden_dim, input_dim, scheme, act_cov, gen)
                with torch.no_grad():
                    self.decoder.weight.copy_(W)                    # (input_dim, hidden_dim)
                self.register_buffer("dec_init", self.decoder.weight.detach().clone())
                if freeze == "frozen_decoder":
                    self.decoder.weight.requires_grad_(False)
            else:  # frozen_encoder
                W = _random_directions(self.hidden_dim, input_dim, scheme, act_cov, gen)
                with torch.no_grad():
                    self.encoder.weight.copy_(W.T)                  # (hidden_dim, input_dim)
                self.encoder.weight.requires_grad_(False)
            self.init_scheme = scheme
        else:
            self.init_scheme = None

    # ---------------- activation ----------------
    def _batch_topk_activation(self, z_pre):
        B, H = z_pre.shape
        n = min(self.k_sparse * B, z_pre.numel())
        vals = torch.topk(z_pre.flatten(), n).values
        thresh = vals[-1].detach()
        z = torch.where((z_pre >= thresh) & (z_pre > 0), z_pre, torch.zeros_like(z_pre))
        if self.training:
            with torch.no_grad():
                pos = vals[vals > 0]
                if pos.numel():
                    m = pos.min()
                    if bool(self.theta_init):
                        self.theta.mul_(self.theta_ema).add_((1 - self.theta_ema) * m)
                    else:
                        self.theta.copy_(m); self.theta_init.fill_(True)
        return z

    def encode(self, x, use_threshold=None):
        x_centered = x - self.b_pre
        z_pre = self.encoder(x_centered)
        if self.activation_kind == "topk":
            return self._topk_activation(z_pre), z_pre
        thresholded = (not self.training) if use_threshold is None else use_threshold
        if thresholded and bool(self.theta_init):
            return torch.where(z_pre > self.theta, z_pre, torch.zeros_like(z_pre)), z_pre
        return self._batch_topk_activation(z_pre), z_pre

    # ---------------- freezing ----------------
    @torch.no_grad()
    def normalize_decoder(self):
        """Skip entirely when the decoder is hard-frozen (it is unit-norm at init and must not
        move). For the soft-frozen variant, normalise first, then re-project onto the cone."""
        if self.freeze_kind == "frozen_decoder":
            return
        super().normalize_decoder()
        if self.freeze_kind == "soft_frozen_decoder":
            self._project_to_cone()

    @torch.no_grad()
    def _project_to_cone(self):
        """Force every decoder column to keep cosine >= tau with its initial direction.

        For a column v with init direction u (unit), if cos(v,u) < tau, replace it with the
        nearest vector on the cone boundary:  w = ||v|| * (tau*u + sqrt(1-tau^2)*p_hat),
        where p_hat is the unit component of v orthogonal to u. Norm is preserved.
        """
        V = self.decoder.weight                    # (D, H)
        U = F.normalize(self.dec_init, dim=0)
        nrm = V.norm(dim=0, keepdim=True).clamp_min(1e-8)
        Vh = V / nrm
        cos = (Vh * U).sum(0, keepdim=True)        # (1, H)
        bad = (cos < self.tau).squeeze(0)
        if not bool(bad.any()):
            return
        perp = Vh - cos * U
        pn = perp.norm(dim=0, keepdim=True)
        # If v is (anti)parallel to u the perpendicular is undefined and we need a fallback axis.
        # The threshold must be a float32 CANCELLATION tolerance, not machine epsilon: for an
        # exactly-antiparallel column the residual of (Vh - cos*U) is ~1e-7, which sails past a
        # 1e-8 test and then gets divided by its own norm, amplifying pure noise into a direction
        # that is not orthogonal to u. Measured symptom: an antiparallel column projected to
        # cos 0.9923 instead of tau=0.8 — constraint satisfied, but not the nearest point.
        degen = (pn < 1e-5).squeeze(0)
        if bool(degen.any()):
            fallback = torch.zeros_like(U[:, degen])
            fallback[0] = 1.0
            fallback = F.normalize(fallback - (fallback * U[:, degen]).sum(0, keepdim=True) * U[:, degen], dim=0)
            perp[:, degen] = fallback
            pn = perp.norm(dim=0, keepdim=True).clamp_min(1e-8)
        p_hat = perp / pn.clamp_min(1e-8)
        w = self.tau * U + math.sqrt(max(0.0, 1 - self.tau ** 2)) * p_hat
        new = F.normalize(w, dim=0) * nrm
        V[:, bad] = new[:, bad]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def activation_covariance(X: np.ndarray, max_rows: int = 200_000) -> np.ndarray:
    """D x D covariance of the (already preprocessed) activations, for `cov` init."""
    if X.shape[0] > max_rows:
        idx = np.random.default_rng(0).choice(X.shape[0], max_rows, replace=False)
        X = X[idx]
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(0, keepdims=True)
    return (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)


def train_variant_sae(X: np.ndarray, input_dim: int, *, device="cpu", epochs=60, lr=5e-5,
                      batch_size=4096, expansion=8, k_sparse=256, k_aux=64,
                      activation="topk", freeze="none", init_scheme=None, tau=0.8,
                      seed=42, dead_threshold=1_000_000, verbose=True,
                      log_every=20) -> VariantSAE:
    """Byte-for-byte the same optimisation recipe as train_sae — this is load-bearing.

    A6 compares a FROZEN arm against the trained TopK reference arm, and the reference arm is
    produced by train_sae(). If the two paths optimise differently, the "retained vs trained"
    ratio folds an optimiser difference into the freezing effect, on the one control the paper's
    fatal objection (R2) rests on. Measured on real ESM-2 L16 per-token-normed activations with
    identical architecture (topk, freeze=none), seed 42, 40 epochs: val_EV +0.3323 via a plain
    constant-lr Adam vs +0.1968 via train_sae's recipe — a 69% relative gap from optimisation
    alone, biasing the ratio TOWARD "a random dictionary reproduces H1".

    So we replicate train_sae.py exactly:
      * AdamW(betas=(0.9,0.999), weight_decay=0.0)          train_sae.py:304
      * CosineAnnealingLR(T_max=epochs, eta_min=lr*0.1)     train_sae.py:314, stepped per epoch
      * clip_grad_norm_(params, 1.0)                        train_sae.py:356
      * drop_last=True batching                             (DataLoader default there)
      * dead_threshold=1_000_000                            train_sae.py:220 — NOT sae.py's 1e7
    The dead_threshold default is a real trap: at 264,603 train residues x 60 epochs = 15.9M
    tokens, 1e6 lets AuxK revive dead latents from ~epoch 4 while 1e7 delays it to ~epoch 38.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cov = activation_covariance(X) if (freeze in ("frozen_decoder",) and
                                       (init_scheme or DEFAULT_INIT[freeze]) == "cov") else None
    sae = VariantSAE(input_dim=input_dim, expansion=expansion, k_sparse=k_sparse, k_aux=k_aux,
                     activation=activation, freeze=freeze, init_scheme=init_scheme, tau=tau,
                     act_cov=cov, seed=seed, dead_threshold=dead_threshold).to(device)
    with torch.no_grad():
        sae.b_pre.copy_(torch.from_numpy(X.mean(0).astype(np.float32)).to(device))
    params = sae.trainable_parameters()
    if not params:
        raise RuntimeError("no trainable parameters — check the freeze mode")
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    sae.train()
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot, nb = 0.0, 0
        # drop_last: match train_sae's DataLoader so the step count per epoch is identical
        for i in range(0, n - batch_size + 1, batch_size):
            xb = torch.from_numpy(X[idx[i:i + batch_size]]).to(device)
            z, z_pre = sae.encode(xb)
            xh = sae.decode(z)
            out = sae.loss_fn(xb, xh, z, z_pre)
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sae.normalize_decoder()
            sae._update_activation_tracking(z.detach())
            tot += out["recon"].detach().item(); nb += 1
        sched.step()
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep:>3}/{epochs}  recon {tot/max(nb,1):.5f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
    sae.eval()
    return sae


@torch.no_grad()
def encode_all_variant(sae: VariantSAE, X: np.ndarray, device="cpu", batch_size=8192) -> np.ndarray:
    sae.eval()
    out = np.empty((X.shape[0], sae.hidden_dim), dtype=np.float32)
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        z, _ = sae.encode(xb, use_threshold=True)
        out[i:i + batch_size] = z.cpu().numpy()
    return out


@torch.no_grad()
def val_ev_variant(sae: VariantSAE, Xv: np.ndarray, device="cpu", batch_size=8192) -> float:
    sae.eval()
    mu = torch.from_numpy(Xv.mean(0)).to(device)
    num = den = 0.0
    for i in range(0, Xv.shape[0], batch_size):
        xb = torch.from_numpy(Xv[i:i + batch_size]).to(device)
        z, _ = sae.encode(xb, use_threshold=True)
        rec = sae.decode(z)
        num += float(((xb - rec) ** 2).sum()); den += float(((xb - mu) ** 2).sum())
    return 1.0 - num / max(den, 1e-12)
