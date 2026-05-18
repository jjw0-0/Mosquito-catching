from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import KFold

from .data import load_competition_data, write_submission
from .features import build_features, physical_predictions
from .metrics import score_summary
from .train import (
    correction_from_prediction,
    find_best_shrink,
    sample_weight_for,
    trajectory_basis_and_scale,
)


@dataclass
class TorchConfig:
    zip_path: str = "/Users/jjw/Downloads/open.zip"
    out_dir: str = "outputs_torch"
    residual_base: str = "accel_c0.58"
    folds: int = 5
    seed: int = 20260519
    epochs: int = 160
    patience: int = 35
    batch_size: int = 512
    hidden: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.12
    lr: float = 8e-4
    weight_decay: float = 2e-4
    boundary_loss: float = 0.30
    noise_std: float = 0.015
    device: str = "auto"
    blend_with: str = "outputs/submission_multibase_local_trap_blend.csv"
    no_submissions: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _project_local(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ntc,nkc->ntk", vectors, basis)


def build_sequence_inputs(
    xyz: np.ndarray,
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build local-frame sequence and small dense features.

    The network intentionally predicts a residual in the same local frame used
    by the strongest HGB model.  This makes GPU augmentation/dropout useful
    without forcing the model to relearn translation and scale invariance from
    only 10k trajectories.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    basis, scale = trajectory_basis_and_scale(xyz)
    last = xyz[:, -1]
    d1 = np.diff(xyz, axis=1)
    d2 = np.diff(d1, axis=1)

    centered = xyz - last[:, None, :]
    vel = np.concatenate([np.zeros_like(xyz[:, :1]), d1], axis=1)
    acc = np.concatenate([np.zeros_like(xyz[:, :2]), d2], axis=1)

    centered_local = _project_local(centered, basis) / scale[:, None, :]
    vel_local = _project_local(vel, basis) / scale[:, None, :]
    acc_local = _project_local(acc, basis) / scale[:, None, :]
    t = np.linspace(-1.0, 1.0, xyz.shape[1], dtype=np.float64)
    t_feat = np.broadcast_to(t[None, :, None], (xyz.shape[0], xyz.shape[1], 1))
    seq = np.concatenate([centered_local, vel_local, acc_local, t_feat], axis=2)

    base_delta = base - last
    base_delta_local = np.einsum("nc,nkc->nk", base_delta, basis) / scale
    recent_speed = np.linalg.norm(d1[:, -3:].mean(axis=1), axis=1, keepdims=True)
    last_speed = np.linalg.norm(d1[:, -1], axis=1, keepdims=True)
    last_acc = np.linalg.norm(d1[:, -1] - d1[:, -2], axis=1, keepdims=True)
    dense_small = np.concatenate(
        [
            base_delta_local,
            basis.reshape(len(xyz), -1),
            scale,
            recent_speed,
            last_speed,
            last_acc,
        ],
        axis=1,
    )
    return seq.astype(np.float32), dense_small.astype(np.float32), basis, scale


def fit_standardize(x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x[idx].mean(axis=0, keepdims=True)
    std = x[idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


class ResidualSequenceNet(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        dense_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dense_proj = nn.Sequential(
            nn.Linear(dense_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, seq: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        h = self.seq_proj(seq)
        h = self.encoder(h)
        last = h[:, -1]
        mean = h.mean(dim=1)
        std = h.std(dim=1, unbiased=False)
        dense_h = self.dense_proj(dense)
        return self.head(torch.cat([last, mean, std, dense_h], dim=1))


def weighted_hit_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    boundary_loss: float,
) -> torch.Tensor:
    # Smooth residual fit in normalized local-frame coordinates.
    huber = F.smooth_l1_loss(pred, target, beta=0.20, reduction="none").sum(dim=1)

    # A differentiable proxy for R-Hit@1cm: focus extra pressure around the
    # threshold instead of spending capacity on very far outliers.
    dist_m = torch.linalg.norm((pred - target) * scale, dim=1)
    boundary = F.softplus((dist_m - 0.010) / 0.0025) * 0.0025
    return torch.mean(weight * (huber + boundary_loss * boundary))


@torch.no_grad()
def predict_scaled(
    model: nn.Module,
    seq: np.ndarray,
    dense: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    for start in range(0, len(seq), batch_size):
        end = min(len(seq), start + batch_size)
        s = torch.as_tensor(seq[start:end], device=device)
        d = torch.as_tensor(dense[start:end], device=device)
        outs.append(model(s, d).detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float64)


def local_scaled_to_abs(
    base: np.ndarray,
    pred_scaled_local: np.ndarray,
    basis: np.ndarray,
    scale: np.ndarray,
    shrink: float = 1.0,
) -> np.ndarray:
    corr_local = pred_scaled_local * scale
    corr_global = correction_from_prediction(corr_local, basis, scale, "local")
    return base + shrink * corr_global


def train_fold(
    cfg: TorchConfig,
    fold: int,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    seq: np.ndarray,
    dense_all: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    basis: np.ndarray,
    scale: np.ndarray,
    sample_weight: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray], ResidualSequenceNet]:
    dense_mean, dense_std = fit_standardize(dense_all, tr_idx)
    dense = standardize(dense_all, dense_mean, dense_std)

    model = ResidualSequenceNet(
        seq_dim=seq.shape[2],
        dense_dim=dense.shape[1],
        hidden=cfg.hidden,
        layers=cfg.layers,
        heads=cfg.heads,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.06)

    tr_seq = torch.as_tensor(seq[tr_idx], device=device)
    tr_dense = torch.as_tensor(dense[tr_idx], device=device)
    tr_target = torch.as_tensor(target[tr_idx], device=device)
    tr_scale = torch.as_tensor(scale[tr_idx].astype(np.float32), device=device)
    tr_weight = torch.as_tensor(sample_weight[tr_idx].astype(np.float32), device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_summary: dict[str, float] | None = None
    best_epoch = -1
    wait = 0
    n = len(tr_idx)
    rng = np.random.default_rng(cfg.seed + fold * 997)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            s = tr_seq[batch]
            if cfg.noise_std > 0:
                s = s + torch.randn_like(s) * cfg.noise_std
            pred = model(s, tr_dense[batch])
            loss = weighted_hit_loss(
                pred,
                tr_target[batch],
                tr_scale[batch],
                tr_weight[batch],
                cfg.boundary_loss,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        pred_scaled = predict_scaled(model, seq[va_idx], dense[va_idx], device, cfg.batch_size)
        pred_abs = local_scaled_to_abs(base[va_idx], pred_scaled, basis[va_idx], scale[va_idx])
        summary = score_summary(pred_abs, y[va_idx])
        if best_summary is None or summary["hit"] > best_summary["hit"] or (
            summary["hit"] == best_summary["hit"] and summary["median_distance"] < best_summary["median_distance"]
        ):
            best_summary = summary
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or wait == 0:
            print(
                f"[fold {fold}] epoch={epoch:03d} loss={np.mean(losses):.5f} "
                f"val_hit={summary['hit']:.5f} best={best_summary['hit']:.5f}@{best_epoch}",
                flush=True,
            )
        if wait >= cfg.patience:
            break

    assert best_state is not None and best_summary is not None
    model.load_state_dict(best_state)
    pred_scaled = predict_scaled(model, seq[va_idx], dense[va_idx], device, cfg.batch_size)
    pred_abs = local_scaled_to_abs(base[va_idx], pred_scaled, basis[va_idx], scale[va_idx])
    fold_info = {"fold": fold, "best_epoch": best_epoch, "summary": best_summary}
    stats = {"dense_mean": dense_mean, "dense_std": dense_std}
    return pred_abs, fold_info, stats, model


def blend_with_existing_submission(
    out_dir: Path,
    ids: np.ndarray,
    torch_pred: np.ndarray,
    blend_path: str,
) -> dict[str, Any]:
    path = Path(blend_path)
    if not path.exists():
        return {"available": False, "reason": f"missing {blend_path}"}
    df = pd.read_csv(path)
    ref = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    report: dict[str, Any] = {"available": True, "blend_path": str(path), "variants": {}}
    for w in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        pred = (1.0 - w) * ref + w * torch_pred
        name = f"submission_gpu_seq_x_multibase_w{int(w * 100):02d}.csv"
        write_submission(out_dir / name, ids, pred)
        d = np.linalg.norm(pred - ref, axis=1)
        report["variants"][name] = {
            "torch_weight": w,
            "delta_vs_reference_mean": float(d.mean()),
            "delta_vs_reference_p95": float(np.quantile(d, 0.95)),
            "delta_vs_reference_max": float(d.max()),
        }
    return report


def parse_args() -> TorchConfig:
    parser = argparse.ArgumentParser(description="GPU-friendly trajectory sequence residual model.")
    parser.add_argument("--zip-path", default=TorchConfig.zip_path)
    parser.add_argument("--out-dir", default=TorchConfig.out_dir)
    parser.add_argument("--residual-base", default=TorchConfig.residual_base)
    parser.add_argument("--folds", type=int, default=TorchConfig.folds)
    parser.add_argument("--seed", type=int, default=TorchConfig.seed)
    parser.add_argument("--epochs", type=int, default=TorchConfig.epochs)
    parser.add_argument("--patience", type=int, default=TorchConfig.patience)
    parser.add_argument("--batch-size", type=int, default=TorchConfig.batch_size)
    parser.add_argument("--hidden", type=int, default=TorchConfig.hidden)
    parser.add_argument("--layers", type=int, default=TorchConfig.layers)
    parser.add_argument("--heads", type=int, default=TorchConfig.heads)
    parser.add_argument("--dropout", type=float, default=TorchConfig.dropout)
    parser.add_argument("--lr", type=float, default=TorchConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TorchConfig.weight_decay)
    parser.add_argument("--boundary-loss", type=float, default=TorchConfig.boundary_loss)
    parser.add_argument("--noise-std", type=float, default=TorchConfig.noise_std)
    parser.add_argument("--device", default=TorchConfig.device)
    parser.add_argument("--blend-with", default=TorchConfig.blend_with)
    parser.add_argument("--no-submissions", action="store_true")
    return TorchConfig(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"[device] {device}", flush=True)
    if device.type == "cuda":
        print(f"[cuda] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_float32_matmul_precision("high")

    data = load_competition_data(cfg.zip_path)
    phys_train = physical_predictions(data.train_xyz)
    phys_test = physical_predictions(data.test_xyz)
    if cfg.residual_base not in phys_train:
        raise ValueError(f"unknown residual base: {cfg.residual_base}")
    base_train = phys_train[cfg.residual_base]
    base_test = phys_test[cfg.residual_base]

    seq_train, dense_small_train, basis_train, scale_train = build_sequence_inputs(data.train_xyz, base_train)
    seq_test, dense_small_test, basis_test, scale_test = build_sequence_inputs(data.test_xyz, base_test)
    # Add the proven hand-engineered feature vector as dense context.  It is
    # standardized fold-wise below to avoid validation leakage.
    dense_train = np.concatenate([dense_small_train, build_features(data.train_xyz).astype(np.float32)], axis=1)
    dense_test = np.concatenate([dense_small_test, build_features(data.test_xyz).astype(np.float32)], axis=1)

    residual_local = np.einsum("nc,nkc->nk", data.y - base_train, basis_train)
    target = (residual_local / scale_train).astype(np.float32)
    sample_weight = sample_weight_for("trap", base_train, data.y).astype(np.float32)
    print(
        f"[base] {cfg.residual_base} physical={score_summary(base_train, data.y)} "
        f"seq={seq_train.shape} dense={dense_train.shape}",
        flush=True,
    )

    kfold = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    oof = np.zeros_like(data.y)
    fold_infos = []
    fold_models: list[dict[str, Any]] = []
    test_scaled_preds = []
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(seq_train), start=1):
        pred_abs, info, stats, model = train_fold(
            cfg,
            fold,
            tr_idx,
            va_idx,
            seq_train,
            dense_train,
            target,
            base_train,
            data.y,
            basis_train,
            scale_train,
            sample_weight,
            device,
        )
        oof[va_idx] = pred_abs
        fold_infos.append(info)
        dense_test_fold = standardize(dense_test, stats["dense_mean"], stats["dense_std"])
        test_scaled = predict_scaled(model, seq_test, dense_test_fold, device, cfg.batch_size)
        test_scaled_preds.append(test_scaled)
        fold_models.append(
            {
                "fold": fold,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "stats": stats,
            }
        )

    raw = score_summary(oof, data.y)
    correction = oof - base_train
    best_shrink = find_best_shrink(base_train, correction, data.y)
    print(f"[oof] raw={raw} best_shrink={best_shrink}", flush=True)

    test_scaled_mean = np.mean(test_scaled_preds, axis=0)
    raw_test = local_scaled_to_abs(base_test, test_scaled_mean, basis_test, scale_test, shrink=1.0)
    shrink_test = local_scaled_to_abs(
        base_test,
        test_scaled_mean,
        basis_test,
        scale_test,
        shrink=best_shrink["shrink"],
    )

    report: dict[str, Any] = {
        "config": asdict(cfg),
        "device": str(device),
        "base_physical": score_summary(base_train, data.y),
        "folds": fold_infos,
        "raw": raw,
        "best_shrink": best_shrink,
    }
    np.savez_compressed(out_dir / "oof_cache.npz", y=data.y, gpu_seq=oof)

    if not cfg.no_submissions:
        write_submission(out_dir / "submission_gpu_seq_raw.csv", data.test_ids, raw_test)
        write_submission(out_dir / "submission_gpu_seq_shrink.csv", data.test_ids, shrink_test)
        report["test_blends"] = blend_with_existing_submission(out_dir, data.test_ids, shrink_test, cfg.blend_with)

    torch.save(
        {
            "config": asdict(cfg),
            "fold_models": fold_models,
            "seq_dim": seq_train.shape[2],
            "dense_dim": dense_train.shape[1],
            "best_shrink": best_shrink,
            "raw": raw,
        },
        out_dir / "gpu_seq_models.pt",
    )
    (out_dir / "gpu_seq_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_dir / 'gpu_seq_report.json'}", flush=True)


if __name__ == "__main__":
    main()
