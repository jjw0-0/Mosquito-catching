from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .candidate_bank import build_candidate_bank
from .data import load_competition_data
from .features import build_features
from .metrics import score_summary
from .train import trajectory_basis_and_scale


def _summary_from_dist(dist: np.ndarray) -> dict[str, float]:
    return {
        "hit": float(np.mean(dist <= 0.01)),
        "mean_distance": float(np.mean(dist)),
        "median_distance": float(np.median(dist)),
        "p90_distance": float(np.quantile(dist, 0.90)),
        "p95_distance": float(np.quantile(dist, 0.95)),
        "p99_distance": float(np.quantile(dist, 0.99)),
    }


def _oracle_summary(stack: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    dist = np.linalg.norm(stack - y[:, None, :], axis=2)
    best = dist.min(axis=1)
    return {
        "oracle": _summary_from_dist(best),
        "topk_oracle_by_truth_distance": {
            str(k): float(np.mean(np.partition(dist, kth=min(k - 1, dist.shape[1] - 1), axis=1)[:, k - 1] <= 0.01))
            for k in (1, 2, 3, 5, 10, 20, 50)
            if k <= dist.shape[1]
        },
        "hit_candidate_count": {
            "mean": float((dist <= 0.01).sum(axis=1).mean()),
            "median": float(np.median((dist <= 0.01).sum(axis=1))),
            "p10": float(np.quantile((dist <= 0.01).sum(axis=1), 0.10)),
            "p90": float(np.quantile((dist <= 0.01).sum(axis=1), 0.90)),
        },
    }


def _load_known_oof(base_dir: Path, y: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    mb_path = base_dir / "outputs/oof_multibase_cache.npz"
    if mb_path.exists():
        mb = np.load(mb_path)
        for key in mb.files:
            if key != "y" and mb[key].shape == y.shape:
                out[f"ml_{key}"] = mb[key]
    gpu_path = base_dir / "outputs_gpu_seq_c0p58/oof_cache.npz"
    if gpu_path.exists() and "ml_multibase_report_opt" in out:
        gpu = np.load(gpu_path)["gpu_seq"]
        out["ml_gpu_seq_c0p58"] = gpu
        out["ml_old_mb_gpu_w0.45"] = 0.55 * out["ml_multibase_report_opt"] + 0.45 * gpu
        out["ml_old_mb_gpu_w0.60"] = 0.40 * out["ml_multibase_report_opt"] + 0.60 * gpu
    rep_path = base_dir / "outputs_repeated_multibase_8s/oof_repeated_multibase.npz"
    rep_report_path = base_dir / "outputs_repeated_multibase_8s/repeated_multibase_report.json"
    if rep_path.exists() and rep_report_path.exists():
        rep = np.load(rep_path)
        report = json.loads(rep_report_path.read_text(encoding="utf-8"))
        for key in rep.files:
            if key != "y" and rep[key].shape == y.shape:
                out[f"rep8_{key}"] = rep[key]
        names = report["blend"]["names"]
        weights = np.asarray(report["blend"]["weights"], dtype=np.float64)
        out["rep8_blend"] = sum(weights[i] * rep[names[i]] for i in range(len(names)))
    for rel, key in [
        ("outputs_transformer_jerk_flip_c0p58_v2/oof_cache.npz", "torch_transformer_jerk_flip_v2"),
        ("outputs_bigru_jerk_flip_c0p58_v2/oof_cache.npz", "torch_bigru_jerk_flip_v2"),
    ]:
        path = base_dir / rel
        if path.exists():
            z = np.load(path)
            if "gpu_seq" in z.files and z["gpu_seq"].shape == y.shape:
                out[key] = z["gpu_seq"]
    return out


def _choose_ref(candidates: dict[str, np.ndarray], y: np.ndarray) -> tuple[str, np.ndarray]:
    best_name = max(candidates, key=lambda name: score_summary(candidates[name], y)["hit"])
    return best_name, candidates[best_name]


def _local_vectors_to_global(local: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("kd,nad->nka", local, basis)


def _unit_directions() -> np.ndarray:
    dirs = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                v = np.array([x, y, z], dtype=np.float64)
                norm = np.linalg.norm(v)
                if norm > 0:
                    dirs.append(v / norm)
    return np.asarray(dirs, dtype=np.float64)


def ref_perturbation_stack(ref: np.ndarray, basis: np.ndarray, radii: list[float]) -> tuple[list[str], np.ndarray]:
    dirs = _unit_directions()
    names = []
    parts = []
    for radius in radii:
        offsets = _local_vectors_to_global(dirs * radius, basis)
        parts.append(ref[:, None, :] + offsets)
        for i in range(len(dirs)):
            names.append(f"ref_perturb_r{radius:g}_d{i:02d}")
    return names, np.concatenate(parts, axis=1)


def interpolation_stack(ref: np.ndarray, stack: np.ndarray, names: list[str], alphas: list[float]) -> tuple[list[str], np.ndarray]:
    keep = [i for i, name in enumerate(names) if name != "ref"]
    chunks = []
    out_names = []
    for alpha in alphas:
        chunks.append(ref[:, None, :] + alpha * (stack[:, keep, :] - ref[:, None, :]))
        out_names.extend([f"interp_a{alpha:g}_{names[i]}" for i in keep])
    return out_names, np.concatenate(chunks, axis=1)


def local_knn_feature(xyz: np.ndarray, variant: str) -> np.ndarray:
    basis, scale = trajectory_basis_and_scale(xyz)
    last = xyz[:, -1]
    d1 = np.diff(xyz, axis=1)
    d2 = np.diff(d1, axis=1)
    centered = np.einsum("ntc,nkc->ntk", xyz - last[:, None, :], basis) / scale[:, None, :]
    vel = np.einsum("ntc,nkc->ntk", d1, basis) / scale[:, None, :]
    acc = np.einsum("ntc,nkc->ntk", d2, basis) / scale[:, None, :]
    if variant == "pos":
        feat = centered.reshape(len(xyz), -1)
    elif variant == "pos_vel":
        feat = np.concatenate([centered.reshape(len(xyz), -1), vel.reshape(len(xyz), -1)], axis=1)
    elif variant == "pos_vel_acc":
        feat = np.concatenate([centered.reshape(len(xyz), -1), vel.reshape(len(xyz), -1), acc.reshape(len(xyz), -1)], axis=1)
    elif variant == "handcrafted":
        feat = build_features(xyz)
    else:
        raise ValueError(variant)
    return StandardScaler().fit_transform(np.clip(feat, -20.0, 20.0)).astype(np.float32)


def knn_template_audit(xyz: np.ndarray, y: np.ndarray, k_values: list[int]) -> dict[str, Any]:
    basis, scale = trajectory_basis_and_scale(xyz)
    last = xyz[:, -1]
    target_local = np.einsum("nc,nkc->nk", y - last, basis) / scale
    max_k = max(k_values)
    report: dict[str, Any] = {}
    for variant in ("pos", "pos_vel", "pos_vel_acc", "handcrafted"):
        feat = local_knn_feature(xyz, variant)
        nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean", algorithm="auto", n_jobs=-1)
        nn.fit(feat)
        neigh_dist, neigh_idx = nn.kneighbors(feat, return_distance=True)
        neigh_dist = neigh_dist[:, 1:]
        neigh_idx = neigh_idx[:, 1:]
        local_candidates = target_local[neigh_idx] * scale[:, None, :]
        pred = last[:, None, :] + np.einsum("nkd,nad->nka", local_candidates, basis)
        dist = np.linalg.norm(pred - y[:, None, :], axis=2)
        row: dict[str, Any] = {
            "top1": _summary_from_dist(dist[:, 0]),
            "topk_oracle": {str(k): _summary_from_dist(dist[:, :k].min(axis=1)) for k in k_values},
            "neighbor_feature_distance": {
                "p01": float(np.quantile(neigh_dist[:, 0], 0.01)),
                "p05": float(np.quantile(neigh_dist[:, 0], 0.05)),
                "p10": float(np.quantile(neigh_dist[:, 0], 0.10)),
                "median": float(np.median(neigh_dist[:, 0])),
            },
        }
        bins = []
        quantiles = np.quantile(neigh_dist[:, 0], [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            mask = (neigh_dist[:, 0] >= lo) & (neigh_dist[:, 0] <= hi)
            if mask.any():
                bins.append(
                    {
                        "lo": float(lo),
                        "hi": float(hi),
                        "n": int(mask.sum()),
                        "top1_hit": float(np.mean(dist[mask, 0] <= 0.01)),
                        "top10_oracle": float(np.mean(dist[mask, : min(10, max_k)].min(axis=1) <= 0.01)),
                        "target_distance_median": float(np.median(dist[mask, 0])),
                    }
                )
        row["nearest_distance_bins"] = bins
        report[variant] = row
        print(f"[knn] {variant} top1={row['top1']['hit']:.4f} top10={row['topk_oracle'].get('10', row['topk_oracle'][str(max_k)])['hit']:.4f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 0.8+ ceiling: information, oracle, and candidate headroom.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip")
    parser.add_argument("--out-dir", default="outputs_ceiling_audit")
    parser.add_argument("--profile", choices=["quick", "strong", "full"], default="strong")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument("--skip-interp", action="store_true")
    args = parser.parse_args()

    base_dir = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_competition_data(args.zip_path)

    print("[bank] building deterministic candidates", flush=True)
    bank = build_candidate_bank(data.train_xyz, profile=args.profile)
    known = _load_known_oof(base_dir, data.y)
    bank.update(known)
    ref_name, ref = _choose_ref(bank, data.y)
    basis, _ = trajectory_basis_and_scale(data.train_xyz)
    names = ["ref"] + [name for name in bank if name != ref_name]
    base_stack = np.stack([ref] + [bank[name] for name in names[1:]], axis=1)
    report: dict[str, Any] = {
        "ref_name": ref_name,
        "num_base_candidates": len(names),
        "reference": score_summary(ref, data.y),
        "base_oracle": _oracle_summary(base_stack, data.y),
    }
    print(f"[ref] {ref_name} {report['reference']}", flush=True)
    print(f"[base oracle] {report['base_oracle']['oracle']}", flush=True)

    p_names, p_stack = ref_perturbation_stack(ref, basis, radii=[0.0015, 0.003, 0.0045, 0.006, 0.008, 0.010, 0.012])
    perturb_stack = np.concatenate([base_stack, p_stack], axis=1)
    report["perturbation"] = {
        "num_candidates_added": len(p_names),
        "oracle": _oracle_summary(perturb_stack, data.y),
    }
    print(f"[perturb oracle] {report['perturbation']['oracle']['oracle']}", flush=True)

    if not args.skip_interp:
        i_names, i_stack = interpolation_stack(ref, base_stack, names, alphas=[0.25, 0.50, 0.75, 1.25])
        interp_stack = np.concatenate([perturb_stack, i_stack], axis=1)
        report["perturbation_plus_interpolation"] = {
            "num_candidates_added": len(p_names) + len(i_names),
            "oracle": _oracle_summary(interp_stack, data.y),
        }
        print(f"[perturb+interp oracle] {report['perturbation_plus_interpolation']['oracle']['oracle']}", flush=True)

    if not args.skip_knn:
        knn = knn_template_audit(data.train_xyz, data.y, k_values=[1, 3, 5, 10, 20, 50])
        report["knn_template"] = knn
        # KNN candidates are deliberately excluded from the main huge stack; this
        # isolates whether repeated trajectories carry future-mode information.

    (out_dir / "ceiling_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_dir / 'ceiling_audit_report.json'}", flush=True)


if __name__ == "__main__":
    main()
