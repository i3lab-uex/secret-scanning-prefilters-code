"""Robustness checks: bootstrap CIs, repo sensitivity, cross-tabulation."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.prefilters import ALL_FILTERS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RNG = np.random.default_rng(42)

_KEY_FILTERS = [
    "entropy_lt_3_0", "entropy_lt_3_3", "entropy_lt_3_5",
    "entropy_lt_3_6", "entropy_lt_3_7", "entropy_lt_3_8", "entropy_lt_4_0",
    "path_test_family", "path_docs_family", "path_vendor_family",
    "path_generated_family", "filename_benign_family",
]


def _precompute_masks(df: pd.DataFrame, filter_names: list[str]) -> dict[str, np.ndarray]:
    masks = {}
    for fname in filter_names:
        if fname in ALL_FILTERS:
            masks[fname] = df.apply(ALL_FILTERS[fname], axis=1).values
    return masks


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    filter_names: list[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if filter_names is None:
        filter_names = _KEY_FILTERS

    masks = _precompute_masks(df, filter_names)
    true_arr = df["true_secret"].values.astype(bool)
    repo_arr = df["repo_name"].values
    repos = np.unique(repo_arr)
    n_repos = len(repos)

    repo_indices = {r: np.where(repo_arr == r)[0] for r in repos}

    rows = []
    for fname in filter_names:
        if fname not in masks:
            continue
        mask = masks[fname]
        n_true = true_arr.sum()
        point_rl = (mask & true_arr).sum() / n_true if n_true else 0
        point_wr = mask.sum() / len(df)

        boot_rl = np.zeros(n_boot)
        boot_wr = np.zeros(n_boot)

        for b in range(n_boot):
            sampled = _RNG.choice(repos, size=n_repos, replace=True)
            idx = np.concatenate([repo_indices[r] for r in sampled])
            b_true = true_arr[idx]
            b_mask = mask[idx]
            b_n_true = b_true.sum()
            boot_rl[b] = (b_mask & b_true).sum() / b_n_true if b_n_true else 0
            boot_wr[b] = b_mask.sum() / len(idx)

        lo = alpha / 2 * 100
        hi = (1 - alpha / 2) * 100

        rows.append({
            "filter_name": fname,
            "recall_loss": round(point_rl, 6),
            "recall_loss_ci_lo": round(np.percentile(boot_rl, lo), 6),
            "recall_loss_ci_hi": round(np.percentile(boot_rl, hi), 6),
            "recall_loss_std": round(np.std(boot_rl), 6),
            "workload_reduction": round(point_wr, 6),
            "wr_ci_lo": round(np.percentile(boot_wr, lo), 6),
            "wr_ci_hi": round(np.percentile(boot_wr, hi), 6),
        })

    return pd.DataFrame(rows)


def leave_one_repo_out(
    df: pd.DataFrame,
    filter_names: list[str] | None = None,
    top_n_repos: int = 10,
) -> pd.DataFrame:
    if filter_names is None:
        filter_names = [
            "entropy_lt_3_7", "entropy_lt_3_8",
            "path_test_family", "path_vendor_family",
        ]

    masks = _precompute_masks(df, filter_names)
    true_arr = df["true_secret"].values.astype(bool)

    repo_ts_counts = df[df["true_secret"]].groupby("repo_name").size().sort_values(ascending=False)
    top_repos = repo_ts_counts.head(top_n_repos).index.tolist()

    rows = []
    for fname in filter_names:
        if fname not in masks:
            continue
        mask = masks[fname]
        n_true = true_arr.sum()
        full_rl = (mask & true_arr).sum() / n_true if n_true else 0

        for repo in top_repos:
            keep = df["repo_name"].values != repo
            sub_true = true_arr[keep]
            sub_mask = mask[keep]
            sub_n_true = sub_true.sum()
            rl = (sub_mask & sub_true).sum() / sub_n_true if sub_n_true else 0
            rows.append({
                "filter_name": fname,
                "excluded_repo": repo,
                "repo_true_secrets": int(repo_ts_counts[repo]),
                "recall_loss_full": round(full_rl, 6),
                "recall_loss_without_repo": round(rl, 6),
                "recall_loss_delta": round(rl - full_rl, 6),
            })

    return pd.DataFrame(rows)


def cross_tabulate(
    df: pd.DataFrame,
    filter_names: list[str] | None = None,
) -> pd.DataFrame:
    if filter_names is None:
        filter_names = ["entropy_lt_4_0", "path_test_family"]

    masks = _precompute_masks(df, filter_names)

    df = df.copy()
    df["path_context"] = "other"
    for pb in ["is_test_path", "is_docs_path", "is_vendor_path", "is_generated_path"]:
        if pb in df.columns:
            df.loc[df[pb], "path_context"] = pb.replace("is_", "").replace("_path", "")

    rows = []
    for fname in filter_names:
        if fname not in masks:
            continue
        mask_arr = masks[fname]
        for cat in df["category"].unique():
            for path_ctx in df["path_context"].unique():
                idx = (df["category"].values == cat) & (df["path_context"].values == path_ctx)
                sub_true = df["true_secret"].values[idx]
                n_true = sub_true.sum()
                if n_true == 0:
                    continue
                sub_mask = mask_arr[idx]
                ts_filtered = (sub_mask & sub_true).sum()
                rows.append({
                    "filter_name": fname,
                    "category": cat,
                    "path_context": path_ctx,
                    "true_secrets": int(n_true),
                    "true_secrets_filtered": int(ts_filtered),
                    "recall_loss": round(ts_filtered / n_true, 6),
                })

    return pd.DataFrame(rows)


def run_all(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Computing cluster-bootstrap CIs (1000 iterations)...")
    ci_df = cluster_bootstrap_ci(df)
    ci_df.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    print("  Bootstrap CIs saved")

    print("  Computing leave-one-repo-out sensitivity...")
    loro_df = leave_one_repo_out(df)
    loro_df.to_csv(output_dir / "leave_one_repo_out.csv", index=False)
    print("  LORO sensitivity saved")

    print("  Computing cross-tabulation (category x path)...")
    xtab_df = cross_tabulate(df)
    xtab_df.to_csv(output_dir / "cross_tabulation.csv", index=False)
    print("  Cross-tabulation saved")
