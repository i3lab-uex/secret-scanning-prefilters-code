"""Evaluate filters and save the paper's result tables."""

from pathlib import Path

import pandas as pd

from src.prefilters import ALL_FILTERS, PAPER_FILTERS

_RESULTS = Path(__file__).resolve().parent.parent / "results"


def evaluate_all_filters(
    df: pd.DataFrame, filter_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    names = filter_names or list(ALL_FILTERS)
    rows = []
    masks = {}
    for name in names:
        mask = df.apply(ALL_FILTERS[name], axis=1)
        masks[name] = mask
        true = df["true_secret"]
        n_filtered = int(mask.sum())
        n_true_filtered = int((mask & true).sum())
        n_true = int(true.sum())
        rows.append({
            "filter_name": name,
            "total_findings": len(df),
            "total_true_secrets": n_true,
            "candidates_filtered": n_filtered,
            "workload_reduction": round(n_filtered / len(df), 6) if len(df) else 0,
            "true_secrets_filtered": n_true_filtered,
            "recall_loss": round(n_true_filtered / n_true, 6) if n_true else 0,
            "retained_recall": round(1 - n_true_filtered / n_true, 6) if n_true else 0,
            "filter_precision": round(
                (n_filtered - n_true_filtered) / n_filtered, 6
            ) if n_filtered else 0,
        })
    return pd.DataFrame(rows), masks


def compute_pareto_frontier(results: pd.DataFrame) -> pd.DataFrame:
    simple = results[results["filter_name"].isin(PAPER_FILTERS)]
    rows = []
    for threshold in (0.999, 0.995, 0.990, 0.980, 0.950, 0.900):
        eligible = simple[simple["retained_recall"] >= threshold]
        if eligible.empty:
            continue
        best = eligible.loc[eligible["workload_reduction"].idxmax()]
        rows.append({
            "recall_threshold": threshold,
            "best_filter": best["filter_name"],
            "max_workload_reduction": best["workload_reduction"],
            "retained_recall": best["retained_recall"],
            "true_secrets_lost": best["true_secrets_filtered"],
        })
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, pareto: pd.DataFrame) -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(_RESULTS / "prefilter_results.csv", index=False)
    pareto.to_csv(_RESULTS / "pareto_frontier.csv", index=False)


def save_secret_type_analysis(
    df: pd.DataFrame, masks: dict[str, pd.Series],
) -> None:
    filters = [
        "entropy_lt_3_0", "entropy_lt_3_5", "entropy_lt_4_0", "entropy_lt_4_5",
        "path_test_family", "path_docs_family", "path_vendor_family",
        "path_generated_family", "filename_benign_family",
    ]
    rows = []
    for category, group in df.groupby("category"):
        n_true = int(group["true_secret"].sum())
        for name in filters:
            group_mask = masks[name].loc[group.index]
            n_true_filtered = int((group_mask & group["true_secret"]).sum())
            rows.append({
                "stratum": category,
                "filter_name": name,
                "total_in_stratum": len(group),
                "true_secrets_in_stratum": n_true,
                "filtered_in_stratum": int(group_mask.sum()),
                "true_secrets_filtered": n_true_filtered,
                "recall_loss": round(n_true_filtered / n_true, 6),
                "retained_recall": round(1 - n_true_filtered / n_true, 6),
            })
    _RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(_RESULTS / "stratified_by_secret_type.csv", index=False)
