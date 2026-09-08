"""Replicate the prefilter measurements on CredData and diff against SecretBench.

The two corpora are not interchangeable, and the split below follows from that.

Recall loss is a true-secrets-only quantity, so it transfers cleanly: CredData
yields a value for 99.2% of its true secrets and the comparison is fair.

Workload reduction is not. Most of CredData's false positives carry no value
bounds in the markup, so the entropy and value-shape families can only be scored
on the subset that does, where true secrets are heavily over-represented. Their
workload reduction there is a lower bound on the full corpus, and is reported as
such. The path and filename families have no such problem -- paths are known for
every row -- so they are scored on the whole corpus.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_prefilters import evaluate_all_filters
from src.features import compute_features
from src.load_creddata import dataset_audit, load_normalized, verify_paths
from src.load_secretbench import load_normalized as load_secretbench
from src.plot_creddata import plot_cliff_comparison, plot_creddata_heatmap
from src.prefilters import PAPER_FILTERS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _PROJECT_ROOT / "results"

ENTROPY_FILTERS = ["entropy_lt_2_5", "entropy_lt_3_0", "entropy_lt_3_3",
                   "entropy_lt_3_5", "entropy_lt_3_6", "entropy_lt_3_7",
                   "entropy_lt_3_8", "entropy_lt_4_0", "entropy_lt_4_2",
                   "entropy_lt_4_5", "entropy_lt_5_0"]
VALUE_FILTERS = ["value_placeholder", "value_template_variable",
                 "looks_hash_or_checksum", "looks_uuid",
                 "looks_public_key_or_pem_public"]
PATH_FILTERS = ["path_test", "path_docs", "path_examples", "path_samples",
                "path_fixtures", "path_mock", "path_vendor", "path_node_modules",
                "path_third_party", "path_generated", "path_build_artifact",
                "path_test_family", "path_docs_family", "path_vendor_family",
                "path_generated_family"]
FILENAME_FILTERS = ["filename_sample", "filename_example", "filename_template",
                    "filename_default", "filename_test", "filename_fixture",
                    "filename_readme", "filename_benign_family"]

# Credential classes whose values are chosen by a person rather than generated.
# Splitting on this is what shows the corpora differ in composition, not in the
# behaviour of the filters.
_HUMAN_CHOSEN = r"Password|URL Credentials|^Auth$"

# Filters worth a sensitivity check. CredData's true secrets are far more
# concentrated by repository than SecretBench's, and its markup rule that
# credentials in test directories are labelled True means the test-path filters
# partly measure an annotation policy. Both effects have to be shown, not
# assumed away.
_SENSITIVITY_FILTERS = ["entropy_lt_3_5", "entropy_lt_3_7", "entropy_lt_3_8",
                        "path_test", "path_test_family", "path_docs_family",
                        "filename_test", "looks_hash_or_checksum"]


# ── the comparison table ─────────────────────────────────────────────

def build_comparison(cred_full: pd.DataFrame, cred_valued: pd.DataFrame,
                     secretbench: pd.DataFrame) -> pd.DataFrame:
    """One row per filter, scored on whichever CredData population is valid for it."""
    rows = []
    for names, source, population in (
        (PATH_FILTERS + FILENAME_FILTERS, cred_full, "full"),
        (ENTROPY_FILTERS + VALUE_FILTERS, cred_valued, "value-bounded subset"),
    ):
        for name in names:
            cd = source[source.filter_name == name]
            sb = secretbench[secretbench.filter_name == name]
            if cd.empty or sb.empty:
                continue
            cd, sb = cd.iloc[0], sb.iloc[0]
            rows.append({
                "filter_name": name,
                "creddata_population": population,
                "creddata_retained_recall": cd.retained_recall,
                "secretbench_retained_recall": sb.retained_recall,
                "retained_recall_delta": cd.retained_recall - sb.retained_recall,
                "creddata_workload_reduction": cd.workload_reduction,
                "secretbench_workload_reduction": sb.workload_reduction,
                "creddata_true_secrets_filtered": int(cd.true_secrets_filtered),
                "secretbench_true_secrets_filtered": int(sb.true_secrets_filtered),
            })
    return pd.DataFrame(rows)


def _spearman(a, b) -> float:
    """Rank correlation, without pulling in scipy for one function."""
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def rank_agreement(comparison: pd.DataFrame) -> dict:
    """Do the two corpora order the filters by cost the same way?

    Ordering is the claim that should survive a change of corpus; the absolute
    levels demonstrably do not.
    """
    # Entropy thresholds are nested, so their recall loss is monotone in the
    # threshold in any corpus whatsoever and the rank correlation is 1.0 by
    # construction. Only the path and filename families carry information here.
    out = {}
    groups = {
        "path_and_filename": PATH_FILTERS + FILENAME_FILTERS,
        "entropy_degenerate": ENTROPY_FILTERS,
    }
    for label, names in groups.items():
        sub = comparison[comparison.filter_name.isin(names)]
        out[label] = {
            "n_filters": len(sub),
            "spearman_recall_loss": round(_spearman(
                1 - sub.creddata_retained_recall,
                1 - sub.secretbench_retained_recall), 3),
        }
    return out


# ── the cliff ────────────────────────────────────────────────────────

def _entropy_grid(lo: float = 2.0, hi: float = 5.0, step: float = 0.1) -> np.ndarray:
    """An exactly-representable threshold grid.

    np.arange on tenths drifts (3.7 becomes 3.7000000000000006), which turns the
    filters' `entropy < T` into `entropy <= T`. SecretBench ships entropy rounded
    to two decimals, so hundreds of true secrets sit exactly on 3.70 and 3.80 and
    that drift moves the measured drop by a full percentage point.
    """
    scale = round(1 / step)
    return np.arange(round(lo * scale), round(hi * scale) + 1) / scale


def locate_cliff(true_secret_entropy: pd.Series, lo: float = 2.0,
                 hi: float = 5.0, step: float = 0.1) -> dict:
    """Find the entropy band where retained recall falls fastest.

    The paper's central claim is that this band sits at 3.7-3.8. Its location is
    what a second corpus can confirm or refute; its depth is corpus-specific.
    """
    grid = _entropy_grid(lo, hi, step)
    recall = np.array([1.0 - (true_secret_entropy < x).mean() for x in grid])
    drops = -np.diff(recall)
    order = np.argsort(drops)[::-1]
    return {
        "steepest_band": (round(float(grid[order[0]]), 1),
                          round(float(grid[order[0] + 1]), 1)),
        "steepest_drop": round(float(drops[order[0]]), 4),
        "runners_up": [((round(float(grid[i]), 1), round(float(grid[i + 1]), 1)),
                        round(float(drops[i]), 4)) for i in order[1:3]],
    }


def cliff_stability(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """How often the steepest band survives resampling repositories.

    The margin between the steepest band and its runner-up is thin on CredData,
    so the point estimate alone would overstate how firmly the cliff is located.
    """
    rng = np.random.default_rng(seed)
    groups = [g.entropy.values for _, g in df[df.true_secret].groupby("repo_name")]
    grid = _entropy_grid()
    counts: dict[tuple[float, float], int] = {}
    for _ in range(n_boot):
        pick = rng.choice(len(groups), len(groups), replace=True)
        pooled = np.concatenate([groups[i] for i in pick])
        recall = np.array([1.0 - (pooled < x).mean() for x in grid])
        i = int(np.argmax(-np.diff(recall)))
        band = (round(float(grid[i]), 1), round(float(grid[i + 1]), 1))
        counts[band] = counts.get(band, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return {"n_clusters": len(groups),
            "modal_band": ranked[0][0],
            "modal_share": round(ranked[0][1] / n_boot, 3),
            "runners_up": [(b, round(c / n_boot, 3)) for b, c in ranked[1:3]]}


def concentration(df: pd.DataFrame, top: int = 5) -> dict:
    """How much of the true-secret mass sits in the largest repositories."""
    counts = df[df.true_secret].repo_name.value_counts()
    total = int(counts.sum())
    return {"n_repositories_with_true_secrets": int(len(counts)),
            "largest": counts.index[0],
            "largest_share": round(float(counts.iloc[0] / total), 4),
            f"top{top}_share": round(float(counts.head(top).sum() / total), 4)}


def run_sensitivity(cred: pd.DataFrame, filters: list[str] | None = None) -> pd.DataFrame:
    """Recompute the headline filters under populations that remove a suspicion.

    ``minus_largest`` and ``minus_top5`` test whether a result is one repository
    wearing a corpus as a disguise. ``minus_placeholders`` drops CredData's X
    label -- test values, examples and placeholders -- which its markup counts as
    non-secrets but which are exactly what several filters are built to catch.
    """
    filters = filters or _SENSITIVITY_FILTERS
    counts = cred[cred.true_secret].repo_name.value_counts()
    populations = {
        "all": cred,
        "minus_largest": cred[cred.repo_name != counts.index[0]],
        "minus_top5": cred[~cred.repo_name.isin(counts.head(5).index)],
        "minus_placeholders": cred[cred.ground_truth != "X"],
    }

    rows = []
    for label, population in populations.items():
        # Entropy and value-shape need a value; path and filename do not.
        valued = population[population.has_value]
        full_results, _ = evaluate_all_filters(population, filters)
        valued_results, _ = evaluate_all_filters(valued, filters)
        for name in filters:
            source = (valued_results if name in ENTROPY_FILTERS + VALUE_FILTERS
                      else full_results)
            row = source[source.filter_name == name]
            if row.empty:
                continue
            row = row.iloc[0]
            rows.append({"population": label, "filter_name": name,
                         "n_true_secrets": int(row.total_true_secrets),
                         "retained_recall": row.retained_recall,
                         "workload_reduction": row.workload_reduction})
    return pd.DataFrame(rows)


def bootstrap_recall(df: pd.DataFrame, threshold: float, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Retained recall with a 95% CI, resampling repositories rather than rows.

    Secrets cluster hard by repository, so row-level resampling would understate
    the interval badly.
    """
    rng = np.random.default_rng(seed)
    groups = [g.entropy.values for _, g in df[df.true_secret].groupby("repo_name")]
    point = 1.0 - (df[df.true_secret].entropy < threshold).mean()
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), len(groups), replace=True)
        pooled = np.concatenate([groups[i] for i in pick])
        draws.append(1.0 - (pooled < threshold).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ── entry point ──────────────────────────────────────────────────────

def run() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)

    print("Verifying the path inversion against CredData's own metadata...")
    check = verify_paths()
    print(f"  {check['matched']}/{check['checked']} FilePaths re-derived exactly "
          f"({check['rate']:.4%})")
    if check["rate"] < 1.0:
        print("  WARNING: inversion is not exact; path results are unsafe")

    print("Loading CredData...")
    cred = compute_features(load_normalized())
    # compute_features scores an absent value as entropy 0, which an entropy
    # filter would read as "certainly low" and discard. Unknown is not zero:
    # restoring NaN makes the filters keep those rows, so their workload
    # reduction on the full corpus is a lower bound rather than a fiction.
    cred.loc[~cred.has_value, "entropy"] = np.nan
    audit = dataset_audit(cred)
    audit["path_inversion"] = {k: check[k] for k in ("checked", "matched", "rate")}
    print(f"  {audit['total_candidates']} candidates, "
          f"{audit['true_secrets']} true secrets, "
          f"{audit['unique_repositories']} repositories")
    print(f"  values recovered for {audit['true_secret_value_coverage']:.1%} "
          f"of true secrets")

    print("Evaluating filters...")
    paper_filters = list(PAPER_FILTERS)
    full_results, _ = evaluate_all_filters(cred, paper_filters)
    valued_results, _ = evaluate_all_filters(cred[cred.has_value], paper_filters)

    secretbench = pd.read_csv(_RESULTS / "prefilter_results.csv")
    comparison = build_comparison(full_results, valued_results, secretbench)
    comparison.to_csv(_RESULTS / "corpus_comparison.csv", index=False)

    print("\nRank agreement (Spearman on recall loss):")
    agreement = rank_agreement(comparison)
    for label, stats in agreement.items():
        print(f"  {label:<20} n={stats['n_filters']:<3} "
              f"rho={stats['spearman_recall_loss']}")

    print("\nEntropy cliff location:")
    sb_frame = compute_features(load_secretbench())
    human = cred.category.str.contains(_HUMAN_CHOSEN, regex=True, na=False)
    cliffs = {
        "secretbench": locate_cliff(sb_frame[sb_frame.true_secret].entropy),
        "creddata": locate_cliff(cred[cred.true_secret & cred.has_value].entropy),
        "creddata_excluding_human_chosen": locate_cliff(
            cred[cred.true_secret & cred.has_value & ~human].entropy),
    }
    for label, cliff in cliffs.items():
        band = cliff["steepest_band"]
        runner = cliff["runners_up"][0]
        print(f"  {label:<34} {band[0]}-{band[1]}  "
              f"({cliff['steepest_drop']:.1%}); "
              f"next {runner[0][0]}-{runner[0][1]} ({runner[1]:.1%})")

    print("\nHow often that band survives resampling repositories:")
    valued = cred[cred.has_value]
    stability = {
        "secretbench": cliff_stability(sb_frame),
        "creddata": cliff_stability(valued),
        "creddata_excluding_human_chosen": cliff_stability(valued[~human.loc[valued.index]]),
    }
    for label, stat in stability.items():
        band = stat["modal_band"]
        print(f"  {label:<34} {band[0]}-{band[1]} in {stat['modal_share']:.1%} "
              f"of draws ({stat['n_clusters']} repositories)")

    conc = concentration(cred)
    print(f"\nTrue-secret concentration: largest repository {conc['largest']} "
          f"holds {conc['largest_share']:.1%}, top 5 hold {conc['top5_share']:.1%} "
          f"({conc['n_repositories_with_true_secrets']} repositories)")

    print("\nSensitivity (retained recall / workload reduction):")
    sens = run_sensitivity(cred)
    sens.to_csv(_RESULTS / "creddata_sensitivity.csv", index=False)
    pivot = sens.pivot(index="filter_name", columns="population",
                       values="retained_recall")
    sb_recall = secretbench.set_index("filter_name").retained_recall
    pivot["secretbench"] = sb_recall.reindex(pivot.index)
    print((pivot[["all", "minus_largest", "minus_top5", "minus_placeholders",
                  "secretbench"]] * 100).round(1).to_string())

    print("\nCredData retained recall, 95% CI (cluster bootstrap by repository):")
    intervals = {}
    for threshold in (3.5, 3.7, 3.8):
        point, lo, hi = bootstrap_recall(valued, threshold)
        intervals[str(threshold)] = {"point": point, "lo": lo, "hi": hi}
        print(f"  entropy<{threshold}: {point:.1%}  [{lo:.1%}, {hi:.1%}]")

    (_RESULTS / "corpus_comparison_summary.json").write_text(json.dumps(
        {"audit": audit, "concentration": conc, "rank_agreement": agreement,
         "cliff": cliffs, "cliff_stability": stability,
         "creddata_recall_ci": intervals}, indent=2, default=str))
    plot_creddata_heatmap(cred)
    plot_cliff_comparison(cred, sb_frame)
    print(f"\nWritten to {_RESULTS}")


if __name__ == "__main__":
    run()
