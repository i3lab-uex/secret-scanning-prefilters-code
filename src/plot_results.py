"""Generate all figures for the prefilter recall-loss study."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Axes show percentages, matching the tables and the prose. The underlying
# values stay as fractions in [0, 1]; only the tick labels change.
_PCT = PercentFormatter(xmax=1, decimals=0)

sns.set_theme(style="whitegrid", font_scale=1.1)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _PROJECT_ROOT / "figures"

_COLORS = {
    "blue": "#4c72b0",
    "orange": "#dd8452",
    "green": "#55a868",
    "red": "#c44e52",
    "purple": "#8172b3",
    "brown": "#937860",
    "pink": "#da8bc3",
    "gray": "#8c8c8c",
}


def _savefig(fig: plt.Figure, name: str) -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ── dataset-level plots ──────────────────────────────────────────────

def plot_entropy_distribution(df: pd.DataFrame) -> None:
    bins = [0, 2.5, 3.0, 3.3, 3.5, 3.6, 3.7, 3.8, 4.0, 4.2, 4.5, 5.0, 7.0]
    labels = [
        "<2.5", "2.5–3.0", "3.0–3.3", "3.3–3.5", "3.5–3.6", "3.6–3.7",
        "3.7–3.8", "3.8–4.0", "4.0–4.2", "4.2–4.5", "4.5–5.0", "≥5.0",
    ]
    df = df.copy()
    df["ebin"] = pd.cut(df["entropy"], bins=bins, labels=labels, right=False)

    grouped = df.groupby("ebin", observed=False).agg(
        fp=("true_secret", lambda s: (~s).sum()),
        ts=("true_secret", "sum"),
    ).reset_index()
    grouped["total"] = grouped["fp"] + grouped["ts"]
    grouped["ts_rate"] = grouped["ts"] / grouped["total"] * 100
    grouped.loc[grouped["total"] == 0, "ts_rate"] = 0

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(grouped))
    width = 0.7

    ax1.bar(x, grouped["fp"], width, color=_COLORS["blue"], alpha=0.7, label="False positives")
    ax1.bar(x, grouped["ts"], width, bottom=grouped["fp"], color=_COLORS["orange"], alpha=0.85, label="True secrets")

    ax1.axvspan(
        x[labels.index("3.7–3.8")] - width / 2,
        x[labels.index("3.7–3.8")] + width / 2,
        alpha=0.12, color="red", zorder=0,
    )

    ax1.set_xlabel("Shannon entropy range")
    ax1.set_ylabel("Number of candidates")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, grouped["ts_rate"], "D-", color=_COLORS["red"], markersize=5,
             linewidth=1.5, label="True-secret rate (%)", zorder=5)
    ax2.set_ylabel("True-secret rate (%)", color=_COLORS["red"])
    ax2.set_ylim(0, 100)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=9)

    ax1.set_title("Entropy distribution: candidate volume and true-secret rate per range")
    _savefig(fig, "entropy_distribution_true_vs_false")


# ── family plots ─────────────────────────────────────────────────────

def _plot_family_scatter(
    results_df: pd.DataFrame,
    filter_names: list[str],
    labels: list[str],
    title: str,
    filename: str,
    annotate: bool = True,
    figsize: tuple[float, float] = (10, 6),
    ylim: tuple[float, float] | None = None,
    nudges: dict[str, tuple[float, float, str]] | None = None,
    thresholds: tuple[float, ...] = (0.999, 0.995, 0.99, 0.95),
    show_stats: bool = True,
    group_label: str | None = None,
    individual_label: str = "Individual filters",
) -> None:
    data = results_df[results_df["filter_name"].isin(filter_names)].copy()
    if data.empty:
        return
    data["label"] = data["filter_name"].map(dict(zip(filter_names, labels)))
    data["sort"] = data["filter_name"].map({n: i for i, n in enumerate(filter_names)})
    data = data.sort_values("sort")
    nudges = nudges or {}

    fig, ax = plt.subplots(figsize=figsize)

    is_group = data["filter_name"].str.contains("family")
    if group_label and is_group.any():
        ax.scatter(data.loc[~is_group, "workload_reduction"],
                   data.loc[~is_group, "retained_recall"],
                   s=100, c=_COLORS["blue"], zorder=3, edgecolors="white",
                   linewidths=0.5, label=individual_label)
        ax.scatter(data.loc[is_group, "workload_reduction"],
                   data.loc[is_group, "retained_recall"],
                   s=160, c=_COLORS["orange"], marker="D", zorder=4,
                   edgecolors="white", label=group_label)
        ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
    else:
        ax.scatter(
            data["workload_reduction"], data["retained_recall"],
            s=100, c=_COLORS["blue"], zorder=3, edgecolors="white", linewidths=0.5,
        )

    if annotate:
        for _, row in data.iterrows():
            wr = row["workload_reduction"]
            rr = row["retained_recall"]
            text = row["label"]
            if show_stats:
                text += f"\n({wr:.1%} WR, {int(row['true_secrets_filtered'])} lost)"
            wide = "family" in row["filter_name"]
            default = (0, -18, "center") if show_stats else (13 if wide else 7, -3, "left")
            dx, dy, ha = nudges.get(row["label"], default)
            ax.annotate(text, (wr, rr), fontsize=9 if not show_stats else 8,
                        ha=ha, xytext=(dx, dy), textcoords="offset points")

    if ylim:
        ax.set_ylim(*ylim)
    for threshold in thresholds:
        ax.axhline(y=threshold, color="red", linestyle=":", alpha=0.3)
        ax.text(ax.get_xlim()[1], threshold, f"{threshold:.1%}", fontsize=9.5,
                color="red", alpha=0.6, ha="left", va="center")

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Workload reduction", fontsize=12.5)
    ax.set_ylabel("Retained recall", fontsize=12.5)
    ax.set_title(title, fontsize=13.5)
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    if not ylim:
        ax.set_ylim(
            max(0, min(data["retained_recall"].min() - 0.05, 0.90)),
            1.005,
        )
    _savefig(fig, filename)


def plot_entropy_family(results_df: pd.DataFrame) -> None:
    entropy_filters = results_df[results_df["filter_name"].str.startswith("entropy_lt_")].copy()
    if entropy_filters.empty:
        return
    thresholds = entropy_filters["filter_name"].str.extract(r"entropy_lt_(\d+_\d+)")[0]
    thresholds = thresholds.str.replace("_", ".").astype(float)
    entropy_filters = entropy_filters.copy()
    entropy_filters["threshold"] = thresholds.values

    # Single y-axis: both series are fractions in [0, 1]. On independently
    # autoscaled twin axes the point where the curves cross is an artefact of
    # the scaling rather than a fact about the data.
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(entropy_filters["threshold"], entropy_filters["retained_recall"], "o-",
            color=_COLORS["orange"], linewidth=2, markersize=8, label="Retained recall")
    ax.plot(entropy_filters["threshold"], entropy_filters["workload_reduction"], "s--",
            color=_COLORS["blue"], linewidth=2, markersize=8, label="Workload reduction")

    ax.axvspan(3.7, 3.8, alpha=0.15, color="red", label="Cliff zone (3.7–3.8)")

    ax.set_xlabel("Entropy threshold (drop if < X)")
    ax.set_ylabel("Percentage of candidates / true secrets")
    ax.set_title("Entropy thresholds: recall vs workload reduction")
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(_PCT)
    ax.axhline(y=0.99, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.3)
    # Labels on the right, where both curves have dropped clear of the guides.
    ax.text(4.05, 0.998, "99% recall", fontsize=8, color="red", alpha=0.5)
    ax.text(4.05, 0.958, "95% recall", fontsize=8, color="red", alpha=0.4)

    for _, row in entropy_filters.iterrows():
        ts = int(row["true_secrets_filtered"])
        if ts > 0:
            ax.annotate(f"{ts}", (row["threshold"], row["retained_recall"]),
                        fontsize=7, ha="center", xytext=(0, -14), textcoords="offset points",
                        color=_COLORS["orange"], alpha=0.7)

    ax.legend(loc="center left")
    _savefig(fig, "family_entropy")


def plot_path_family(results_df: pd.DataFrame) -> None:
    filters = [
        "path_test", "path_docs", "path_examples", "path_samples",
        "path_fixtures", "path_mock", "path_vendor", "path_node_modules",
        "path_third_party", "path_generated", "path_build_artifact",
        "path_test_family", "path_docs_family",
        "path_vendor_family", "path_generated_family",
    ]
    labels = [
        "test", "docs", "examples", "samples",
        "fixtures", "mock", "vendor", "node_modules",
        "third_party", "generated", "build_artifact",
        "test_paths", "docs_paths",
        "vendor_paths", "generated_paths",
    ]
    data = results_df[results_df["filter_name"].isin(filters)].copy()
    if data.empty:
        return
    data["label"] = data["filter_name"].map(dict(zip(filters, labels)))
    is_family = data["filter_name"].str.contains("family")

    # Tall single axis rather than a broken one: the empty band between the
    # cluster and the test filters is itself informative, and a break would
    # compress exactly the distance the figure is meant to show.
    fig, ax = plt.subplots(figsize=(11, 9))

    ind = data[~is_family]
    fam = data[is_family]

    # (dx, dy, ha) nudges for the label pairs that still collide.
    nudges = {
        "mock": (0, 11, "center"),
        "samples": (7, -18, "left"),
        "third_party": (8, 1, "left"),
        "build_artifact": (6, -9, "left"),
        "node_modules": (-6, -10, "right"),
        "test_paths": (-13, -2, "right"),
    }

    ax.scatter(ind["workload_reduction"], ind["retained_recall"],
               s=80, c=_COLORS["blue"], zorder=3, edgecolors="white",
               linewidths=0.5, label="Individual path filters")
    ax.scatter(fam["workload_reduction"], fam["retained_recall"],
               s=140, c=_COLORS["orange"], marker="D", zorder=4,
               edgecolors="white", label="Path groups")
    for _, row in data.iterrows():
        # Group markers are larger (s=140), so their labels need a wider
        # offset or the diamond covers the first character.
        wide = "family" in row["filter_name"]
        dx, dy, ha = nudges.get(row["label"], (12 if wide else 6, -3, "left"))
        ax.annotate(
            row["label"], (row["workload_reduction"], row["retained_recall"]),
            fontsize=9, ha=ha,
            xytext=(dx, dy), textcoords="offset points",
        )
    for threshold in [0.99, 0.95]:
        ax.axhline(y=threshold, color="red", linestyle=":", alpha=0.3)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Workload reduction", fontsize=12.5)
    ax.set_ylabel("Retained recall", fontsize=12.5)
    ax.set_title("Path filters: workload reduction vs retained recall",
                 fontsize=13.5)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
    ax.set_ylim(0.75, 1.005)
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    _savefig(fig, "family_path")


def plot_filename_family(results_df: pd.DataFrame) -> None:
    filters = [
        "filename_sample", "filename_example", "filename_template",
        "filename_default", "filename_test", "filename_fixture",
        "filename_readme", "filename_benign_family",
    ]
    labels = [
        "sample", "example", "template", "default",
        "test", "fixture", "readme", "benign_filenames",
    ]
    # template/sample/fixture sit within 0.0011 of each other in recall, so
    # their labels are fanned out by hand; the per-filter numbers live in
    # Table 5 rather than in the point labels.
    _plot_family_scatter(
        results_df, filters, labels,
        "Filename filters: workload reduction vs retained recall",
        "family_filename",
        figsize=(10, 8), ylim=(0.865, 1.005), show_stats=False,
        thresholds=(0.995, 0.99, 0.95),
        group_label="Filename group", individual_label="Individual filename filters",
        nudges={
            "template": (0, 11, "center"),
            "sample": (13, 2, "left"),
            "fixture": (13, -11, "left"),
            "test": (-9, -3, "right"),
        },
    )


# ── combined frontier (summary) ──────────────────────────────────────

def plot_cost_recall_frontier(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # Categorize the simple filters (the four groups the paper discusses).
    # Compositions/cascades are excluded: the paper's frontier argument is
    # about simple filters only.
    entropy = results_df[results_df["filter_name"].str.startswith("entropy_lt_")]
    paths = results_df[results_df["filter_name"].str.match(r"^path_")]
    filenames = results_df[results_df["filter_name"].str.match(r"^filename_")]
    values = results_df[results_df["filter_name"].isin([
        "value_placeholder", "value_template_variable", "looks_hash_or_checksum",
        "looks_uuid", "looks_public_key_or_pem_public",
    ])]

    simple = pd.concat([entropy, paths, filenames, values])

    ax.scatter(entropy["workload_reduction"], entropy["retained_recall"],
               s=60, c=_COLORS["blue"], label="Entropy thresholds", zorder=3)
    ax.scatter(paths["workload_reduction"], paths["retained_recall"],
               s=60, c=_COLORS["green"], marker="^", label="Path filters", zorder=3)
    ax.scatter(filenames["workload_reduction"], filenames["retained_recall"],
               s=50, c=_COLORS["purple"], marker="s", label="Filename filters", zorder=3)
    ax.scatter(values["workload_reduction"], values["retained_recall"],
               s=60, c=_COLORS["red"], marker="D", label="Value-shape filters", zorder=3)

    # Only three guides: 0.999/0.995/0.99 sit within 0.009 of each other and
    # their labels collide. Labels go on the right, clear of the point cluster.
    for threshold in [0.99, 0.95, 0.90]:
        ax.axhline(y=threshold, color="red", linestyle=":", alpha=0.3)
        ax.text(1.01, threshold, f"recall={threshold:.0%}", fontsize=9.5,
                color="red", alpha=0.6, va="center", ha="left")

    # Non-dominated frontier over the simple filters
    sorted_desc = simple.sort_values("workload_reduction", ascending=False)
    pareto_points = []
    max_recall = -1
    for _, row in sorted_desc.iterrows():
        if row["retained_recall"] >= max_recall:
            max_recall = row["retained_recall"]
            pareto_points.append((row["workload_reduction"], row["retained_recall"]))
    pareto_points.sort()
    if pareto_points:
        px, py = zip(*pareto_points)
        ax.step(px, py, where="post", color="red", alpha=0.3, linewidth=2, label="Non-dominated frontier")

    ax.set_xlabel("Workload reduction (% of candidates filtered)", fontsize=12.5)
    ax.set_ylabel("Retained recall (% of true secrets kept)", fontsize=12.5)
    ax.set_title("Workload/recall frontier", fontsize=13.5)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.95, borderpad=0.8,
              labelspacing=0.6, handletextpad=0.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.24, 1.005)
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    _savefig(fig, "cost_recall_frontier")


# ── heatmaps ─────────────────────────────────────────────────────────

def _clean_filter_label(name: str) -> str:
    """Map internal filter names to reader-facing labels used in the paper."""
    if name.startswith("entropy_lt_"):
        return "entropy<" + name.replace("entropy_lt_", "").replace("_", ".")
    mapping = {
        "path_test_family": "test_paths",
        "path_docs_family": "docs_paths",
        "path_vendor_family": "vendor_paths",
        "path_generated_family": "generated_paths",
        "filename_benign_family": "benign_filenames",
    }
    return mapping.get(name, name)


def _plot_heatmap(strat_df: pd.DataFrame, title: str, filename: str, exclude_none: bool = False) -> None:
    if strat_df.empty:
        return
    pivot = strat_df.pivot_table(
        values="recall_loss", index="stratum", columns="filter_name", aggfunc="first"
    )
    if exclude_none:
        pivot = pivot[pivot.index != "none"]
    if pivot.empty:
        return

    pivot = pivot.rename(columns=_clean_filter_label)

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.8), max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={"label": "Recall loss (%)"})

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Secret category")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _savefig(fig, filename)



# ── main ─────────────────────────────────────────────────────────────

def generate_all_figures(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    results_dir = _PROJECT_ROOT / "results"

    print("  Plotting entropy distribution...")
    plot_entropy_distribution(df)

    print("  Plotting entropy thresholds...")
    plot_entropy_family(results_df)

    print("  Plotting path filters...")
    plot_path_family(results_df)

    print("  Plotting filename filters...")
    plot_filename_family(results_df)

    print("  Plotting workload/recall frontier...")
    plot_cost_recall_frontier(results_df)

    strat_type_path = results_dir / "stratified_by_secret_type.csv"
    if strat_type_path.exists():
        print("  Plotting recall loss heatmap by secret type...")
        strat = pd.read_csv(strat_type_path)
        _plot_heatmap(strat, "Recall loss by secret type and filter",
                      "recall_loss_heatmap_by_secret_type")
