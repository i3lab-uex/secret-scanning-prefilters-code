"""Generate the two CredData figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from src.features import compute_features
from src.load_creddata import load_normalized
from src.prefilters import ALL_FILTERS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIGURES = _PROJECT_ROOT / "figures"

_COLUMNS = {
    "entropy_lt_3_0": "entropy<3.0",
    "entropy_lt_3_5": "entropy<3.5",
    "entropy_lt_4_0": "entropy<4.0",
    "entropy_lt_4_5": "entropy<4.5",
    "filename_benign_family": "benign_filenames",
    "path_docs_family": "docs_paths",
    "path_generated_family": "generated_paths",
    "path_test_family": "test_paths",
    "path_vendor_family": "vendor_paths",
}

_MIN_TRUE_SECRETS = 100

_HUMAN_CHOSEN = r"Password|URL Credentials|^Auth$"


def recall_loss_by_category(df: pd.DataFrame) -> pd.DataFrame:
    true_secrets = df[df.true_secret].copy()
    for name in _COLUMNS:
        true_secrets[name] = true_secrets.apply(ALL_FILTERS[name], axis=1)
    grouped = true_secrets.groupby("category").agg(
        n=("true_secret", "size"),
        **{name: (name, "mean") for name in _COLUMNS},
    )
    grouped = grouped[grouped.n >= _MIN_TRUE_SECRETS].sort_values("n", ascending=False)
    return grouped[list(_COLUMNS)].rename(columns=_COLUMNS)


def plot_creddata_heatmap(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = compute_features(load_normalized())
        # An absent value is unknown, not zero; see compare_corpora.
        df.loc[~df.has_value, "entropy"] = np.nan

    pivot = recall_loss_by_category(df)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.8),
                                    max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={"label": "Recall loss (%)"})

    ax.set_title("CredData: recall loss by secret type and filter")
    ax.set_xlabel("")
    ax.set_ylabel("Secret category")
    plt.xticks(rotation=45, ha="right", fontsize=8)

    _FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES / "creddata_recall_loss_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cliff_comparison(
    cred: pd.DataFrame | None = None, secretbench: pd.DataFrame | None = None,
) -> None:
    """Retained recall against entropy threshold, both datasets on one axis.

    The claim the figure has to carry is that the curves bend in the same place
    and sit at different heights, so both are drawn against the same axis rather
    than on twin scales, where the vertical offset would be an artefact.
    """
    if cred is None:
        cred = compute_features(load_normalized())
        cred.loc[~cred.has_value, "entropy"] = np.nan
    if secretbench is None:
        from src.load_secretbench import load_normalized as load_secretbench
        secretbench = compute_features(load_secretbench())

    human = cred.category.str.contains(_HUMAN_CHOSEN, regex=True, na=False)
    grid = np.arange(20, 51) / 10

    series = [
        (secretbench[secretbench.true_secret].entropy, "SecretBench",
         "#4c72b0", "-", 2.2),
        (cred[cred.true_secret].entropy, "CredData", "#dd8452", "-", 2.2),
        (cred[cred.true_secret & ~human].entropy,
         "CredData, excluding human-typed credentials", "#dd8452", "--", 1.6),
    ]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(3.7, 3.8, alpha=0.15, color="red", label="Cliff (3.7–3.8)")

    for entropy, label, colour, style, width in series:
        recall = [1.0 - (entropy < x).mean() for x in grid]
        ax.plot(grid, recall, style, color=colour, linewidth=width, label=label)

    ax.set_xlabel("Entropy threshold (drop if < X)")
    ax.set_ylabel("Retained recall")
    ax.set_title("Retained recall against entropy threshold, on both datasets")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(2.0, 5.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="lower left", fontsize=9)

    _FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES / "creddata_cliff_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
