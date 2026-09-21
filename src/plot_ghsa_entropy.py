"""Generate the advisory-backed corpus figure.

Drawn on the same axes and grid as the SecretBench entropy figure
(``plot_results.plot_entropy_family``) so the two can be read against each other.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

sns.set_theme(style="whitegrid", font_scale=1.1)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LITERALS = _PROJECT_ROOT / "corpus" / "ghsa_literals.csv"
_FIGURES = _PROJECT_ROOT / "figures"

_PCT = PercentFormatter(xmax=1, decimals=0)
_ORANGE = "#dd8452"

# The thresholds reported in Table 3, so the two entropy figures share an x grid.
_THRESHOLDS = [2.5, 3.0, 3.3, 3.5, 3.6, 3.7, 3.8, 4.0, 4.2, 4.5, 5.0]


def plot_ghsa_entropy(
    literals_path: Path = _LITERALS,
    output_dir: Path = _FIGURES,
) -> None:
    entropy = pd.read_csv(literals_path)["entropy"].to_numpy()
    total = len(entropy)
    if total != 50:
        raise ValueError(f"expected 50 literals, found {total}")

    retained = np.array([float(np.mean(entropy >= t)) for t in _THRESHOLDS])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(_THRESHOLDS, retained, "o-", color=_ORANGE, linewidth=2, markersize=8,
            label="Retained recall")

    ax.set_xlabel("Entropy threshold (drop if < X)")
    ax.set_ylabel("Percentage of disclosed secrets")
    ax.set_title("Advisory-backed secrets: retained recall by entropy threshold")
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(_PCT)

    for threshold, value in zip(_THRESHOLDS, retained):
        lost = total - round(value * total)
        ax.annotate(f"{lost}", (threshold, value), fontsize=7, ha="center",
                    xytext=(0, -14), textcoords="offset points",
                    color=_ORANGE, alpha=0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ghsa_entropy_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_ghsa_entropy()
