"""Reproduce the SecretBench results and figures."""

import json
from pathlib import Path

from src.evaluate_prefilters import (
    compute_pareto_frontier,
    evaluate_all_filters,
    save_results,
    save_secret_type_analysis,
)
from src.features import compute_features
from src.load_secretbench import dataset_audit, load_normalized
from src.plot_pipeline import plot_pipeline
from src.plot_results import generate_all_figures
from src.robustness import run_all as run_robustness

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    print("Loading SecretBench")
    raw = load_normalized()
    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dataset_audit.json").write_text(
        json.dumps(dataset_audit(raw), indent=2, default=str)
    )

    print("Evaluating filters")
    findings = compute_features(raw)
    results, masks = evaluate_all_filters(findings)
    save_results(results, compute_pareto_frontier(results))
    save_secret_type_analysis(findings, masks)
    run_robustness(findings)

    print("Generating figures")
    generate_all_figures(findings, results)
    plot_pipeline()
    print(f"Wrote {results_dir} and {_ROOT / 'figures'}")


if __name__ == "__main__":
    main()
