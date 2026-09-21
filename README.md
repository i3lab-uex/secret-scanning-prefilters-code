# What Do Cheap Prefilters Cost?

Analysis code for *"What Do Cheap Prefilters Cost? Measuring Recall Loss in
Secret-Scanning Triage Pipelines."*

## Data

The datasets are not included.

- Obtain SecretBench from its authors and place `secretbench.csv` at
  `secret-bench-data/secretbench.csv`.
- Obtain unobfuscated CredData from the [CredData maintainers][creddata] and
  place it at `creddata/`, or set `CREDDATA_DIR`. The public artifact has hashed
  paths and rewritten true-secret values, so it cannot reproduce these results.

CredData must contain `meta/`, `snapshot.json`, `word_in_path.json`, and the
pinned repository checkouts under `tmp/<repo-id>/`. The paper uses CredData
commit `c09c0c52`. The run verifies every recovered path before evaluating it.

The advisory corpus is included as `corpus/ghsa_literals.csv`: one row per
literal, with its advisory, repository, pinned vulnerable commit, path,
filename, length, and entropy. It carries no secret values.

[creddata]: https://github.com/Samsung/CredData

## Run

```bash
uv sync
uv run python -m src.run_pipeline      # SecretBench
uv run python -m src.compare_corpora   # CredData comparison and figures
uv run python -m src.plot_ghsa_entropy # advisory entropy figure
```

Run the SecretBench pipeline first. CredData path and filename filters use the
full corpus; value-based filters use rows with value bounds, so their workload
reduction is a lower bound.

## Outputs

| Output | Contents |
|---|---|
| `results/dataset_audit.json` | SecretBench dataset summary |
| `results/prefilter_results.csv` | Per-filter metrics |
| `results/pareto_frontier.csv` | Best filter at each recall floor |
| `results/stratified_by_secret_type.csv` | Recall loss by secret category |
| `results/bootstrap_ci.csv` | Repository-bootstrap confidence intervals |
| `results/leave_one_repo_out.csv` | Repository sensitivity |
| `results/cross_tabulation.csv` | Category-by-path results |
| `results/corpus_comparison.csv` | SecretBench and CredData comparison |
| `results/creddata_sensitivity.csv` | CredData sensitivity checks |
| `results/corpus_comparison_summary.json` | CredData audit and summary statistics |
| `figures/*.pdf` | All paper figures |

No secret values are written to outputs. CredData values are held in memory only
while features are computed.

## License

MIT; see [LICENSE](LICENSE). Dataset licenses remain with their authors.
