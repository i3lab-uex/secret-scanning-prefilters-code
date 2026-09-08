"""Load and normalize SecretBench."""

from pathlib import Path

import pandas as pd

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "secret-bench-data" / "secretbench.csv"


def load_normalized(path: Path = _DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    df["true_secret"] = df["label"].str.upper() == "Y"
    df["secret_value"] = df["secret"].map(
        lambda value: value[1:-1] if value.startswith("[") and value.endswith("]") else value
    )
    df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    return df


def dataset_audit(df: pd.DataFrame) -> dict:
    true = df[df["true_secret"]]
    return {
        "total_findings": len(df),
        "true_secrets": len(true),
        "false_positives": len(df) - len(true),
        "true_secret_prevalence": round(float(df["true_secret"].mean()), 4),
        "unique_repositories": int(df["repo_name"].nunique()),
        "unique_file_types": int(df["file_type"].nunique()),
        "unique_categories": int(df["category"].nunique()),
        "unique_secret_values": int(df["secret_value"].nunique()),
        "unique_true_secret_values": int(true["secret_value"].nunique()),
        "unique_true_repo_value_pairs": int(
            true[["repo_name", "secret_value"]].drop_duplicates().shape[0]
        ),
        "categories": df["category"].value_counts().to_dict(),
        "true_secrets_by_category": true["category"].value_counts().to_dict(),
    }
