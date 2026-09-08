"""Load unobfuscated CredData and recover its original paths and values.

Candidate values are returned in memory for feature computation. Callers must
not persist or print them.
"""

import binascii
import csv
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = Path(os.environ.get("CREDDATA_DIR", _PROJECT_ROOT / "creddata"))

_LABEL_TRUE = "T"


# ── recovering the original paths ────────────────────────────────────

def _repo_key(repo_id: str) -> str:
    """The meta-file name CredData derives from a snapshot key."""
    return f"{binascii.crc32(binascii.unhexlify(repo_id)):08x}"


def _file_scope(path_without_extension: str, words: list[str]) -> str:
    """Reimplementation of CredData's get_file_scope, used to verify inversion."""
    scope = "/"
    lowered = f"./{path_without_extension.lower()}"
    for word in words:
        if word in lowered:
            scope += word[1:] if word.startswith("/") else word
            if not scope.endswith("/"):
                scope += "/"
    return "/_/" if scope == "/" else scope


def _index_repo_files(repo_dir: Path, wanted: set[str]) -> dict[str, str]:
    """FileID -> path relative to the repository root, for the labelled files.

    Mirrors the exclusions in CredData's move_files: no symlinks, nothing under
    .git, and no .xml (raw text line numbering is unreliable there, so CredData
    never labels one).

    Eight hex digits is 32 bits, and the corpus holds enough files that unlabelled
    paths do collide. Those are harmless and ignored. A collision on a *labelled*
    id would silently attach a row to the wrong file, so that one is fatal.
    """
    index: dict[str, str] = {}
    for path in repo_dir.glob("**/*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = os.path.relpath(path, repo_dir).replace("\\", "/")
        if rel.startswith(".git/") or "/.git/" in rel or rel.endswith(".xml"):
            continue
        file_id = hashlib.sha256(rel.encode()).hexdigest()[:8]
        if file_id not in wanted:
            continue
        if file_id in index:
            raise RuntimeError(
                f"FileID collision on a labelled file in {repo_dir.name}: "
                f"{index[file_id]} / {rel}")
        index[file_id] = rel
    return index


# ── reading candidate values ─────────────────────────────────────────

def _read_lines(path: Path) -> list[str] | None:
    """Split a file the way CredData's own tooling does, so offsets line up."""
    try:
        text = path.read_bytes().decode(errors="replace")
    except OSError:
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _extract_value(row: dict, lines: list[str]) -> str:
    """The candidate value, following the bounds convention of the obfuscator.

    Returns "" when the markup records no value bounds, which is the case for
    most false positives.
    """
    try:
        start, end = int(row["LineStart"]), int(row["LineEnd"])
    except ValueError:
        return ""
    if start < 1 or end > len(lines):
        return ""

    raw_start, raw_end = row["ValueStart"].strip(), row["ValueEnd"].strip()
    a = int(float(raw_start)) if raw_start else -1
    b = int(float(raw_end)) if raw_end else -1

    if start == end:
        return lines[start - 1][a:b] if 0 <= a < b else ""

    # Confirmed by the CredData maintainers: a multiline span runs from
    # (LineStart, ValueStart) to (LineEnd, ValueEnd), so the first and last
    # lines are trimmed rather than taken whole. Rows carrying no bounds are
    # whole-line annotations and are taken as they stand.
    block = lines[start - 1:end]
    if a >= 0 and b >= 0:
        block[0] = block[0][a:]
        block[-1] = block[-1][:b]
    return "\n".join(block)


# ── loading ──────────────────────────────────────────────────────────

def _iter_rows(root: Path):
    """Yield (meta row, original relative path, repo url) for every labelled row."""
    snapshot = json.loads((root / "snapshot.json").read_text())
    words = json.loads((root / "word_in_path.json").read_text())

    for repo_id, url in snapshot.items():
        meta_path = root / "meta" / f"{_repo_key(repo_id)}.csv"
        repo_dir = root / "tmp" / repo_id
        if not meta_path.exists() or not repo_dir.is_dir():
            continue
        with open(meta_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        index = _index_repo_files(repo_dir, {r["FileID"] for r in rows})
        for row in rows:
            rel = index.get(row["FileID"])
            if rel is not None:
                yield row, rel, url, repo_dir, _repo_key(repo_id), words


def verify_paths(root: Path = _DEFAULT_ROOT) -> dict:
    """Re-derive each shipped FilePath from the recovered path.

    CredData's FilePath encodes a bucket computed from the original path plus the
    original extension, so agreement on every row is direct evidence that the
    hash inversion recovered the true path and not a collision.
    """
    checked = matched = 0
    mismatches = []
    for row, rel, _url, _dir, repo_key, words in _iter_rows(root):
        stem, ext = os.path.splitext(rel)
        expected = f"data/{repo_key}{_file_scope(stem, words)}{row['FileID']}{ext.lower()}"
        checked += 1
        if expected == row["FilePath"]:
            matched += 1
        elif len(mismatches) < 10:
            mismatches.append((row["FilePath"], expected))
    return {"checked": checked, "matched": matched,
            "rate": round(matched / checked, 6) if checked else 0.0,
            "examples": mismatches}


def load_normalized(root: Path = _DEFAULT_ROOT) -> pd.DataFrame:
    """Build a candidate table shaped like the normalised SecretBench frame.

    Rows without value bounds remain available to path and filename filters;
    ``has_value`` identifies the population valid for value-based filters.
    """
    records = []
    for row, rel, url, repo_dir, _key, _words in _iter_rows(root):
        lines = _read_lines(repo_dir / rel)
        if lines is None:
            continue
        value = _extract_value(row, lines)
        records.append({
            "secret_value": value,
            "file_path": rel,
            "file_type": os.path.splitext(rel)[1].lower().lstrip("."),
            "length": len(value),
            "entropy": pd.NA,
            "true_secret": row["GroundTruth"] == _LABEL_TRUE,
            "ground_truth": row["GroundTruth"],
            "category": row["Category"],
            "repo_name": url.removeprefix("https://github.com/"),
            "is_multiline_flag": row["LineStart"] != row["LineEnd"],
            "has_value": bool(value),
        })

    df = pd.DataFrame.from_records(records)
    # Left NA so features.compute_features fills it with Shannon entropy, the
    # same quantity SecretBench ships (rounded there to two decimals).
    df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    return df


def dataset_audit(df: pd.DataFrame) -> dict:
    n = len(df)
    with_value = df[df["has_value"]]
    return {
        "total_candidates": n,
        "true_secrets": int(df["true_secret"].sum()),
        "true_secret_prevalence": round(df["true_secret"].mean(), 4),
        "unique_repositories": int(df["repo_name"].nunique()),
        "ground_truth_counts": df["ground_truth"].value_counts().to_dict(),
        "candidates_with_value": len(with_value),
        "true_secrets_with_value": int(with_value["true_secret"].sum()),
        "true_secret_value_coverage": round(
            with_value["true_secret"].sum() / max(int(df["true_secret"].sum()), 1), 4),
    }
