"""Compute the features used by the paper's filters."""

import math
import os
import re
from collections import Counter

import pandas as pd

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_PRIVATE_KEY_MARKERS = (
    "PRIVATE KEY", "RSA PRIVATE", "EC PRIVATE", "DSA PRIVATE",
    "OPENSSH PRIVATE", "PGP PRIVATE",
)
_PLACEHOLDER_PATTERNS = [
    re.compile(r"(?i)^(xxx+|yyy+|zzz+|aaa+|000+|111+|123+|test|fake|dummy|sample|example|placeholder|change.?me|replace.?me|your[_-]?(api[_-]?key|token|secret|password)|insert[_-]?(here|token|key)|todo|fixme|redacted|removed|hidden|n/?a|none|null|undefined|empty)$"),
    re.compile(r"^[x*#]{4,}$"),
    re.compile(r"(?i)^<[a-z_-]+(key|token|secret|password|api)[a-z_-]*>$"),
]
_TEMPLATE_PATTERNS = [
    re.compile(r"^\$\{[^}]+\}$"),
    re.compile(r"^\{\{[^}]+\}\}$"),
    re.compile(r"^%\([^)]+\)s$"),
    re.compile(r"^\$[A-Z_]{2,}$"),
    re.compile(r"^@[A-Za-z_.]+@$"),
    re.compile(r"^<[A-Z_]{2,}>$"),
    re.compile(r"^\{[a-z_]+\}$"),
]
_PATH_BUCKETS = {
    "is_test_path": re.compile(r"(?i)(^|/)tests?/|/spec/|/__tests__/|/test_|_test/"),
    "is_docs_path": re.compile(r"(?i)(^|/)docs?/|/documentation/"),
    "is_example_path": re.compile(r"(?i)(^|/)examples?/"),
    "is_sample_path": re.compile(r"(?i)(^|/)samples?/"),
    "is_fixture_path": re.compile(r"(?i)(^|/)fixtures?/|/testdata/|/test[_-]?data/"),
    "is_mock_path": re.compile(r"(?i)(^|/)mocks?/|/__mocks__/|/stubs?/"),
    "is_vendor_path": re.compile(r"(?i)(^|/)vendor/|/third[_-]?party/|/external/"),
    "is_node_modules_path": re.compile(r"(?i)(^|/)node_modules/"),
    "is_third_party_path": re.compile(r"(?i)(^|/)third[_-]?party/"),
    "is_generated_path": re.compile(
        r"(?i)(^|/)(generated|auto[_-]?generated|zz_generated|\.generated\.)/"
        r"|/dist/|\.min\.(js|css)$|/coverage/"
    ),
    "is_build_artifact_path": re.compile(
        r"(?i)(^|/)(build|target|out|bin|obj|cmake-build)/|\.class$"
    ),
}
_FILENAME_BUCKETS = {
    name: re.compile(rf"(?i){name}")
    for name in (
        "test", "sample", "example", "template", "default", "dummy", "mock",
        "fixture", "readme",
    )
}


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    values = out["secret_value"]
    out["basename"] = out["file_path"].map(lambda path: os.path.basename(path) if path else "")

    # SecretBench's supplied entropy is rounded; recompute it at full precision
    # so strict thresholds behave identically on both datasets.
    out["entropy"] = values.map(_shannon_entropy)
    out["looks_uuid"] = values.map(lambda value: bool(_UUID_RE.fullmatch(value)))
    out["looks_pem"] = values.map(lambda value: "-----BEGIN" in value or "-----END" in value)
    out["looks_private_key"] = values.map(
        lambda value: any(marker in value for marker in _PRIVATE_KEY_MARKERS)
    )
    out["looks_hash_or_checksum"] = values.map(
        lambda value: len(value) in {32, 40, 64} and bool(_HEX_RE.fullmatch(value))
    )
    out["placeholder_like_value"] = values.map(
        lambda value: any(pattern.match(value) for pattern in _PLACEHOLDER_PATTERNS)
    )
    out["template_variable_like"] = values.map(
        lambda value: any(pattern.match(value) for pattern in _TEMPLATE_PATTERNS)
    )

    for column, pattern in _PATH_BUCKETS.items():
        out[column] = out["file_path"].map(
            lambda path, regex=pattern: bool(regex.search(path)) if path else False
        )
    for name, pattern in _FILENAME_BUCKETS.items():
        out[f"filename_contains_{name}"] = out["basename"].map(
            lambda basename, regex=pattern: bool(regex.search(basename)) if basename else False
        )
    return out
