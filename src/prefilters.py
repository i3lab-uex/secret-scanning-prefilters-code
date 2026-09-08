"""The 39 paper filters and its one layered-filter example."""

from typing import Callable

import pandas as pd

FilterFn = Callable[[pd.Series], bool]


def _make_entropy_filter(threshold: float) -> FilterFn:
    def discard(row: pd.Series) -> bool:
        entropy = row.get("entropy")
        return not pd.isna(entropy) and float(entropy) < threshold
    return discard


entropy_lt_2_5 = _make_entropy_filter(2.5)
entropy_lt_3_0 = _make_entropy_filter(3.0)
entropy_lt_3_3 = _make_entropy_filter(3.3)
entropy_lt_3_5 = _make_entropy_filter(3.5)
entropy_lt_3_6 = _make_entropy_filter(3.6)
entropy_lt_3_7 = _make_entropy_filter(3.7)
entropy_lt_3_8 = _make_entropy_filter(3.8)
entropy_lt_4_0 = _make_entropy_filter(4.0)
entropy_lt_4_2 = _make_entropy_filter(4.2)
entropy_lt_4_5 = _make_entropy_filter(4.5)
entropy_lt_5_0 = _make_entropy_filter(5.0)


def _flag(column: str) -> FilterFn:
    return lambda row: bool(row.get(column, False))


path_test = _flag("is_test_path")
path_docs = _flag("is_docs_path")
path_examples = _flag("is_example_path")
path_samples = _flag("is_sample_path")
path_fixtures = _flag("is_fixture_path")
path_mock = _flag("is_mock_path")
path_vendor = _flag("is_vendor_path")
path_node_modules = _flag("is_node_modules_path")
path_third_party = _flag("is_third_party_path")
path_generated = _flag("is_generated_path")
path_build_artifact = _flag("is_build_artifact_path")
filename_sample = _flag("filename_contains_sample")
filename_example = _flag("filename_contains_example")
filename_template = _flag("filename_contains_template")
filename_default = _flag("filename_contains_default")
filename_test = _flag("filename_contains_test")
filename_fixture = _flag("filename_contains_fixture")
filename_readme = _flag("filename_contains_readme")
value_placeholder = _flag("placeholder_like_value")
value_template_variable = _flag("template_variable_like")
looks_hash_or_checksum = _flag("looks_hash_or_checksum")
looks_uuid = _flag("looks_uuid")


def looks_public_key_or_pem_public(row: pd.Series) -> bool:
    return bool(row.get("looks_pem", False)) and not bool(row.get("looks_private_key", False))


def _any(row: pd.Series, columns: tuple[str, ...]) -> bool:
    return any(bool(row.get(column, False)) for column in columns)


def path_test_family(row: pd.Series) -> bool:
    return _any(row, ("is_test_path", "is_fixture_path", "is_mock_path"))


def path_docs_family(row: pd.Series) -> bool:
    return _any(row, ("is_docs_path", "is_example_path", "is_sample_path"))


def path_vendor_family(row: pd.Series) -> bool:
    return _any(row, ("is_vendor_path", "is_third_party_path", "is_node_modules_path"))


def path_generated_family(row: pd.Series) -> bool:
    return _any(row, ("is_generated_path", "is_build_artifact_path"))


def filename_benign_family(row: pd.Series) -> bool:
    return _any(row, tuple(
        f"filename_contains_{name}"
        for name in ("sample", "example", "template", "default", "dummy", "mock", "fixture")
    ))


def layered_filters(row: pd.Series) -> bool:
    """All path, filename, placeholder filters, and entropy < 4.0."""
    return any((
        path_test_family(row), path_docs_family(row), path_vendor_family(row),
        path_generated_family(row), filename_benign_family(row), filename_test(row),
        filename_readme(row), value_placeholder(row), value_template_variable(row),
        entropy_lt_4_0(row),
    ))


PAPER_FILTERS: dict[str, FilterFn] = {
    "entropy_lt_2_5": entropy_lt_2_5,
    "entropy_lt_3_0": entropy_lt_3_0,
    "entropy_lt_3_3": entropy_lt_3_3,
    "entropy_lt_3_5": entropy_lt_3_5,
    "entropy_lt_3_6": entropy_lt_3_6,
    "entropy_lt_3_7": entropy_lt_3_7,
    "entropy_lt_3_8": entropy_lt_3_8,
    "entropy_lt_4_0": entropy_lt_4_0,
    "entropy_lt_4_2": entropy_lt_4_2,
    "entropy_lt_4_5": entropy_lt_4_5,
    "entropy_lt_5_0": entropy_lt_5_0,
    "path_test": path_test,
    "path_docs": path_docs,
    "path_examples": path_examples,
    "path_samples": path_samples,
    "path_fixtures": path_fixtures,
    "path_mock": path_mock,
    "path_vendor": path_vendor,
    "path_node_modules": path_node_modules,
    "path_third_party": path_third_party,
    "path_generated": path_generated,
    "path_build_artifact": path_build_artifact,
    "filename_sample": filename_sample,
    "filename_example": filename_example,
    "filename_template": filename_template,
    "filename_default": filename_default,
    "filename_test": filename_test,
    "filename_fixture": filename_fixture,
    "filename_readme": filename_readme,
    "value_placeholder": value_placeholder,
    "value_template_variable": value_template_variable,
    "looks_hash_or_checksum": looks_hash_or_checksum,
    "looks_uuid": looks_uuid,
    "looks_public_key_or_pem_public": looks_public_key_or_pem_public,
    "path_test_family": path_test_family,
    "path_docs_family": path_docs_family,
    "path_vendor_family": path_vendor_family,
    "path_generated_family": path_generated_family,
    "filename_benign_family": filename_benign_family,
}

ALL_FILTERS: dict[str, FilterFn] = {**PAPER_FILTERS, "layered_filters": layered_filters}
