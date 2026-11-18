"""Tests for the configuration models, loader, and hashing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from day0.config import (
    Day0Config,
    config_hash,
    load_config,
    reference_fingerprint,
    resolve_path,
)
from day0.errors import ConfigError

FAKE_SUITE_FILE = "fake_suites/echo_mixed.jsonl"


def _base_dict() -> dict[str, Any]:
    """Return a minimal, valid configuration as a plain dictionary.

    Returns:
        A dictionary that validates as :class:`Day0Config`.
    """
    return {
        "schema_version": 1,
        "run": {"name": "unit-test", "seed": 1},
        "model": {"id": "fake-model", "max_model_len": 4096},
        "hardware": {"gpu_hour_usd": 3.0},
        "backend": {"kind": "fake", "port": 8199},
        "reference": {"name": "ref"},
        "candidates": [{"name": "cand", "args": ["--noise", "0.1"]}],
        "quality": {
            "suites": [{"path": FAKE_SUITE_FILE, "max_tokens": 16}],
            "slices": [0, 1024, 4096],
        },
        "perf": {
            "concurrency_levels": [1, 4],
            "target_concurrency": 4,
            "input_len": {"dist": "uniform", "min": 16, "max": 128},
            "output_len": 16,
        },
        "slo": {"ttft_p99_ms": 100, "tpot_p99_ms": 10},
    }


def _write_yaml(directory: Path, data: dict[str, Any], name: str = "cfg.yaml") -> Path:
    """Write a configuration dictionary to a YAML file.

    Args:
        directory: Directory in which to create the file.
        data: Configuration content.
        name: File name.

    Returns:
        Path of the written file.
    """
    path = directory / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    """Create a directory with an empty suite file at the expected relative location.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        The directory that contains the YAML file.
    """
    (tmp_path / "fake_suites").mkdir()
    (tmp_path / "fake_suites" / "echo_mixed.jsonl").write_text("", encoding="utf-8")
    return tmp_path


def _load_from(directory: Path, data: dict[str, Any]) -> Day0Config:
    """Write ``data`` to a YAML file in ``directory`` and load it.

    Args:
        directory: Directory that holds the configuration and its suites.
        data: Configuration content.

    Returns:
        The loaded configuration.
    """
    return load_config(_write_yaml(directory, data))


def test_valid_minimal_config_loads(suite_dir: Path) -> None:
    """A minimal configuration with an existing suite file loads."""
    cfg = _load_from(suite_dir, _base_dict())
    assert cfg.run.name == "unit-test"
    assert cfg.candidates[0].args == ["--noise", "0.1"]
    assert cfg.base_dir == suite_dir.resolve()


def test_example_fake_config_loads(examples_dir: Path) -> None:
    """The fake-server demo in examples/ is valid and lists six candidates."""
    cfg = load_config(examples_dir / "fake.yaml")
    assert [server.name for server in cfg.candidates] == [
        "identical",
        "fast-ok",
        "fast-noisy",
        "longctx-bad",
        "slow",
        "spec-fake",
    ]


def test_example_llama_config_is_valid_once_suites_exist(
    examples_dir: Path, tmp_path: Path
) -> None:
    """The real-run example validates when its three suite files exist.

    The GSM8K suite is built from the official dataset, so this test uses placeholder files in a
    temporary copy. It checks only the configuration rules, not the data.

    Args:
        examples_dir: Directory with the example configurations.
        tmp_path: Pytest's per-test temporary directory.
    """
    yaml_text = (examples_dir / "llama31-8b-h100.yaml").read_text(encoding="utf-8")
    config_dir = tmp_path / "examples"
    config_dir.mkdir()
    (config_dir / "llama31-8b-h100.yaml").write_text(yaml_text, encoding="utf-8")
    suites = tmp_path / "suites"
    suites.mkdir()
    for name in ("gsm8k_200.jsonl", "longctx_synth.jsonl", "chat_open_100.jsonl"):
        (suites / name).write_text("", encoding="utf-8")
    cfg = load_config(config_dir / "llama31-8b-h100.yaml")
    assert cfg.perf.target_concurrency == 32
    assert [s.name for s in cfg.candidates if not s.enabled] == ["spec-eagle"]


def test_missing_suite_file_is_reported_with_its_path(tmp_path: Path) -> None:
    """A missing suite file raises a ConfigError that names the file.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    with pytest.raises(ConfigError, match="suite file not found") as info:
        _load_from(tmp_path, _base_dict())
    assert "echo_mixed.jsonl" in str(info.value)


def test_suite_paths_resolve_against_yaml_directory(suite_dir: Path, monkeypatch: Any) -> None:
    """Relative suite paths resolve against the YAML file, not the working directory.

    Args:
        suite_dir: Directory with the suite file.
        monkeypatch: Pytest fixture used to change the working directory.
    """
    monkeypatch.chdir(suite_dir.parent)
    cfg = _load_from(suite_dir, _base_dict())
    resolved = resolve_path(cfg, cfg.quality.suites[0].path)
    assert resolved == suite_dir.resolve() / FAKE_SUITE_FILE
    assert resolved.is_file()


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d["candidates"].append({"name": "ref"}), "must be unique"),
        (lambda d: d["reference"].update(enabled=False), "cannot be disabled"),
        (lambda d: d["perf"].update(target_concurrency=3), "target_concurrency"),
        (lambda d: d["quality"].update(slices=[0, 4096, 1024]), "strictly ascending"),
        (lambda d: d["quality"].update(slices=[16, 1024, 4096]), "must start at 0"),
        (lambda d: d["quality"].update(slices=[0, 1024, 2048]), "end at or above"),
        (lambda d: d["backend"].update(kind="external"), "has no base_url"),
        (lambda d: d["perf"]["input_len"].update(dist="fixed"), "requires a positive 'value'"),
        (
            lambda d: d["perf"]["input_len"].update(dist="uniform", min=200, max=100),
            "0 < min <= max",
        ),
        (lambda d: d["perf"].update(output_len=4000), "exceeds model.max_model_len"),
        (lambda d: d["backend"].update(kind="docker"), "backend"),
        (lambda d: d["run"].update(name="Bad Name"), "run.name"),
        (lambda d: d.update(unknown_key=1), "unknown_key"),
        (lambda d: d["quality"].update(top_k_logprobs=21), "top_k_logprobs"),
    ],
)
def test_validation_rejects_bad_configs(suite_dir: Path, mutate: Any, expected: str) -> None:
    """Each validation rule rejects a violating configuration with a helpful message.

    Args:
        suite_dir: Directory with the suite file.
        mutate: Function that breaks one rule in the base dictionary.
        expected: Substring that the error message must contain.
    """
    data = _base_dict()
    mutate(data)
    with pytest.raises(ConfigError) as info:
        _load_from(suite_dir, data)
    assert expected in str(info.value)


def test_external_backend_accepts_per_server_base_url(suite_dir: Path) -> None:
    """An external backend without a backend-level URL is valid when every server has one.

    Args:
        suite_dir: Directory with the suite file.
    """
    data = _base_dict()
    data["backend"]["kind"] = "external"
    data["reference"]["base_url"] = "http://127.0.0.1:9000"
    data["candidates"][0]["base_url"] = "http://127.0.0.1:9001"
    cfg = _load_from(suite_dir, data)
    assert cfg.backend.kind == "external"


def test_config_hash_ignores_key_order(suite_dir: Path) -> None:
    """Reordering keys in the YAML file does not change the hash.

    Args:
        suite_dir: Directory with the suite file.
    """
    data = _base_dict()
    reordered = dict(reversed(list(copy.deepcopy(data).items())))
    first = _load_from(suite_dir, data)
    second = _load_from(suite_dir, reordered)
    assert config_hash(first) == config_hash(second)


def test_config_hash_changes_with_content(suite_dir: Path) -> None:
    """Changing one value changes the hash.

    Args:
        suite_dir: Directory with the suite file.
    """
    data = _base_dict()
    changed = copy.deepcopy(data)
    changed["candidates"][0]["args"] = ["--noise", "0.2"]
    assert config_hash(_load_from(suite_dir, data)) != config_hash(_load_from(suite_dir, changed))


def test_config_hash_ignores_directory_of_yaml(suite_dir: Path, tmp_path: Path) -> None:
    """The same configuration in another directory has the same hash.

    Args:
        suite_dir: Directory with the suite file.
        tmp_path: Pytest's per-test temporary directory.
    """
    other = tmp_path / "other"
    (other / "fake_suites").mkdir(parents=True)
    (other / "fake_suites" / "echo_mixed.jsonl").write_text("", encoding="utf-8")
    data = _base_dict()
    assert config_hash(_load_from(suite_dir, data)) == config_hash(_load_from(other, data))


def test_reference_fingerprint_depends_on_suites_hash_and_reference_args(suite_dir: Path) -> None:
    """The reference fingerprint changes with the suite contents and the reference arguments.

    Args:
        suite_dir: Directory with the suite file.
    """
    cfg = _load_from(suite_dir, _base_dict())
    base = reference_fingerprint(cfg, "aaa")
    assert base == reference_fingerprint(cfg, "aaa")
    assert base != reference_fingerprint(cfg, "bbb")
    changed = _base_dict()
    changed["reference"]["args"] = ["--dtype", "float16"]
    other = _load_from(suite_dir, changed)
    assert base != reference_fingerprint(other, "aaa")


def test_reference_fingerprint_ignores_candidates(suite_dir: Path) -> None:
    """Adding or changing candidates does not invalidate stored reference artifacts.

    Args:
        suite_dir: Directory with the suite file.
    """
    data = _base_dict()
    changed = copy.deepcopy(data)
    changed["candidates"].append({"name": "extra", "args": ["--noise", "0.5"]})
    assert reference_fingerprint(_load_from(suite_dir, data), "x") == reference_fingerprint(
        _load_from(suite_dir, changed), "x"
    )
