"""Reading versions from package metadata, and comparing earlier runs' versions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import environment
from fab_benchmark_runner.environment import (
    package_version,
    run_environment,
    version_differences,
)


class FakeDistribution:
    def __init__(self, version: str, direct_url: dict[str, Any] | None) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


def _with_distribution(monkeypatch: pytest.MonkeyPatch, distribution) -> None:
    monkeypatch.setattr(
        environment.metadata, "distribution", lambda _name: distribution
    )


def test_a_git_install_records_its_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_distribution(
        monkeypatch,
        FakeDistribution(
            "0.30.1",
            {"url": "https://github.com/x/y.git", "vcs_info": {"commit_id": "abc123"}},
        ),
    )

    assert package_version("fabdata-llm") == {"commit": "abc123", "version": "0.30.1"}


def test_an_editable_install_records_its_checkouts_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-qm",
            "x",
        ],
        check=True,
    )
    _with_distribution(
        monkeypatch,
        FakeDistribution(
            "0.1.0", {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
        ),
    )

    clean = package_version("fab-benchmark-runner")
    (tmp_path / "f.txt").write_text("changed")
    dirty = package_version("fab-benchmark-runner")

    assert len(clean["commit"]) == 40
    assert (clean["dirty"], dirty["dirty"]) == (False, True)


def test_a_missing_package_has_no_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str):
        raise environment.metadata.PackageNotFoundError("x")

    monkeypatch.setattr(environment.metadata, "distribution", missing)

    assert package_version("pillow") == {"commit": None, "version": None}


def test_the_image_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAB_BENCH_IMAGE", "europe-docker.pkg.dev/p/r/fab-bench:1")

    assert run_environment()["image"] == "europe-docker.pkg.dev/p/r/fab-bench:1"


def test_differences_name_each_changed_package_once() -> None:
    current = {
        "fdllm": {"commit": "bbbbbbb1", "version": "0.30.1"},
        "pillow": "13.0.0",
        "python": "3.13.5",
        "runner": {"commit": "r", "version": "0.1.0"},
    }
    earlier = [
        {**current, "fdllm": {"commit": "aaaaaaa1", "version": "0.30.1"}},
        {**current, "fdllm": {"commit": "aaaaaaa1", "version": "0.30.1"}},
        {**current, "pillow": "12.3.0", "python": "3.12.8"},
    ]

    assert version_differences(current, earlier) == [
        "fdllm: 0.30.1 (aaaaaaa) in earlier results, 0.30.1 (bbbbbbb) now",
        "pillow: 12.3.0 in earlier results, 13.0.0 now",
    ]
