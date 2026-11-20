"""The versions a run used, so results from different setups can be told apart.

The runner doesn't pin FDLLM, and a project that imports it may override FDLLM's
version, so every run records what it used: the runner, FDLLM, Pillow (which FDLLM
uses to resize images), Python, and the container image when there is one. Versions
come from the installed packages' metadata, not from the working directory's git
checkout, so local and cloud runs record the same things.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from importlib import metadata
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

IMAGE_ENV_VAR = "FAB_BENCH_IMAGE"
# The runner image installs the runner from its build context, which carries no git
# metadata, so the image records the commit it was built from here.
RUNNER_COMMIT_ENV_VAR = "FAB_BENCH_RUNNER_COMMIT"
# What can change a model's input or the IDs of its results; Python and the image
# are recorded but not compared.
COMPARED_PACKAGES = ("fdllm", "pillow", "runner")


def run_environment() -> dict[str, Any]:
    """Return the versions this process runs with."""

    runner = package_version("fab-benchmark-runner")
    if runner["commit"] is None and os.environ.get(RUNNER_COMMIT_ENV_VAR):
        runner["commit"] = os.environ[RUNNER_COMMIT_ENV_VAR]
    return {
        "fdllm": package_version("fabdata-llm"),
        "image": os.environ.get(IMAGE_ENV_VAR) or None,
        "pillow": package_version("pillow").get("version"),
        "python": platform.python_version(),
        "runner": runner,
    }


def package_version(name: str) -> dict[str, Any]:
    """Return an installed package's version and, if known, its git commit.

    A package installed from git records the commit in its metadata. One installed
    from a local folder in editable mode, as during development, gets the commit
    of that folder's checkout, marked ``dirty`` when it has uncommitted changes.
    """

    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"commit": None, "version": None}
    found: dict[str, Any] = {"commit": None, "version": distribution.version}
    direct_url = _direct_url(distribution)
    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, dict) and isinstance(vcs_info.get("commit_id"), str):
        found["commit"] = vcs_info["commit_id"]
    elif isinstance(direct_url.get("dir_info"), dict):
        found.update(_checkout_commit(direct_url.get("url")))
    return found


def version_differences(
    current: Mapping[str, Any], earlier: Iterable[Mapping[str, Any]]
) -> list[str]:
    """Describe how earlier runs' environments differ from ``current``.

    Only COMPARED_PACKAGES count. Each difference is described once.
    """

    differences: list[str] = []
    for environment in earlier:
        for key in COMPARED_PACKAGES:
            if key not in environment:
                continue
            if environment[key] != current.get(key):
                description = (
                    f"{key}: {_describe(environment[key])} in earlier results, "
                    f"{_describe(current.get(key))} now"
                )
                if description not in differences:
                    differences.append(description)
    return differences


def _direct_url(distribution: metadata.Distribution) -> dict[str, Any]:
    text = distribution.read_text("direct_url.json")
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _checkout_commit(url: Any) -> dict[str, Any]:
    if not isinstance(url, str) or not url.startswith("file://"):
        return {}
    # url2pathname turns /C:/x into C:\x on Windows, and unquotes everywhere.
    folder = url2pathname(urlparse(url).path)
    try:
        commit = subprocess.check_output(
            ["git", "-C", folder, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", folder, "status", "--porcelain", "--", "."],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit, "dirty": bool(status.strip())}


def _describe(value: Any) -> str:
    if isinstance(value, Mapping):
        version = value.get("version")
        commit = value.get("commit")
        text = str(version) if version else "unknown version"
        if commit:
            text += f" ({str(commit)[:7]}{', dirty' if value.get('dirty') else ''})"
        return text
    return "unknown" if value is None else str(value)
