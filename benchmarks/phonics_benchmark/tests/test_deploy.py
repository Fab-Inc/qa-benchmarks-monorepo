"""Tests for the Cloud Run deployment helper scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Deployment helpers require a POSIX shell",
)
@pytest.mark.parametrize(
    ("remote_items", "expected_exit_code", "expected_message", "executes_job"),
    [
        (b'{"id":"item-1"}\n', 0, "Reusing the identical", True),
        (b'{"id":"stale"}\n', 1, "Remote items differ", False),
    ],
)
def test_launch_run_reuses_only_byte_identical_items(
    tmp_path: Path,
    remote_items: bytes,
    expected_exit_code: int,
    expected_message: str,
    executes_job: bool,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud_log = tmp_path / "gcloud.log"
    remote_items_path = tmp_path / "remote-items.jsonl"
    remote_items_path.write_bytes(remote_items)
    local_items_path = tmp_path / "local-items.jsonl"
    local_items_path.write_bytes(b'{"id":"item-1"}\n')
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")

    _write_executable(
        bin_dir / "jq",
        """#!/usr/bin/env bash
set -euo pipefail
input="${@: -1}"
sed -n '1,$p' "${input}"
""",
    )
    _write_executable(
        bin_dir / "gcloud",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"${GCLOUD_LOG}"
if [[ "$1 $2" == "storage ls" ]]; then
  if [[ "$3" == */config.json ]]; then
    exit 1
  fi
  exit 0
fi
if [[ "$1 $2" == "storage cp" && "$3" == gs://* ]]; then
  cp "${FAKE_REMOTE_ITEMS}" "$4"
fi
""",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ID": "fab-qaf-decodability",
        "RESULTS_BUCKET": "gs://phonics-benchmark",
        "GCLOUD_LOG": str(gcloud_log),
        "FAKE_REMOTE_ITEMS": str(remote_items_path),
    }

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "deploy" / "launch-run.sh"),
            str(config_path),
            str(local_items_path),
            "run-a",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_exit_code
    assert expected_message in result.stdout + result.stderr
    calls = gcloud_log.read_text(encoding="utf-8")
    assert ("run jobs execute" in calls) is executes_job
    config_was_uploaded = any(
        line.startswith("storage cp ") and line.endswith("/config.json")
        for line in calls.splitlines()
    )
    assert config_was_uploaded is executes_job


@pytest.mark.skipif(
    os.name == "nt",
    reason="Deployment helpers require a POSIX shell",
)
@pytest.mark.parametrize(
    ("git_status", "expected_dirty"),
    [
        ("", "false"),
        (" M Dockerfile\n?? untracked.txt\n", "true"),
    ],
)
def test_build_image_passes_git_provenance_to_cloud_build(
    tmp_path: Path,
    git_status: str,
    expected_dirty: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud_log = tmp_path / "gcloud.log"

    _write_executable(
        bin_dir / "git",
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "rev-parse --short HEAD")
    printf '%s\n' 'abc123'
    ;;
  "status --porcelain --untracked-files=normal")
    printf '%s' "${FAKE_GIT_STATUS}"
    ;;
  *)
    exit 1
    ;;
esac
""",
    )
    _write_executable(
        bin_dir / "gcloud",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${GCLOUD_LOG}"
""",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ID": "fab-qaf-decodability",
        "GCLOUD_LOG": str(gcloud_log),
        "FAKE_GIT_STATUS": git_status,
    }

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "deploy" / "build-image.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    call = gcloud_log.read_text(encoding="utf-8")
    assert "builds submit" in call
    assert "--config=deploy/cloudbuild.yaml" in call
    assert "_CODE_GIT_HASH=abc123" in call
    assert f"_CODE_GIT_DIRTY={expected_dirty}" in call
    assert (
        "_IMAGE_URI=europe-west1-docker.pkg.dev/fab-qaf-decodability/"
        "fab-benchmarks/phonics-benchmark:latest" in call
    )
    assert f"git revision abc123 (dirty={expected_dirty})" in result.stdout


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)
