"""Tests for the Cloud Run deployment helper scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)
