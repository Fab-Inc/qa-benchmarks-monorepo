"""The deploy files a benchmark copies into its repository, and their scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
SCRIPTS = ("build-image.sh", "deploy-job.sh", "launch-run.sh")

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="Deployment scripts require a POSIX shell"
)


def test_the_files_a_benchmark_copies_are_all_there() -> None:
    assert sorted(path.name for path in DEPLOY_DIR.iterdir()) == sorted(
        [
            "Dockerfile",
            "cloudbuild.yaml",
            "dockerignore",
            "gcloudignore",
            *SCRIPTS,
        ]
    )
    for script in SCRIPTS:
        assert os.access(DEPLOY_DIR / script, os.X_OK), script


def test_values_to_set_are_marked() -> None:
    dockerfile = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "# SET: the Python version the benchmark runs on.\nARG PYTHON_VERSION=" in (
        dockerfile
    )
    assert "\nARG UV_PACKAGE=\n" in dockerfile
    assert (
        "# SET: the benchmark's console script, from [project.scripts].\n"
        'ENTRYPOINT ["my-benchmark"]'
    ) in dockerfile
    for script in SCRIPTS:
        text = (DEPLOY_DIR / script).read_text(encoding="utf-8")
        assert "# SET: the benchmark's name." in text, script
        assert 'BENCHMARK="${BENCHMARK:-my-benchmark}"' in text, script


def test_cloud_build_builds_the_dockerfile_and_smoke_tests_it() -> None:
    cloud_build = yaml.safe_load(
        (DEPLOY_DIR / "cloudbuild.yaml").read_text(encoding="utf-8")
    )

    build, smoke = cloud_build["steps"]
    assert build["args"][:3] == ["build", "--file", "deploy/Dockerfile"]
    assert smoke["args"] == ["run", "--rm", "${_IMAGE_URI}", "run-cloud", "--help"]
    assert cloud_build["images"] == ["${_IMAGE_URI}"]


def test_ignore_files_allow_what_the_build_needs() -> None:
    docker = (DEPLOY_DIR / "dockerignore").read_text(encoding="utf-8").splitlines()
    gcloud = (DEPLOY_DIR / "gcloudignore").read_text(encoding="utf-8").splitlines()

    for needed in ("!pyproject.toml", "!uv.lock", "!src/", "!deploy/Dockerfile"):
        assert needed in docker
        assert needed in gcloud
    assert "!deploy/cloudbuild.yaml" in gcloud
    assert "!.dockerignore" in gcloud
    assert "**/.env" in docker
    assert "**/.env" in gcloud


@posix_only
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
    local_items = tmp_path / "local-items.jsonl"
    local_items.write_bytes(b'{"id":"item-1"}\n')
    remote = tmp_path / "remote-items.jsonl"
    remote.write_bytes(remote_items)

    result, calls = _launch(
        tmp_path, local_items, remote, existing="datasets", BENCHMARK="mathv"
    )

    assert result.returncode == expected_exit_code
    assert expected_message in result.stdout + result.stderr
    assert ("run jobs execute mathv-run" in calls) is executes_job
    uploaded_config = any(
        line.startswith("storage cp ") and line.endswith("/run-configs/run-a.json")
        for line in calls.splitlines()
    )
    assert uploaded_config is executes_job
    if executes_job:
        assert (
            "--args=run-cloud,--config,gs://bucket/mathv/run-configs/run-a.json"
            in calls
        )


@posix_only
def test_launch_run_uploads_items_with_their_extension(tmp_path: Path) -> None:
    local_items = tmp_path / "level1.parquet"
    local_items.write_bytes(b"PAR1")

    result, calls = _launch(tmp_path, local_items, None, existing=None)

    assert result.returncode == 0, result.stderr
    assert (
        f"storage cp --if-generation-match=0 {local_items} "
        "gs://bucket/my-benchmark/datasets/run-a.parquet" in calls
    )
    assert "run jobs execute my-benchmark-run" in calls


@posix_only
def test_launch_run_prefix_and_job_can_be_overridden(tmp_path: Path) -> None:
    local_items = tmp_path / "items.jsonl"
    local_items.write_bytes(b"{}\n")

    result, calls = _launch(
        tmp_path,
        local_items,
        None,
        existing=None,
        BENCHMARK="mathv",
        GCS_PREFIX="shared/mathv",
        JOB_NAME="mathv-large",
    )

    assert result.returncode == 0, result.stderr
    assert "gs://bucket/shared/mathv/run-configs/run-a.json" in calls
    assert "run jobs execute mathv-large" in calls


@posix_only
@pytest.mark.parametrize(("status", "dirty"), [("", "false"), (" M x.py\n", "true")])
def test_build_image_passes_provenance_to_cloud_build(
    tmp_path: Path, status: str, dirty: str
) -> None:
    repo = _repo(tmp_path)
    bin_dir = _fake_bin(
        tmp_path,
        git="""case "$*" in
  "rev-parse --short HEAD") printf '%s\\n' abc123 ;;
  "status --porcelain --untracked-files=normal") printf '%s' "${FAKE_GIT_STATUS}" ;;
  *) exit 1 ;;
esac
""",
    )

    result = subprocess.run(
        ["bash", "deploy/build-image.sh"],
        cwd=repo,
        env=_env(tmp_path, bin_dir, FAKE_GIT_STATUS=status, BENCHMARK="mathv"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    call = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "builds submit --project=project-a --config=deploy/cloudbuild.yaml" in call
    assert f"_CODE_GIT_DIRTY={dirty},_CODE_GIT_HASH=abc123" in call
    assert (
        "_IMAGE_URI=europe-west1-docker.pkg.dev/project-a/fab-benchmarks/mathv:latest"
        in call
    )


@posix_only
def test_deploy_job_configures_the_cloud_run_job(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    bin_dir = _fake_bin(tmp_path)

    result = subprocess.run(
        ["bash", "deploy/deploy-job.sh"],
        cwd=repo,
        env=_env(
            tmp_path,
            bin_dir,
            BENCHMARK="mathv",
            SERVICE_ACCOUNT="runner@project-a.iam.gserviceaccount.com",
            SECRET_MAPPINGS="OPENAI_API_KEY=openai:latest",
            MEMORY="4Gi",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    call = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert call.startswith("run jobs deploy mathv-run --project=project-a")
    assert (
        "--image=europe-west1-docker.pkg.dev/project-a/fab-benchmarks/mathv:latest"
        in call
    )
    assert "--memory=4Gi" in call
    assert "--task-timeout=48h" in call
    assert "--set-secrets=OPENAI_API_KEY=openai:latest" in call


def _repo(tmp_path: Path) -> Path:
    """A benchmark repository with the deploy files copied in."""

    repo = tmp_path / "repo"
    shutil.copytree(DEPLOY_DIR, repo / "deploy")
    return repo


def _launch(
    tmp_path: Path,
    local_items: Path,
    remote_items: Path | None,
    *,
    existing: str | None,
    **extra_env: str,
) -> tuple[subprocess.CompletedProcess[str], str]:
    repo = _repo(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    bin_dir = _fake_bin(
        tmp_path,
        jq='input="${@: -1}"\ncat "${input}"\n',
        gcloud_extra=f"""if [[ "$1 $2" == "storage ls" ]]; then
  case "$3" in
    */{existing or "none"}/*) exit 0 ;;
  esac
  exit 1
fi
if [[ "$1 $2" == "storage cp" && "$3" == gs://* ]]; then
  cp "{remote_items}" "$4"
fi
""",
    )
    result = subprocess.run(
        ["bash", "deploy/launch-run.sh", str(config), str(local_items), "run-a"],
        cwd=repo,
        env=_env(tmp_path, bin_dir, RESULTS_BUCKET="gs://bucket", **extra_env),
        check=False,
        capture_output=True,
        text=True,
    )
    return result, (tmp_path / "gcloud.log").read_text(encoding="utf-8")


def _fake_bin(
    tmp_path: Path,
    *,
    git: str | None = None,
    jq: str | None = None,
    gcloud_extra: str = "",
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    scripts = {
        "gcloud": 'printf \'%s\\n\' "$*" >>"${GCLOUD_LOG}"\n' + gcloud_extra,
        "git": git,
        "jq": jq,
    }
    for name, body in scripts.items():
        if body is None:
            continue
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)
    return bin_dir


def _env(tmp_path: Path, bin_dir: Path, **extra: str) -> dict[str, str]:
    """The test environment, without any deploy overrides from the caller's shell."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"BENCHMARK", "GCS_PREFIX", "JOB_NAME", "IMAGE_NAME"}
    }
    return {
        **env,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ID": "project-a",
        "GCLOUD_LOG": str(tmp_path / "gcloud.log"),
        **extra,
    }
