"""The shared fab-bench image and the scripts that build, deploy and launch it.

The scripts run against fake ``git``, ``gcloud`` and ``fab-bench`` commands that log
their arguments; nothing reaches Google Cloud.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PACKAGE_DIR / "deploy"
SCRIPTS = ("build-image.sh", "deploy-job.sh", "launch-run.sh")

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="Deployment scripts require a POSIX shell"
)


def test_the_deploy_files_are_all_there() -> None:
    assert sorted(path.name for path in DEPLOY_DIR.iterdir()) == sorted(
        ["Dockerfile", "cloudbuild.yaml", *SCRIPTS]
    )
    for script in SCRIPTS:
        assert os.access(DEPLOY_DIR / script, os.X_OK), script


def test_the_image_installs_the_locked_runner_and_runs_fab_bench() -> None:
    dockerfile = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "RUN uv sync --frozen --no-dev --no-editable\n" in dockerfile
    assert 'FAB_BENCH_RUNNER_COMMIT="${RUNNER_COMMIT}"' in dockerfile
    assert 'RUN test -n "${RUNNER_COMMIT}"' in dockerfile
    assert 'ENTRYPOINT ["fab-bench"]' in dockerfile
    assert "ARG PYTHON_VERSION=3.12\n" in dockerfile


def test_the_image_uses_a_uv_the_package_accepts() -> None:
    dockerfile = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    pyproject = tomllib.loads((PACKAGE_DIR / "pyproject.toml").read_text("utf-8"))

    [image_uv] = re.findall(r"^ARG UV_VERSION=([\d.]+)$", dockerfile, re.MULTILINE)
    minimum = pyproject["tool"]["uv"]["required-version"].removeprefix(">=")
    assert _version(image_uv) >= _version(minimum)


def test_cloud_build_builds_smoke_tests_and_checks_recorded_versions() -> None:
    cloud_build = yaml.safe_load(
        (DEPLOY_DIR / "cloudbuild.yaml").read_text(encoding="utf-8")
    )

    build, smoke, versions = cloud_build["steps"]
    assert build["args"][:3] == ["build", "--file", "deploy/Dockerfile"]
    assert "RUNNER_COMMIT=${_RUNNER_COMMIT}" in build["args"]
    assert smoke["args"] == ["run", "--rm", "${_IMAGE_URI}", "run-cloud", "--help"]
    assert (
        'found["runner"]["commit"] and found["fdllm"]["commit"]'
        in (versions["args"][-1])
    )
    assert cloud_build["images"] == ["${_IMAGE_URI}"]


def test_ignore_files_send_only_what_the_build_needs() -> None:
    needed = [
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!src/",
        "!deploy/Dockerfile",
    ]
    for name, extra in ((".dockerignore", []), (".gcloudignore", ["!.dockerignore"])):
        lines = (PACKAGE_DIR / name).read_text(encoding="utf-8").splitlines()
        rules = [line for line in lines if line and not line.startswith("#")]

        assert rules[0] == "*", name
        for rule in [*needed, *extra, "**/.env", "**/.env.*"]:
            assert rule in rules, (name, rule)
    gcloud = (PACKAGE_DIR / ".gcloudignore").read_text(encoding="utf-8")
    assert "!deploy/cloudbuild.yaml" in gcloud.splitlines()


@posix_only
@pytest.mark.parametrize(("status", "builds"), [("", True), (" M src/x.py\n", False)])
def test_build_image_tags_the_image_with_a_clean_commit(
    tmp_path: Path, status: str, builds: bool
) -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    bin_dir = _fake_bin(
        tmp_path,
        git=f"""case "$*" in
  "status --porcelain --untracked-files=normal -- .")
    printf '%s' "${{FAKE_GIT_STATUS}}" ;;
  "rev-parse HEAD") printf '%s\\n' {commit} ;;
  *) exit 1 ;;
esac
""",
    )

    result = _script(
        tmp_path, "build-image.sh", bin_dir=bin_dir, FAKE_GIT_STATUS=status
    )

    calls = _calls(tmp_path)
    if not builds:
        assert result.returncode == 1
        assert "uncommitted changes" in result.stderr
        assert calls == ""
        return
    assert result.returncode == 0, result.stderr
    image = (
        "europe-west1-docker.pkg.dev/project-a/fab-benchmarks/fab-bench:0123456789ab"
    )
    assert calls.startswith(f"cwd={tmp_path / 'package'}\n")
    assert "builds submit --project=project-a --config=deploy/cloudbuild.yaml" in calls
    assert f"_IMAGE_URI={image},_RUNNER_COMMIT={commit}" in calls
    assert f"IMAGE={image} deploy/deploy-job.sh" in result.stdout


@posix_only
def test_deploy_job_runs_the_given_image_and_records_it(tmp_path: Path) -> None:
    image = "europe-west1-docker.pkg.dev/shared/fab-benchmarks/fab-bench:0123456789ab"

    result = _script(
        tmp_path,
        "deploy-job.sh",
        IMAGE=image,
        SERVICE_ACCOUNT="runner@project-a.iam.gserviceaccount.com",
        SECRET_MAPPINGS="OPENAI_API_KEY=openai:latest",
        MEMORY="4Gi",
    )

    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    assert "run jobs deploy fab-bench --project=project-a" in calls
    assert f"--image={image}" in calls
    assert f"--set-env-vars=FAB_BENCH_IMAGE={image}" in calls
    assert "--memory=4Gi" in calls
    assert "--task-timeout=48h" in calls
    assert "--max-retries=2" in calls
    assert "--set-secrets=OPENAI_API_KEY=openai:latest" in calls


@posix_only
def test_deploy_job_needs_an_image(tmp_path: Path) -> None:
    result = _script(tmp_path, "deploy-job.sh", SERVICE_ACCOUNT="runner@x")

    assert result.returncode != 0
    assert "Set IMAGE" in result.stderr
    assert _calls(tmp_path) == ""


@posix_only
@pytest.mark.parametrize(
    ("extra_env", "upload_args", "execute_flags"),
    [
        ({}, "upload-run-config run.json", ""),
        (
            {"RUN_CONFIGS_URI": "gs://configs/mathv", "WAIT": "true"},
            "upload-run-config run.json --dest gs://configs/mathv",
            " --wait",
        ),
    ],
)
def test_launch_run_uploads_the_config_then_runs_the_job_on_its_uri(
    tmp_path: Path, extra_env: dict[str, str], upload_args: str, execute_flags: str
) -> None:
    uri = "gs://bucket/run-configs/" + "a" * 64 + ".json"
    fab_bench = _fake_command(
        tmp_path,
        "fake-fab-bench",
        f'printf \'fab-bench %s\\n\' "$*" >>"${{GCLOUD_LOG}}"\necho {uri}\n',
    )

    result = _script(
        tmp_path,
        "launch-run.sh",
        "run.json",
        FAB_BENCH=f"{fab_bench}",
        JOB_NAME="fab-bench-large",
        **extra_env,
    )

    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path).splitlines()
    assert calls[0] == f"fab-bench {upload_args}"
    assert calls[1] == (
        "run jobs execute fab-bench-large --project=project-a --region=europe-west1 "
        f"--args=run-cloud,--config,{uri}{execute_flags}"
    )


@posix_only
def test_launch_run_stops_when_the_config_is_refused(tmp_path: Path) -> None:
    fab_bench = _fake_command(
        tmp_path, "fake-fab-bench", 'echo "ERROR: local experiment" >&2\nexit 1\n'
    )

    result = _script(tmp_path, "launch-run.sh", "run.json", FAB_BENCH=str(fab_bench))

    assert result.returncode == 1
    assert "ERROR: local experiment" in result.stderr
    assert "run jobs execute" not in _calls(tmp_path)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _script(
    tmp_path: Path,
    name: str,
    *args: str,
    bin_dir: Path | None = None,
    **extra_env: str,
) -> subprocess.CompletedProcess[str]:
    """Run a deploy script from a copy of the package's deploy folder."""

    package = tmp_path / "package"
    if not package.exists():
        shutil.copytree(DEPLOY_DIR, package / "deploy")
    bin_dir = bin_dir or _fake_bin(tmp_path)
    return subprocess.run(
        ["bash", f"deploy/{name}", *args],
        cwd=package,
        env=_env(tmp_path, bin_dir, **extra_env),
        check=False,
        capture_output=True,
        text=True,
    )


def _calls(tmp_path: Path) -> str:
    log = tmp_path / "gcloud.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _fake_command(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)
    return path


def _fake_bin(tmp_path: Path, *, git: str | None = None) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gcloud = """if [[ "$1 $2" == "builds submit" ]]; then
  printf 'cwd=%s\\n' "$PWD" >>"${GCLOUD_LOG}"
fi
printf '%s\\n' "$*" >>"${GCLOUD_LOG}"
"""
    _fake_command(bin_dir, "gcloud", gcloud)
    if git is not None:
        _fake_command(bin_dir, "git", git)
    return bin_dir


def _env(tmp_path: Path, bin_dir: Path, **extra: str) -> dict[str, str]:
    """The test environment, without deploy settings from the caller's shell."""

    settings = {
        "ARTIFACT_REPOSITORY",
        "FAB_BENCH",
        "IMAGE",
        "IMAGE_NAME",
        "JOB_NAME",
        "REGION",
        "RUN_CONFIGS_URI",
        "SECRET_MAPPINGS",
        "WAIT",
    }
    env = {key: value for key, value in os.environ.items() if key not in settings}
    return {
        **env,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ID": "project-a",
        "GCLOUD_LOG": str(tmp_path / "gcloud.log"),
        **extra,
    }
