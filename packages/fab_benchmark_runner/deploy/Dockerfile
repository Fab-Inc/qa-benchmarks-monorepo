# syntax=docker/dockerfile:1

# Benchmark image for Cloud Run jobs, from fab-benchmark-runner's deploy/ folder.
# Copy it to deploy/Dockerfile in the benchmark's repository and set the values
# marked SET. deploy/build-image.sh builds it with the repository root as context.

ARG UV_VERSION=0.11.8
# SET: the Python version the benchmark runs on.
ARG PYTHON_VERSION=3.12
ARG CODE_GIT_HASH=unknown
ARG CODE_GIT_DIRTY=unknown

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /uvx /bin/

# git fetches git-sourced dependencies such as fab-benchmark-runner and fabdata-llm.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# .dockerignore limits the build context to what the install needs.
COPY . .

# SET in a uv workspace: the member that provides the benchmark CLI. Leave it
# empty for a single-project repository.
ARG UV_PACKAGE=
RUN uv sync --frozen --no-dev --no-editable ${UV_PACKAGE:+--package "$UV_PACKAGE"}


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG CODE_GIT_HASH
ARG CODE_GIT_DIRTY

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    CODE_GIT_HASH="${CODE_GIT_HASH}" \
    CODE_GIT_DIRTY="${CODE_GIT_DIRTY}"

LABEL org.opencontainers.image.revision="${CODE_GIT_HASH}"

RUN groupadd --system benchmark \
    && useradd --system --gid benchmark --home-dir /workspace benchmark \
    && mkdir -p /workspace \
    && chown benchmark:benchmark /workspace

# Installed code stays immutable; downloads and run outputs live under /workspace.
COPY --from=builder /app/.venv /app/.venv

USER benchmark
WORKDIR /workspace

# SET: the benchmark's console script, from [project.scripts].
ENTRYPOINT ["my-benchmark"]
CMD ["run-cloud", "--help"]
