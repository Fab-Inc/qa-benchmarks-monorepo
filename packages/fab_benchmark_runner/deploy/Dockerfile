# syntax=docker/dockerfile:1

# The shared fab-bench image for Cloud Run jobs: the runner and its locked
# dependencies, and no benchmark code. A job runs `fab-bench run-cloud --config
# gs://…`. deploy/build-image.sh builds it with the runner package's folder as the
# context.

ARG UV_VERSION=0.11.8
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /uvx /bin/

# git fetches FDLLM, which is installed from its git repository.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# .dockerignore keeps the context to what the install needs.
COPY . .
RUN uv sync --frozen --no-dev --no-editable


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# The runner's git commit: the install above carries no git metadata, so runs
# read it from FAB_BENCH_RUNNER_COMMIT.
ARG RUNNER_COMMIT
RUN test -n "${RUNNER_COMMIT}" || { echo "Set the RUNNER_COMMIT build arg" >&2; exit 1; }

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    FAB_BENCH_RUNNER_COMMIT="${RUNNER_COMMIT}"

LABEL org.opencontainers.image.revision="${RUNNER_COMMIT}"

RUN groupadd --system fab-bench \
    && useradd --system --gid fab-bench --home-dir /workspace fab-bench \
    && mkdir -p /workspace \
    && chown fab-bench:fab-bench /workspace

# Installed code stays immutable; downloads and results live under /workspace.
COPY --from=builder /app/.venv /app/.venv

USER fab-bench
WORKDIR /workspace

ENTRYPOINT ["fab-bench"]
CMD ["run-cloud", "--help"]
