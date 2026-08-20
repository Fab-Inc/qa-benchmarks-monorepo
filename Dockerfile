# syntax=docker/dockerfile:1

ARG UV_VERSION=0.11.8
ARG CODE_GIT_HASH=unknown
ARG CODE_GIT_DIRTY=unknown

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.13-slim-bookworm AS builder

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages
COPY benchmarks ./benchmarks

RUN uv sync --frozen --package phonics-benchmark --no-dev --no-editable


FROM python:3.13-slim-bookworm AS runtime

ARG CODE_GIT_HASH
ARG CODE_GIT_DIRTY

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    CODE_GIT_HASH="${CODE_GIT_HASH}" \
    CODE_GIT_DIRTY="${CODE_GIT_DIRTY}"

LABEL org.opencontainers.image.revision="${CODE_GIT_HASH}"

RUN groupadd --system phonics \
    && useradd --system --gid phonics --home-dir /workspace phonics \
    && mkdir -p /workspace \
    && chown phonics:phonics /workspace

# Keep installed code immutable; runtime downloads and outputs live under /workspace.
COPY --from=builder /app/.venv /app/.venv

USER phonics
WORKDIR /workspace

ENTRYPOINT ["phonics-benchmark"]
CMD ["run-cloud", "--help"]
