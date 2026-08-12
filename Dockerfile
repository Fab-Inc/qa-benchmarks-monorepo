# syntax=docker/dockerfile:1

ARG UV_VERSION=0.11.8

FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages
COPY benchmarks ./benchmarks

RUN uv sync --frozen --package phonics-benchmark --no-dev --no-editable


FROM python:3.13-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

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
