"""Which code produced a run: the image's baked-in revision, or the local checkout."""

from __future__ import annotations

import os
import subprocess
import sys


def git_hash() -> str | None:
    """Return the image revision or the current checkout's short git hash."""

    image_revision = os.environ.get("CODE_GIT_HASH", "").strip()
    if image_revision and image_revision != "unknown":
        return image_revision
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as e:
        print(f"WARNING: could not read git hash: {e}", file=sys.stderr)
        return None


def git_dirty() -> bool | None:
    """Return whether the image source or current checkout had local changes."""

    image_dirty = os.environ.get("CODE_GIT_DIRTY", "").strip().lower()
    if image_dirty == "true":
        return True
    if image_dirty == "false":
        return False
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(status.strip())
    except Exception as e:
        print(f"WARNING: could not read git dirty state: {e}", file=sys.stderr)
        return None
