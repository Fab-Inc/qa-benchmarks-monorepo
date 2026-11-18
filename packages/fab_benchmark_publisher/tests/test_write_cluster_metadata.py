"""Tests for scripts/write_cluster_metadata.py, loaded by path since scripts/ is not a
package. Delete together with that script."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "write_cluster_metadata.py"
spec = importlib.util.spec_from_file_location("write_cluster_metadata", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def pairs_file(tmp_path) -> Path:
    path = tmp_path / "leaderboard_pairs.json"
    path.write_text(
        json.dumps(
            {
                "_comment": ["ignored"],
                "cluster_name": "foundational-literacy",
                "pairs": [
                    {"benchmark_domain": "LK-sw", "display_name": "LK Kiswahili"},
                    {"benchmark_domain": "DA-sw", "display_name": "DA Kiswahili"},
                    {"benchmark_domain": "LK-en", "display_name": "LK English"},
                ],
            }
        )
    )
    return path


def test_writes_cluster_metadata_in_pairs_order(tmp_path) -> None:
    path = script.main(
        ["--pairs-file", str(pairs_file(tmp_path)), "--out-root", str(tmp_path / "out")]
    )

    assert path == tmp_path / "out" / "foundational-literacy_cluster_metadata.json"
    assert json.loads(path.read_text()) == {
        "cluster_slug": "foundational-literacy",
        "display_name": "Foundational Literacy",
        "benchmark_ids": ["LK-sw", "DA-sw", "LK-en"],
    }


def test_display_name_flag_overrides_the_default(tmp_path) -> None:
    path = script.main(
        [
            "--pairs-file",
            str(pairs_file(tmp_path)),
            "--out-root",
            str(tmp_path),
            "--display-name",
            "Early Literacy",
        ]
    )

    assert json.loads(path.read_text())["display_name"] == "Early Literacy"


def test_repo_pairs_file_is_valid(tmp_path) -> None:
    """The real leaderboard_pairs.json must produce valid cluster metadata."""
    meta = script.cluster_meta_from_pairs(
        script.REPO_ROOT / "leaderboard_pairs.json", None
    )
    assert meta.cluster_slug == "foundational-literacy"
    assert "LK-en" in meta.benchmark_ids and "LK-sw" in meta.benchmark_ids
