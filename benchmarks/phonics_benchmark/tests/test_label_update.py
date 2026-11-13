"""End-to-end test of scripts/update_benchmark_item_labels.py.

The fixture writes a few Kiswahili items whose labels follow deliberately wrong
rules: every decodable word is marked non-decodable and every non-decodable word
is left out. Running the script should replace them with the labels the current
decodability rules give.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "update_benchmark_item_labels.py"


def run_script(input_path: Path, output_path: Path) -> subprocess.CompletedProcess:
    """Run the script as a subprocess and capture its log output.

    The script calls typer.run at import time, so it can't be imported. Its
    stderr is forced to UTF-8 because Windows defaults to a code page that
    cannot encode IPA symbols such as "ɪ" and would write them as "\\u026a".
    """
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(input_path), str(output_path)],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


# A deliberately tiny student: only a, m, k, t, plus "na" as a sight word.
# Words built from those graphemes are decodable; anything else is not. That
# keeps the expected labels obvious without touching the nasal rules (no word
# below has a nasal followed by a consonant).
STUDENT_KNOWLEDGE = {
    "graphemes": ["a", "m", "k", "t"],
    "clusters": [],
    "cluster_patterns": [],
    "whole_words": ["na"],
}

# (content, wrong labels written to the input, labels the script should write)
CASES = [
    # All decodable, but labelled as all non-decodable
    ("Mama na kaka.", ["kaka", "mama", "na"], []),
    # "ana" and "soko" use unknown graphemes; labels swap them for decodable words
    ("Kaka ana soko.", ["kaka"], ["ana", "soko"]),
    # Nothing decodable, but labelled as if everything were
    ("Tata ni bibi.", ["tata"], ["bibi", "ni"]),
    # All decodable; wrong label names a word not even in the content
    ("Mama kata kaka.", ["soko"], []),
]


@pytest.fixture
def wrongly_labelled_items_path(tmp_path: Path) -> Path:
    """Write the CASES as a benchmark items JSONL with the script's required keys."""
    path = tmp_path / "wrong_items.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, (content, wrong_labels, _) in enumerate(CASES):
            item = {
                "benchmark_item_id": f"test-item-{i}",
                "language_iso_code": "sw",
                "content": content,
                "student_knowledge": STUDENT_KNOWLEDGE,
                "nondecodable_words": wrong_labels,
                "num_nondecodable_words": len(wrong_labels),
                # Wrong on purpose too, so we can see the script recompute it
                "num_total_words": 99,
                "nondecodable_ratio": len(wrong_labels) / 99,
            }
            f.write(json.dumps(item) + "\n")
    return path


def test_script_corrects_wrong_labels(
    wrongly_labelled_items_path: Path, tmp_path: Path
) -> None:
    output_path = tmp_path / "relabelled_items.jsonl"

    result = run_script(wrongly_labelled_items_path, output_path)
    assert result.returncode == 0, result.stderr

    with output_path.open(encoding="utf-8") as f:
        relabelled = [json.loads(line) for line in f]

    assert len(relabelled) == len(CASES)
    for item, (content, _, expected) in zip(relabelled, CASES, strict=True):
        n_words = len(set(content.lower().rstrip(".").split()))
        assert item["content"] == content  # order and content preserved
        assert sorted(item["nondecodable_words"]) == expected, content
        assert item["num_nondecodable_words"] == len(expected)
        assert item["num_total_words"] == n_words
        assert item["nondecodable_ratio"] == pytest.approx(len(expected) / n_words)

    # Every item was wrong, so the report should say all of them changed
    assert f"Items with changed labels: {len(CASES)} / {len(CASES)}" in result.stderr


def test_script_keeps_stale_gpcs_in_saved_knowledge(tmp_path: Path) -> None:
    """A renamed GPC is read in its current form for labelling only.

    ("uni", "juni") is how items built before d9bf660 store G159. The current
    inventory rejects it, so the script must label with ("uni", "junɪ"), yet
    write the stored pair back: it is what the models saw, and the merger
    refuses overlays whose knowledge differs from the source items.
    """
    knowledge = {
        "gpcs": [["a", "æ"], ["m", "m"], ["t", "t"], ["uni", "juni"]],
        "sight_words": ["the"],
    }
    input_path = tmp_path / "stale_items.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "benchmark_item_id": "test-stale-gpc",
                "language_iso_code": "en",
                "content": "The mat.",
                "student_knowledge": knowledge,
                "nondecodable_words": [],
                "num_nondecodable_words": 0,
                "num_total_words": 2,
                "nondecodable_ratio": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "relabelled_items.jsonl"

    result = run_script(input_path, output_path)
    assert result.returncode == 0, result.stderr

    (item,) = [json.loads(line) for line in output_path.open(encoding="utf-8")]
    assert item["student_knowledge"] == knowledge
    # The script reports which stale pairs it read in their current form
    assert (
        "Labelling with stale GPC (uni, juni) read as (uni, junɪ) in 1 items"
        in result.stderr
    )
