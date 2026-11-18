"""End-to-end test of the documented path with a stubbed model

A one-text workbook is generated in ``tmp_path``; the same functions the commands call
run ingestion, item building, the harness with a fake evaluator, and extraction. No
provider or network call is made and no workbook is committed.
"""

import json

from phonics_benchmark.config import load_run_config
from phonics_benchmark.runner import execute_run
from reading_comprehension_question_generation.items import load_items
from reading_comprehension_question_generation.pairs import extract_run
from reading_comprehension_question_generation.prompts import (
    DEFAULT_PROMPT_NAME,
    prompt_for_item,
)

THREE = json.dumps([{"question": f"Q{i}?", "answer": f"A{i}"} for i in (1, 2, 3)])


def test_one_text_one_model_goes_from_workbook_to_three_pairs(
    tmp_path, ingest_script, build_script, text_row, write_workbook
) -> None:
    workbook = write_workbook(
        tmp_path / "one_text.xlsx",
        [text_row(text_id="ENG-E2E-001", text_content="A hen sat on a mat.")],
    )
    data_root = tmp_path / "data"
    ingest_script.ingest_workbook(workbook, "english", data_root)
    items_path, item_count = build_script.build_items_for_language("english", data_root)
    assert item_count == 1

    models_path = tmp_path / "custom_models.yaml"
    models_path.write_text(
        json.dumps({"provider-a": {"models": {"model-a": {}}}}), encoding="utf-8"
    )
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "e2e",
                "language": "english",
                "items_path": str(items_path),
                "output_dir": str(tmp_path / "out"),
                "models": ["model-a"],
                "model_workers": 1,
                "custom_models_path": str(models_path),
            }
        ),
        encoding="utf-8",
    )
    prompts_seen: list[str] = []

    def stub(_config, _model, _item, prompt: str) -> dict:
        prompts_seen.append(prompt)
        return {"output": THREE}

    output = execute_run(
        load_run_config(
            config_path,
            task="question-generation",
            default_prompt_name=DEFAULT_PROMPT_NAME,
        ),
        evaluator=stub,
        item_loader=load_items,
        prompt_builder=prompt_for_item,
    )
    assert (output["executed_count"], output["error_count"]) == (1, 0)
    assert len(prompts_seen) == 1 and "A hen sat on a mat." in prompts_seen[0]

    summary = extract_run(tmp_path / "out" / "e2e")

    assert summary["models"]["model-a"] == {
        "responses": 1,
        "total_pairs": 3,
        "status_counts": {"ok": 1},
    }
    assert summary["items_file"]["match"] is True  # the harness's real config.json
    pairs = [
        json.loads(line)
        for line in (tmp_path / "out" / "e2e" / "pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(pairs) == 3
    assert {(p["text_id"], p["benchmark_item_id"], p["model"]) for p in pairs} == {
        ("ENG-E2E-001", "ENG-E2E-001", "model-a")
    }
