"""The FDLLM call wrapper and the default way runs call models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from fab_benchmark_runner import ImageRef, ModelInput
from fab_benchmark_runner.llm import call_llm, call_model_with_fdllm


def test_call_llm_sends_system_and_user_messages() -> None:
    caller = MagicMock()
    caller.call.return_value = SimpleNamespace(
        Message="answer",
        TokensUsed=30,
        TokensUsedCompletion=10,
        TokensUsedReasoning=4,
        ResponseModel="model-a-2026",
    )

    with patch("fdllm.get_caller", return_value=caller) as get_caller:
        record = call_llm(
            "prompt", model="model-a", max_tokens=99, system_prompt="Be brief."
        )

    get_caller.assert_called_once_with("model-a")
    messages = caller.call.call_args.args[0]
    assert [(m.Role, m.Message) for m in messages] == [
        ("system", "Be brief."),
        ("user", "prompt"),
    ]
    assert caller.call.call_args.kwargs == {"max_tokens": 99}
    assert record["response_text"] == "answer"
    assert record["tokens_used"] == 30
    assert record["tokens_used_completion"] == 10
    assert record["tokens_used_reasoning"] == 4
    assert record["model_version"] == "model-a-2026"
    assert record["max_tokens"] == 99
    assert record["latency_seconds"] >= 0


def test_call_llm_treats_a_missing_response_as_empty() -> None:
    caller = MagicMock()
    caller.call.return_value = None

    with patch("fdllm.get_caller", return_value=caller):
        record = call_llm("prompt", model="model-a", max_tokens=5)

    assert [m.Role for m in caller.call.call_args.args[0]] == ["user"]
    assert record["response_text"] == ""
    assert record["tokens_used"] is None


def test_runs_call_fdllm_with_the_stored_input_and_the_experiments_limit() -> None:
    record = {
        "response_text": "B",
        "max_tokens": 321,
        "tokens_used": 12,
        "tokens_used_completion": 7,
        "tokens_used_reasoning": None,
        "model_version": None,
        "timestamp_utc": "20261116T120000_000000Z",
        "latency_seconds": 1.25,
    }
    model_input = ModelInput("mcq-1", "Which?", "Answer with one letter.")

    with patch("fab_benchmark_runner.llm.call_llm", return_value=record) as call:
        payload = call_model_with_fdllm("model-a", model_input, 321)

    call.assert_called_once_with(
        "Which?",
        model="model-a",
        max_tokens=321,
        system_prompt="Answer with one letter.",
        images=(),
    )
    assert payload == {
        "output": "B",
        "tokens_used": 12,
        "tokens_used_completion": 7,
        "tokens_used_reasoning": None,
        "model_version": None,
        "timestamp_utc": "20261116T120000_000000Z",
        "latency_seconds": 1.25,
    }


def test_call_llm_puts_images_in_the_user_message_as_fdllm_resizes_them(
    tmp_path: Path,
) -> None:
    wide = _image(tmp_path / "wide.png", "red", (4096, 1024))
    small = _image(tmp_path / "small.png", "blue", (40, 30))
    caller = MagicMock()
    caller.call.return_value = SimpleNamespace(Message="answer")

    with patch("fdllm.get_caller", return_value=caller):
        call_llm(
            "Compare.",
            model="model-a",
            max_tokens=10,
            images=(ImageRef(str(wide), "high", "a" * 64), ImageRef(str(small), "low")),
        )

    [user] = caller.call.call_args.args[0]
    assert user.Message == "Compare."
    assert [(image.Detail, image.Img.size) for image in user.Images] == [
        ("high", (2048, 512)),
        ("low", (512, 512)),
    ]
    assert user.Images[1].Img.convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_each_call_builds_its_own_fdllm_images(tmp_path: Path) -> None:
    """Calls running in parallel share no image objects."""

    image = ImageRef(str(_image(tmp_path / "f.png", "red", (40, 30))), "high")
    caller = MagicMock()
    caller.call.return_value = SimpleNamespace(Message="answer")

    with patch("fdllm.get_caller", return_value=caller):
        call_llm("?", model="model-a", max_tokens=10, images=(image,))
        call_llm("?", model="model-a", max_tokens=10, images=(image,))

    first, second = (call.args[0][0].Images[0] for call in caller.call.call_args_list)
    assert first is not second
    assert first.Img is not second.Img


def test_a_text_call_sends_no_images() -> None:
    caller = MagicMock()
    caller.call.return_value = SimpleNamespace(Message="answer")

    with patch("fdllm.get_caller", return_value=caller):
        call_llm("?", model="model-a", max_tokens=10)

    assert caller.call.call_args.args[0][0].Images is None


def test_no_limit_is_passed_to_fdllm_as_none() -> None:
    """FDLLM then sizes the call to what the model's context window leaves."""

    caller = MagicMock()
    caller.call.return_value = SimpleNamespace(Message="answer")

    with patch("fdllm.get_caller", return_value=caller):
        record = call_llm("?", model="model-a", max_tokens=None)

    assert caller.call.call_args.kwargs == {"max_tokens": None}
    assert record["max_tokens"] is None


def test_runs_hand_fdllm_the_inputs_images() -> None:
    image = ImageRef("/tmp/copy", "high", "a" * 64, source_uri="gs://b/1.png")
    model_input = ModelInput("v-1", "Which?", images=(image,))

    with patch("fab_benchmark_runner.llm.call_llm", return_value=_RECORD) as call:
        call_model_with_fdllm("model-a", model_input, None)

    assert call.call_args.kwargs["images"] == (image,)
    assert call.call_args.kwargs["max_tokens"] is None


_RECORD = {
    "response_text": "B",
    "max_tokens": None,
    "tokens_used": None,
    "tokens_used_completion": None,
    "tokens_used_reasoning": None,
    "model_version": None,
    "timestamp_utc": "20261116T120000_000000Z",
    "latency_seconds": 1.0,
}


def _image(path: Path, color: str, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, color).save(path)
    return path
