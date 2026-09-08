"""Tests for extracting the generated text out of a raw model response."""

import pytest
from phonics_benchmark.parsing import ExtractResult, extract_generated_text


class TestDelimiterFound:
    """Responses that honour the <text>...</text> output contract."""

    def test_extracts_text_between_tags(self) -> None:
        """The tag contents are returned without the tags themselves."""
        result = extract_generated_text("<text>Mama anaona paka.</text>")
        assert result == ExtractResult("Mama anaona paka.", True)

    def test_strips_surrounding_commentary(self) -> None:
        """Prose before and after the tags is discarded, not scored."""
        response = (
            "Sure! Here is a story:\n\n<text>\nMama anaona paka.\n</text>\n\n"
            "I hope this helps!"
        )
        assert extract_generated_text(response) == ExtractResult(
            "Mama anaona paka.", True
        )

    def test_text_may_span_multiple_lines(self) -> None:
        """Internal newlines are preserved so paragraph breaks survive."""
        result = extract_generated_text("<text>\nLine one.\n\nLine two.\n</text>")
        assert result.text == "Line one.\n\nLine two."
        assert result.delimiter_found is True

    @pytest.mark.parametrize(
        "response",
        [
            "<TEXT>Mama.</TEXT>",
            "<Text>Mama.</Text>",
            "< text >Mama.< / text >",
        ],
    )
    def test_tags_are_case_and_whitespace_tolerant(self, response: str) -> None:
        """Models vary the tag casing and spacing; all forms are accepted."""
        assert extract_generated_text(response) == ExtractResult("Mama.", True)

    def test_first_complete_pair_wins(self) -> None:
        """Two tag pairs yield the first block, not everything between them."""
        response = "<text>First.</text> junk <text>Second.</text>"
        assert extract_generated_text(response).text == "First."

    def test_empty_tags_give_empty_text(self) -> None:
        """A well-formed but empty block is a found delimiter with no text."""
        assert extract_generated_text("<text></text>") == ExtractResult("", True)


class TestDelimiterMissing:
    """Responses that ignored the output contract."""

    def test_missing_tags_returns_whole_response_and_flags_it(self) -> None:
        """Without tags the caller is told, so it can score a format failure."""
        result = extract_generated_text("  Mama anaona paka.  ")
        assert result == ExtractResult("Mama anaona paka.", False)

    def test_unclosed_tag_is_not_a_match(self) -> None:
        """A half-written block is a format failure rather than a lucky parse."""
        result = extract_generated_text("<text>Mama anaona paka.")
        assert result.delimiter_found is False

    def test_empty_response(self) -> None:
        """An empty response is a format failure with no text."""
        assert extract_generated_text("") == ExtractResult("", False)
