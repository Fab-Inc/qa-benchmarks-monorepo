"""Tests for website_names.py. Delete together with that module."""

from fab_benchmark_publisher.website_names import WEBSITE_NAMES, website_name


def test_internal_price_names_map_to_the_website_names() -> None:
    assert website_name("input_price_usd_per_mtok") == "input_price"
    assert website_name("output_price_usd_per_mtok") == "output_price"
    assert website_name("blended_price_usd_per_mtok") == "blended_price"


def test_unmapped_names_pass_through_unchanged() -> None:
    assert website_name("size") == "size"
    assert "size" not in WEBSITE_NAMES
