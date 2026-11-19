"""The ONE place internal field names are translated to the names the Fab website reads.

Internally, prices are named after models.csv's columns, with their unit:
input_price_usd_per_mtok, output_price_usd_per_mtok, and the derived
blended_price_usd_per_mtok. The website's schema still uses the names from before that
rename, so anything written for the website must go through website_name() at the
point of export.

To switch the website to the new names: empty WEBSITE_NAMES (website_name() then
returns every name unchanged). To remove this module entirely: replace each
website_name(X) call with X and delete this file and tests/test_website_names.py.
"""

from __future__ import annotations

# internal name -> name in the website's files
WEBSITE_NAMES: dict[str, str] = {
    "input_price_usd_per_mtok": "input_price",
    "output_price_usd_per_mtok": "output_price",
    "blended_price_usd_per_mtok": "blended_price",
}


def website_name(name: str) -> str:
    """The name the website expects for internal field `name` (itself if unmapped)."""
    return WEBSITE_NAMES.get(name, name)
