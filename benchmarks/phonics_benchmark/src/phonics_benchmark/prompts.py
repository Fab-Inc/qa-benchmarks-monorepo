from functools import cache
from importlib.resources import files

import yaml


@cache
def _load_prompts() -> dict[str, str]:
    """Load and cache prompt templates from the packaged prompts.yaml."""
    text = (
        files("phonics_benchmark").joinpath("prompts.yaml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def get_prompt(name: str, **kwargs: object) -> str:
    """Return the named prompt template, formatted with the given fields."""
    prompts = _load_prompts()
    try:
        template = prompts[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt '{name}'; available: {sorted(prompts)}"
        ) from None
    return template.format(**kwargs)
