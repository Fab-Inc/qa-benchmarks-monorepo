"""Command-line interface."""

from pathlib import Path

import typer

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

app = typer.Typer()


@app.command()
def main() -> None:
    """Phonics benchmark."""


if __name__ == "__main__":
    app()
