"""Command-line interface."""

import typer

app = typer.Typer()


@app.command()
def main() -> None:
    """Phonics benchmark."""


if __name__ == "__main__":
    app()
