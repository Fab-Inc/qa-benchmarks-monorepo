"""Command-line interface."""

import typer

app = typer.Typer()


@app.command()
def main() -> None:
    """Language knowledge benchmark."""


if __name__ == "__main__":
    app()
