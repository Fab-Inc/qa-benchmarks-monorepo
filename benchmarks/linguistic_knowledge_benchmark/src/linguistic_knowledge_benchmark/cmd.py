"""Command-line interface for linguistic_knowledge_benchmark."""

from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from fdllm import register_models

from linguistic_knowledge_benchmark.judge_bench import (
    compute_summary,
    judge_results,
    judge_results_fallback,
)
from linguistic_knowledge_benchmark.run_bench import run_benchmark
from linguistic_knowledge_benchmark.shared_utils import DATA_DIR, EVAL_MODELS

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load provider API keys from the package-level .env into os.environ so fdllm
# can find them. Existing environment variables take precedence (override=False).
load_dotenv(PROJECT_ROOT / ".env")

_MODELS_YAML = PROJECT_ROOT / "custom_models.yaml"
register_models(str(_MODELS_YAML))

# Add a new language by adding an entry here and dropping its JSON in data/.
# The medium of evaluation is English for every language: questions are written
# in English (only their subject changes from language to language).
DATASETS: dict[str, dict] = {
    "luganda": {
        "path": DATA_DIR / "linguistic_knowledge_eval_luganda.json",
        "language_name": "Luganda",
    },
}

app = typer.Typer(
    name="linguistic-knowledge-benchmark",
    help="Benchmark system for evaluating LLM performance on language knowledge evals",
    no_args_is_help=True,
)


@app.command("run-bench")
def run_bench_cmd(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name to run evaluation on",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-run even if results already exist",
    ),
    language: str = typer.Option(
        "luganda",
        "--language",
        "-l",
        help=f"Language: one of {', '.join(DATASETS)}",
    ),
) -> None:
    """Run Linguistic Knowledge evaluation on LLM models.

    If --model is not provided, runs all models from list_models.yaml.
    """
    if language not in DATASETS:
        typer.echo(f"ERROR: Invalid language '{language}'.", err=True)
        typer.echo(f"Available: {', '.join(DATASETS)}", err=True)
        raise typer.Exit(1)

    if model:
        if model not in EVAL_MODELS:
            typer.echo(
                f"ERROR: Model '{model}' not found in list_models.yaml.", err=True
            )
            typer.echo(f"Available: {', '.join(EVAL_MODELS.keys())}", err=True)
            raise typer.Exit(1)
        models_to_run = [model]
    else:
        models_to_run = list(EVAL_MODELS.keys())

    entry = DATASETS[language]
    typer.echo(f"\n=== Running Linguistic Knowledge Evaluation ({language}) ===")

    for current_model in models_to_run:
        try:
            run_benchmark(
                model_name=current_model,
                language_name=entry["language_name"],
                dataset_path=entry["path"],
                force=force,
            )
        except Exception as e:
            typer.echo(f"ERROR [{current_model}]: {e}", err=True)
            raise typer.Exit(1)


@app.command("judge-bench")
def judge_bench_cmd(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name whose results to judge",
    ),
    judge_model: str = typer.Option(
        "claude-sonnet-4-5-20250929",
        "--judge-model",
        "-j",
        help="Model to use as LLM judge",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-judge even if judged results exist",
    ),
    language: str = typer.Option(
        "luganda",
        "--language",
        "-l",
        help=f"Language: one of {', '.join(DATASETS)}",
    ),
    judge_fallback: bool = typer.Option(
        False,
        "--judge-fallback",
        help=(
            "Use LLM judge as fallback for MCQ/true-false rows where "
            "regex parsing failed (is_correct is null). Reads from "
            "answers_judged/ and writes to answers_judged_fallback/."
        ),
    ),
) -> None:
    """Judge Linguistic Knowledge evaluation results.

    Scores MC answers with regex and short_form/true_false with an LLM judge.
    If --model is not provided, judges all models that have step 1 results.
    """
    if language not in DATASETS:
        typer.echo(f"ERROR: Invalid language '{language}'.", err=True)
        typer.echo(f"Available: {', '.join(DATASETS)}", err=True)
        raise typer.Exit(1)
    if model:
        if model not in EVAL_MODELS:
            typer.echo(
                f"ERROR: Model '{model}' not found in list_models.yaml.", err=True
            )
            typer.echo(f"Available: {', '.join(EVAL_MODELS.keys())}", err=True)
            raise typer.Exit(1)
        models_to_judge = [model]
    else:
        models_to_judge = list(EVAL_MODELS.keys())

    entry = DATASETS[language]
    mode = "fallback" if judge_fallback else "standard"
    typer.echo(f"\n=== Judging Linguistic Knowledge Results ({language}, {mode}) ===")

    if judge_fallback:
        typer.echo(f"Inspecting {len(models_to_judge)} model(s) for fallback")

    summaries = []

    for current_model in models_to_judge:
        try:
            if judge_fallback:
                judged_df = judge_results_fallback(
                    model_name=current_model,
                    language_name=entry["language_name"],
                    dataset_path=entry["path"],
                    judge_model=judge_model,
                    force=force,
                )
            else:
                judged_df = judge_results(
                    model_name=current_model,
                    language_name=entry["language_name"],
                    dataset_path=entry["path"],
                    judge_model=judge_model,
                    force=force,
                )
            summary = compute_summary(judged_df, current_model)
            summaries.append(summary)

            typer.echo(f"\n{current_model}:")
            typer.echo(f"  Overall accuracy: {summary['accuracy']}%")
            typer.echo(f"  MC accuracy: {summary['accuracy_mc']}%")
            typer.echo(f"  Short-form accuracy: {summary['accuracy_short_form']}%")
            typer.echo(f"  True/false accuracy: {summary['accuracy_true_false']}%")

        except FileNotFoundError as e:
            typer.echo(f"SKIP [{current_model}]: {e}", err=True)
        except Exception as e:
            typer.echo(f"ERROR [{current_model}]: {e}", err=True)
            raise typer.Exit(1)

    if summaries:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        summary_df = pd.DataFrame(summaries)
        suffix = "_fallback" if judge_fallback else ""
        summary_name = f"summary_linguistic_knowledge_{language}{suffix}.csv"
        summary_path = results_dir / summary_name
        summary_df.to_csv(summary_path, index=False)
        typer.echo(f"\nSummary saved to {summary_path}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()  # pragma: no cover
