"""Command-line adapters for scoring runs and replotting saved reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from phonics_benchmark.reporting import DEFAULT_OUTPUT_ROOT, plot_report, report_on_run


def main():
    parser = argparse.ArgumentParser(
        description="Score runs and generate F1/F2/BA reports."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--run-id",
        action="append",
        help="Run ID under --output-root; repeat for multiple runs.",
    )
    inputs.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="Run directory; repeat for multiple runs.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Separate report destination; multiple runs use subdirectories.",
    )
    parser.add_argument("--inspect-sample-size", type=int)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-items", type=int, default=1)
    parser.add_argument(
        "--min-coverage", type=float, help="Strict coverage threshold, e.g. 0.9."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional models.csv, providers.csv and display_overrides.json directory.",
    )
    parser.add_argument("--hardware-tiers", action="store_true")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs = args.run_dir or [args.output_root / run_id for run_id in args.run_id]
    runs = [path.resolve() for path in runs]
    if len(set(runs)) != len(runs) or (
        args.output_dir and len({p.name for p in runs}) != len(runs)
    ):
        parser.error(
            "Run directories must be distinct, with distinct names "
            "for a shared output directory"
        )
    if args.hardware_tiers and (not args.metadata or not args.output_dir):
        parser.error("--hardware-tiers requires --metadata and --output-dir")
    if args.top_n < 1:
        parser.error("--top-n must be positive")
    reports = []
    for run in runs:
        destination = args.output_dir
        if destination and len(runs) > 1:
            destination = destination / run.name
        report = report_on_run(
            run.name,
            run.parent,
            inspect_sample_size=args.inspect_sample_size,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            seed=args.seed,
            output_dir=destination,
            metadata=args.metadata,
            min_items=args.min_items,
            min_coverage=args.min_coverage,
            make_plots=not args.no_plots,
        )
        reports.append(report)
        print(f"Report: {report}")
    if args.output_dir and len(reports) > 1:
        from phonics_benchmark.reporting import export_collection

        export_collection(reports, args.output_dir)
    if args.hardware_tiers:
        from phonics_benchmark.hardware import report_hardware

        report_hardware(
            reports,
            args.metadata / "models.csv",
            args.output_dir / "hardware_tiers",
            top_n=args.top_n,
            make_plots=not args.no_plots,
        )


def plot_main():
    parser = argparse.ArgumentParser(
        description="Redraw saved reports without scoring or bootstrapping."
    )
    parser.add_argument("--report-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--hardware-metadata",
        type=Path,
        help="Optional models.csv for hardware comparisons.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    if args.hardware_metadata and not args.output_dir:
        parser.error("--hardware-metadata requires --output-dir")
    if args.top_n < 1:
        parser.error("--top-n must be positive")
    if (
        len(args.report_dir) > 1
        and args.output_dir
        and len({p.resolve().name for p in args.report_dir}) != len(args.report_dir)
    ):
        parser.error(
            "Report directories need distinct names for a shared output directory"
        )
    for report in args.report_dir:
        destination = args.output_dir
        if destination and len(args.report_dir) > 1:
            destination = destination / report.name
        plot_report(report, output_dir=destination)
    if args.hardware_metadata:
        from phonics_benchmark.hardware import report_hardware

        report_hardware(
            args.report_dir,
            args.hardware_metadata,
            args.output_dir / "hardware_tiers",
            top_n=args.top_n,
        )
