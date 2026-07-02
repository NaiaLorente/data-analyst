"""Command-line entry point for What Changed — the same pandas/scipy drift and
significance analysis the Streamlit app runs, without a browser or a server.
Built for cron jobs and CI pipelines: point it at this week's export vs. last
week's, and let --fail-on-significant turn a real change into a non-zero exit
code your pipeline can act on.

Examples:
    # Human-readable report, auto-picking a metric and segment
    python -m agent.cli check before.csv after.csv

    # Pin the metric/segment explicitly
    python -m agent.cli check before.csv after.csv --metric revenue --segment region

    # In a cron job or CI step: exit 2 if the change is statistically significant
    python -m agent.cli check before.csv after.csv --metric revenue --segment region \\
        --fail-on-significant

    # Also get a plain-English AI narrative (BYOK — same rules as the app)
    python -m agent.cli check before.csv after.csv --metric revenue --segment region \\
        --narrate --provider groq --api-key "$GROQ_API_KEY"

No AI key is required unless --narrate is passed — the drift report, the
significance tests, and the exit code are all computed locally with zero AI cost.
"""

import argparse
import json
import os
import sys

from agent.drift import (
    drift_report_to_markdown,
    generate_drift_report,
    suggest_metric_column,
    suggest_segment_column,
)
from agent.loaders import load_dataframe
from agent.providers import DEFAULT_PROVIDER, PROVIDERS

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SIGNIFICANT_CHANGE = 2

API_KEY_ENV_VAR = "AI_DATA_ANALYST_API_KEY"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.cli",
        description="Run What Changed's root-cause + significance analysis from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Compare two dataset snapshots and report what changed.")
    check.add_argument("file_a", help="The 'before' snapshot (CSV/Excel/JSON/Parquet).")
    check.add_argument("file_b", help="The 'after' snapshot (CSV/Excel/JSON/Parquet).")
    check.add_argument("--metric", help="Numeric column to explain. Auto-picked if omitted.")
    check.add_argument("--segment", help="Categorical column to break the change down by. Auto-picked if omitted.")
    check.add_argument("--agg", choices=["sum", "mean"], default="sum", help="Aggregation for --metric (default: sum).")
    check.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text).")
    check.add_argument(
        "--fail-on-significant",
        action="store_true",
        help=(
            "Exit with status 2 if the metric's or segment's change is statistically "
            "significant (Welch's t-test / chi-square) — for cron/CI alerting."
        ),
    )
    check.add_argument("--narrate", action="store_true", help="Also generate a plain-English AI narration (BYOK).")
    check.add_argument(
        "--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDERS), help="AI provider for --narrate."
    )
    check.add_argument("--model", help="Model name for --narrate. Defaults to the provider's default model.")
    check.add_argument(
        "--api-key",
        help=f"API key for --narrate. Falls back to the {API_KEY_ENV_VAR} environment variable.",
    )
    check.add_argument("--base-url", help="Override the provider's API base URL (e.g. a custom Ollama host).")
    return parser


def _has_significant_change(report: dict) -> bool:
    metric_sig = report.get("metric_significance") or {}
    segment_sig = report.get("segment_significance") or {}
    return bool(metric_sig.get("significant")) or bool(segment_sig.get("significant"))


def _run_check(args: argparse.Namespace) -> int:
    try:
        df_a = load_dataframe(args.file_a, args.file_a)
        df_b = load_dataframe(args.file_b, args.file_b)
    except Exception as exc:
        print(f"Error reading input files: {exc}", file=sys.stderr)
        return EXIT_ERROR

    metric = args.metric or suggest_metric_column(df_a, df_b)
    segment = args.segment or suggest_segment_column(df_a, df_b)

    try:
        report = generate_drift_report(df_a, df_b, metric=metric, segment=segment, agg=args.agg)
    except Exception as exc:
        print(f"Error computing drift report: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.narrate:
        from agent.analyst import narrate_drift

        provider = PROVIDERS[args.provider]
        api_key = args.api_key or os.environ.get(API_KEY_ENV_VAR, "")
        if provider.requires_key and not api_key:
            print(
                f"--narrate requires an API key for {provider.label}. Pass --api-key or set {API_KEY_ENV_VAR}.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        model = args.model or provider.default_model
        try:
            report["narrative"] = narrate_drift(
                report, provider_id=args.provider, model=model, api_key=api_key, base_url=args.base_url
            )
        except Exception as exc:
            print(f"Error generating narration: {exc}", file=sys.stderr)
            return EXIT_ERROR

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(drift_report_to_markdown(report))
        if "narrative" in report:
            print("\nAI narrative:")
            print(report["narrative"])

    if args.fail_on_significant and _has_significant_change(report):
        return EXIT_SIGNIFICANT_CHANGE
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run_check(args)


if __name__ == "__main__":
    sys.exit(main())
