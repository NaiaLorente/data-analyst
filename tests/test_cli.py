"""Unit tests for the CLI drift-check entry point (no API key required)."""

import json
from unittest.mock import patch

import pandas as pd
import pytest
from agent.cli import EXIT_ERROR, EXIT_OK, EXIT_SIGNIFICANT_CHANGE, main


@pytest.fixture
def snapshots(tmp_path):
    before = pd.DataFrame(
        {
            "region": ["West"] * 10 + ["East"] * 10,
            "revenue": [1000] * 10 + [500] * 10,
        }
    )
    after = pd.DataFrame(
        {
            "region": ["West"] * 10 + ["East"] * 10,
            "revenue": [100] * 10 + [600] * 10,
        }
    )
    before_path = tmp_path / "before.csv"
    after_path = tmp_path / "after.csv"
    before.to_csv(before_path, index=False)
    after.to_csv(after_path, index=False)
    return str(before_path), str(after_path)


def test_check_prints_report_and_exits_ok(snapshots, capsys):
    before_path, after_path = snapshots
    code = main(["check", before_path, after_path, "--metric", "revenue", "--segment", "region"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "revenue" in out
    assert "Significance:" in out


def test_check_auto_picks_metric_and_segment(snapshots, capsys):
    before_path, after_path = snapshots
    code = main(["check", before_path, after_path])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "revenue" in out


def test_check_json_format_is_valid_json(snapshots, capsys):
    before_path, after_path = snapshots
    code = main(["check", before_path, after_path, "--metric", "revenue", "--segment", "region", "--format", "json"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["metric_significance"]["metric"] == "revenue"


def test_fail_on_significant_exits_2_when_significant(snapshots, capsys):
    before_path, after_path = snapshots
    code = main(
        ["check", before_path, after_path, "--metric", "revenue", "--segment", "region", "--fail-on-significant"]
    )
    capsys.readouterr()
    assert code == EXIT_SIGNIFICANT_CHANGE


def test_fail_on_significant_exits_0_when_not_significant(tmp_path, capsys):
    # Same values in both snapshots — nothing changed, nothing significant.
    df = pd.DataFrame({"region": ["West", "East"] * 10, "revenue": [100, 200] * 10})
    before_path, after_path = tmp_path / "a.csv", tmp_path / "b.csv"
    df.to_csv(before_path, index=False)
    df.to_csv(after_path, index=False)
    code = main(
        [
            "check",
            str(before_path),
            str(after_path),
            "--metric",
            "revenue",
            "--segment",
            "region",
            "--fail-on-significant",
        ]
    )
    capsys.readouterr()
    assert code == EXIT_OK


def test_missing_file_exits_error(capsys):
    code = main(["check", "/nonexistent/a.csv", "/nonexistent/b.csv"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "Error reading input files" in err


def test_narrate_without_api_key_exits_error(snapshots, capsys):
    before_path, after_path = snapshots
    code = main(
        ["check", before_path, after_path, "--metric", "revenue", "--segment", "region", "--narrate", "--provider", "groq"]
    )
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "requires an API key" in err


def test_narrate_reads_api_key_from_env_var(snapshots, capsys, monkeypatch):
    before_path, after_path = snapshots
    monkeypatch.setenv("AI_DATA_ANALYST_API_KEY", "fake-key")
    with patch("agent.analyst.narrate_drift", return_value="It went down."):
        code = main(
            [
                "check",
                before_path,
                after_path,
                "--metric",
                "revenue",
                "--segment",
                "region",
                "--narrate",
                "--provider",
                "groq",
            ]
        )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "AI narrative:" in out
    assert "It went down." in out


def test_narrate_with_explicit_api_key_flag(snapshots, capsys):
    before_path, after_path = snapshots
    with patch("agent.analyst.narrate_drift", return_value="Summary here.") as mock_narrate:
        code = main(
            [
                "check",
                before_path,
                after_path,
                "--metric",
                "revenue",
                "--segment",
                "region",
                "--narrate",
                "--provider",
                "groq",
                "--api-key",
                "explicit-key",
            ]
        )
    capsys.readouterr()
    assert code == EXIT_OK
    assert mock_narrate.call_args.kwargs["api_key"] == "explicit-key"


def test_narrate_local_provider_needs_no_key(snapshots, capsys):
    before_path, after_path = snapshots
    with patch("agent.analyst.narrate_drift", return_value="Local summary."):
        code = main(
            [
                "check",
                before_path,
                after_path,
                "--metric",
                "revenue",
                "--segment",
                "region",
                "--narrate",
                "--provider",
                "ollama",
            ]
        )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Local summary." in out
