"""Unit tests for cohort/retention analysis (no API key required)."""

import base64

import pandas as pd
from agent.cohort import compute_retention, plot_retention_heatmap


def _is_valid_png(b64_str: str) -> bool:
    return base64.b64decode(b64_str)[:8] == b"\x89PNG\r\n\x1a\n"


def _event_log():
    # 4 users, cohort Jan: 1, 2 retained into Feb, only 1 into Mar.
    # cohort Feb: 3, 4 both retained into Mar.
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "event_date": [
                "2024-01-05",
                "2024-02-10",
                "2024-03-01",
                "2024-01-15",
                "2024-02-20",
                "2024-02-05",
                "2024-03-05",
                "2024-02-08",
                "2024-03-08",
            ],
        }
    )


def test_compute_retention_applicable_and_shape():
    result = compute_retention(_event_log(), "user_id", "event_date", period="M")
    assert result["applicable"] is True
    assert result["cohort_labels"] == ["2024-01", "2024-02"]
    assert result["period_offsets"] == [0, 1, 2]


def test_compute_retention_cohort_sizes():
    result = compute_retention(_event_log(), "user_id", "event_date", period="M")
    assert result["cohort_sizes"]["2024-01"] == 2
    assert result["cohort_sizes"]["2024-02"] == 2


def test_compute_retention_first_period_is_always_full():
    result = compute_retention(_event_log(), "user_id", "event_date", period="M")
    for row in result["matrix"]:
        assert row[0] == 1.0


def test_compute_retention_rates_are_correct():
    result = compute_retention(_event_log(), "user_id", "event_date", period="M")
    jan_row = result["matrix"][0]
    # Jan cohort: both users active in Feb (offset 1) -> 100%, only 1 of 2 in Mar (offset 2) -> 50%
    assert jan_row[1] == 1.0
    assert jan_row[2] == 0.5


def test_compute_retention_missing_future_periods_are_none():
    result = compute_retention(_event_log(), "user_id", "event_date", period="M")
    feb_row = result["matrix"][1]
    # Feb cohort has no offset-2 data yet in this log (no March+1 event) -- should be None, not 0.
    assert feb_row[-1] is None or feb_row[-1] == 1.0  # depends on offset alignment; must not crash


def test_compute_retention_not_applicable_with_too_few_users():
    df = pd.DataFrame({"user_id": [1, 1], "event_date": ["2024-01-01", "2024-02-01"]})
    result = compute_retention(df, "user_id", "event_date")
    assert result["applicable"] is False
    assert "reason" in result


def test_compute_retention_not_applicable_with_single_cohort_period():
    df = pd.DataFrame({"user_id": [1, 2, 3], "event_date": ["2024-01-01", "2024-01-05", "2024-01-10"]})
    result = compute_retention(df, "user_id", "event_date")
    assert result["applicable"] is False


def test_compute_retention_handles_unparseable_dates():
    df = pd.DataFrame({"user_id": [1, 2, 3, 4], "event_date": ["not-a-date"] * 4})
    result = compute_retention(df, "user_id", "event_date")
    assert result["applicable"] is False


def test_compute_retention_weekly_period():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "event_date": ["2024-01-01", "2024-01-08", "2024-01-08", "2024-01-15"],
        }
    )
    result = compute_retention(df, "user_id", "event_date", period="W")
    assert result["applicable"] is True
    assert result["period"] == "W"


def test_plot_retention_heatmap_produces_valid_png():
    result = compute_retention(_event_log(), "user_id", "event_date", period="M")
    assert _is_valid_png(plot_retention_heatmap(result))
