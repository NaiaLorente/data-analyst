"""Unit tests for PII column detection (no API key required)."""

import pandas as pd
from agent.pii import detect_pii_columns


def _findings_by_column(df: pd.DataFrame) -> dict:
    return {f["column"]: f["kind"] for f in detect_pii_columns(df)}


def test_detects_email_column():
    df = pd.DataFrame({"contact": [f"user{i}@example.com" for i in range(10)]})
    assert _findings_by_column(df)["contact"] == "email"


def test_detects_phone_column_with_separators():
    df = pd.DataFrame({"phone": ["555-123-4567", "(555) 234-5678", "+1 555 345 6789"] * 4})
    assert _findings_by_column(df)["phone"] == "phone"


def test_does_not_flag_plain_numeric_id_as_phone():
    df = pd.DataFrame({"order_id": [1000001 + i for i in range(10)]})
    assert "order_id" not in _findings_by_column(df)


def test_does_not_flag_iso_dates_as_phone():
    # Dash-separated and digit-count-in-range like a phone number, but a date.
    df = pd.DataFrame({"event_date": [f"2024-01-{i:02d}" for i in range(1, 11)]})
    assert "event_date" not in _findings_by_column(df)


def test_detects_ssn_column():
    df = pd.DataFrame({"ssn": [f"{100 + i:03d}-{10 + i:02d}-{1000 + i:04d}" for i in range(10)]})
    assert _findings_by_column(df)["ssn"] == "ssn"


def test_detects_ip_address_column():
    df = pd.DataFrame({"ip": ["192.168.1.1", "10.0.0.1", "8.8.8.8", "1.1.1.1"] * 3})
    assert _findings_by_column(df)["ip"] == "ip_address"


def test_does_not_flag_version_numbers_as_ip():
    # Shape-similar (dotted numbers) but octets exceed 255 -- must not false-positive.
    df = pd.DataFrame({"version": ["999.999.999.999"] * 10})
    assert "version" not in _findings_by_column(df)


def test_detects_credit_card_with_valid_luhn_numbers():
    valid_cards = ["4111111111111111", "4242424242424242", "5555555555554444", "4012888888881881"]
    df = pd.DataFrame({"card": (valid_cards * 3)[:10]})
    assert _findings_by_column(df)["card"] == "credit_card"


def test_does_not_flag_random_16_digit_ids_as_credit_card():
    # Sequential IDs of the same length/shape as a card number but failing Luhn.
    df = pd.DataFrame({"account_id": [f"{1000000000000000 + i}" for i in range(10)]})
    assert "account_id" not in _findings_by_column(df)


def test_does_not_flag_ordinary_numeric_or_categorical_columns():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "department": ["Eng", "HR", "Sales", "Eng", "HR"],
            "amount": [10.5, 20.1, 30.9, 40.2, 50.6],
        }
    )
    assert detect_pii_columns(df) == []


def test_handles_empty_dataframe():
    assert detect_pii_columns(pd.DataFrame()) == []


def test_handles_all_missing_column():
    df = pd.DataFrame({"maybe_email": [None, None, None]})
    assert detect_pii_columns(df) == []


def test_partial_match_below_threshold_not_flagged():
    # Only 2 of 10 values look like emails -- shouldn't flag the whole column.
    df = pd.DataFrame({"mixed": ["user@example.com", "user2@example.com"] + ["not an email"] * 8})
    assert "mixed" not in _findings_by_column(df)


def test_each_column_flagged_at_most_once():
    df = pd.DataFrame({"contact": [f"user{i}@example.com" for i in range(10)]})
    findings = detect_pii_columns(df)
    assert len([f for f in findings if f["column"] == "contact"]) == 1
