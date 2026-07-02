"""PII detection: flag columns that likely contain personally identifiable
information (emails, phone numbers, SSNs, credit card numbers, IP addresses)
*before* they can be sent to a third-party AI provider via chat or narration.
Everything else in this app that touches a chosen AI provider only ever sends
tool results and schema info — but a tool like filter_rows can return raw row
previews, so a PII column really can leak into a chat request if nobody's
watching for it.

Detection is a pure-pandas/regex heuristic — no ML, no network call — checking
what fraction of a column's non-null values match a known PII shape, the same
"looks like X" pattern already used elsewhere in this app (agent.cleaning).
It's deliberately conservative and format-based: free-text name detection is
out of scope (there's no reliable way to do that without a name database or an
LLM call, which would defeat the purpose of catching PII before it reaches one).
"""

import re

import pandas as pd

MATCH_THRESHOLD = 0.8

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_PHONE_SEPARATOR_RE = re.compile(r"[-\s()+]")  # no "." -- avoids colliding with dotted IPs/version numbers
_NON_DIGIT_RE = re.compile(r"\D")
_IP_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
_CREDIT_CARD_SHAPE_RE = re.compile(r"^\d[\d\s-]{11,22}\d$")

PII_LABELS = {
    "email": "Email address",
    "phone": "Phone number",
    "ssn": "Social Security Number",
    "credit_card": "Credit card number",
    "ip_address": "IP address",
}


def _match_rate(values: pd.Series, pattern: re.Pattern) -> float:
    return float(values.str.match(pattern).mean())


def _looks_like_date(values: pd.Series) -> bool:
    converted = pd.to_datetime(values, errors="coerce", format="mixed")
    return bool(converted.notna().mean() >= MATCH_THRESHOLD)


def _looks_like_phone(values: pd.Series) -> bool:
    # A dash-separated date ("2024-01-05") has both a separator and a digit
    # count in the phone-length range, so it would otherwise false-positive —
    # check dates first and bail out before the phone heuristic below.
    if _looks_like_date(values):
        return False
    # Require a separator (dash/space/parens/leading +) to avoid matching plain
    # numeric ID columns, which are far more common than unformatted phone digits.
    has_separator = values.str.contains(_PHONE_SEPARATOR_RE)
    digit_count = values.str.replace(_NON_DIGIT_RE, "", regex=True).str.len()
    plausible_length = digit_count.between(7, 15)
    return bool((has_separator & plausible_length).mean() >= MATCH_THRESHOLD)


def _looks_like_ip(values: pd.Series) -> bool:
    shape_match = values.str.match(_IP_RE)
    if shape_match.mean() < MATCH_THRESHOLD:
        return False
    candidates = values[shape_match]

    def _octets_valid(value: str) -> bool:
        m = _IP_RE.match(value)
        return m is not None and all(0 <= int(g) <= 255 for g in m.groups())

    return bool(candidates.apply(_octets_valid).mean() >= MATCH_THRESHOLD)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _looks_like_credit_card(values: pd.Series) -> bool:
    shape_match = values.str.match(_CREDIT_CARD_SHAPE_RE)
    if shape_match.mean() < MATCH_THRESHOLD:
        return False
    digits_only = values[shape_match].str.replace(_NON_DIGIT_RE, "", regex=True)
    valid_length = digits_only[digits_only.str.len().between(13, 19)]
    if valid_length.empty:
        return False
    # Luhn checksum: a random long digit string (a generic ID) passes only ~10%
    # of the time, so this meaningfully separates real card numbers from IDs.
    return bool(valid_length.apply(_luhn_valid).mean() >= MATCH_THRESHOLD)


_PII_CHECKS = [
    ("email", lambda v: _match_rate(v, _EMAIL_RE) >= MATCH_THRESHOLD),
    ("ssn", lambda v: _match_rate(v, _SSN_RE) >= MATCH_THRESHOLD),
    ("credit_card", _looks_like_credit_card),
    ("ip_address", _looks_like_ip),
    ("phone", _looks_like_phone),
]


def detect_pii_columns(df: pd.DataFrame) -> list[dict]:
    """Returns [{"column": ..., "kind": ..., "label": ...}, ...] for columns
    whose non-null values mostly match a known PII shape. Each column is
    reported with at most one kind (the first matching check, in the fixed
    order above), so a column isn't double-flagged."""
    findings = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        values = non_null.astype(str).str.strip()
        for kind, check in _PII_CHECKS:
            try:
                matched = check(values)
            except (TypeError, ValueError):
                continue
            if matched:
                findings.append({"column": col, "kind": kind, "label": PII_LABELS[kind]})
                break
    return findings
