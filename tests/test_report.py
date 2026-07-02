"""Unit tests for the shareable report builder (no API key required)."""

from agent.report import PROJECT_URL, build_html_report, build_markdown_report


def _messages():
    return [
        {"role": "user", "content": "Summarize this dataset"},
        {"role": "assistant", "content": "It has 100 rows.", "charts": []},
    ]


def test_build_html_report_contains_content():
    html = build_html_report("data.csv", "<ul><li>ok</li></ul>", _messages())
    assert "data.csv" in html
    assert "Summarize this dataset" in html
    assert "<ul><li>ok</li></ul>" in html


def test_build_markdown_report_contains_content():
    md = build_markdown_report("data.csv", "- ok", _messages())
    assert "data.csv" in md
    assert "Summarize this dataset" in md
    assert "- ok" in md


def test_html_report_escapes_user_content():
    messages = [{"role": "user", "content": "<script>alert(1)</script>"}]
    html = build_html_report("data.csv", "", messages)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_html_report_has_attribution_footer():
    html = build_html_report("data.csv", "<ul><li>ok</li></ul>", _messages())
    assert PROJECT_URL in html
    assert "<footer>" in html


def test_markdown_report_has_attribution_footer():
    md = build_markdown_report("data.csv", "- ok", _messages())
    assert PROJECT_URL in md
