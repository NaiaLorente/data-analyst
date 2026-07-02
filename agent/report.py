"""Turn a chat session into a shareable, self-contained report.

Produces both a Markdown version (readable anywhere) and an HTML version
(charts embedded as base64 images, no external files needed to share it).
"""

from datetime import datetime, timezone

from agent.insights import escape_html

PROJECT_URL = "https://github.com/NaiaLorente/ai-data-analyst"
ATTRIBUTION = f"Made with AI Data Analyst — free, open source, bring your own AI ({PROJECT_URL})"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def build_markdown_report(
    filename: str,
    insights_md: str,
    messages: list[dict],
    drift_md: str | None = None,
    drift_narrative: str | None = None,
) -> str:
    lines = [
        f"# AI Data Analyst Report — {filename}",
        f"_Generated {_timestamp()}_",
        "",
        "## Auto-Insights",
        insights_md,
    ]
    if drift_md:
        lines += ["", "## What Changed", drift_md]
        if drift_narrative:
            lines += ["", f"> {drift_narrative}"]
    lines += ["", "## Conversation"]
    for msg in messages:
        role = "**You**" if msg["role"] == "user" else "**AI Analyst**"
        lines.append(f"\n{role}: {msg['content']}")
        for i in range(len(msg.get("charts", []))):
            lines.append(f"\n_(chart {i + 1} — see the HTML report for the image)_")
    lines += ["", "---", f"_{ATTRIBUTION}_"]
    return "\n".join(lines)


def build_html_report(
    filename: str,
    insights_html: str,
    messages: list[dict],
    drift_html: str | None = None,
    drift_narrative: str | None = None,
) -> str:
    drift_section = ""
    if drift_html:
        narrative_html = (
            f'<p><em>{escape_html(drift_narrative)}</em></p>' if drift_narrative else ""
        )
        drift_section = f"""<h2>What Changed</h2>
<div class="insights">{drift_html}{narrative_html}</div>
"""
    parts = [
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Data Analyst Report — {escape_html(filename)}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 860px;
       margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.6; }}
h1 {{ font-size: 1.6rem; }}
.msg {{ padding: 14px 18px; border-radius: 10px; margin: 10px 0; }}
.user {{ background: #eef4ff; }}
.assistant {{ background: #f6f6f6; }}
img {{ max-width: 100%; border-radius: 8px; margin-top: 10px; }}
.insights {{ background: #fffbea; padding: 16px 20px; border-radius: 10px; }}
footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid #e1e0d9; font-size: 0.85rem; color: #898781; }}
footer a {{ color: #2a78d6; }}
</style></head><body>
<h1>📊 AI Data Analyst Report</h1>
<p><em>Dataset: {escape_html(filename)} · Generated {_timestamp()}</em></p>
<h2>Auto-Insights</h2>
<div class="insights">{insights_html}</div>
{drift_section}<h2>Conversation</h2>
"""
    ]
    for msg in messages:
        css = "user" if msg["role"] == "user" else "assistant"
        who = "You" if msg["role"] == "user" else "AI Analyst"
        content_html = escape_html(msg["content"]).replace("\n", "<br>")
        parts.append(f'<div class="msg {css}"><strong>{who}:</strong><br>{content_html}')
        for chart_b64 in msg.get("charts", []):
            parts.append(f'<br><img src="data:image/png;base64,{chart_b64}">')
        parts.append("</div>")
    parts.append(f'<footer>Made with <a href="{PROJECT_URL}">AI Data Analyst</a> — free, open source, bring your own AI.</footer>')
    parts.append("</body></html>")
    return "\n".join(parts)
