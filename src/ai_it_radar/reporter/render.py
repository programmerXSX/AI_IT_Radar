"""Jinja2 renderer for radar reports — produces both HTML (with feedback buttons) and Markdown."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..schemas import Report

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_report(report: Report, out_dir: Path) -> dict[str, Path]:
    """Render report to {out_dir}/report-YYYYMMDD-HHMM.{html,md}. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%d-%H%M")
    env = _env()

    html_tpl = env.get_template("report.html.j2")
    md_tpl = env.get_template("report.md.j2")

    html_path = out_dir / f"report-{stamp}.html"
    md_path = out_dir / f"report-{stamp}.md"

    html_path.write_text(html_tpl.render(report=report, now=datetime.utcnow()), encoding="utf-8")
    md_path.write_text(md_tpl.render(report=report, now=datetime.utcnow()), encoding="utf-8")

    # Also keep a stable "latest" pointer.
    (out_dir / "latest.html").write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {"html": html_path, "md": md_path}
