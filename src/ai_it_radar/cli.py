"""Typer CLI for the radar."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .feedback.profile_updater import ProfileUpdater
from .feedback.store import FeedbackStore
from .graph import run_cycle
from .harness.regression import GoldenSet, run_regression
from .memory.kb import KnowledgeBase
from .schemas import Feedback, FeedbackTag
from .settings import get_settings

app = typer.Typer(
    name="radar",
    help="AI IT Radar — multi-agent technology intelligence pipeline.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.callback()
def root(verbose: bool = typer.Option(False, "--verbose", "-v")):
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Print version."""
    typer.echo(__version__)


@app.command()
def init(
    seed: bool = typer.Option(True, help="Index the golden seed set into the KB."),
) -> None:
    """Bootstrap data dirs and (optionally) load the golden seed set into KB."""
    settings = get_settings()
    settings.ensure_dirs()
    KnowledgeBase()  # create schema
    console.print(f"[green]Initialized[/]: data_dir={settings.data_dir}")

    if seed:
        seed_path = Path("tests/golden/seeds.yaml")
        if not seed_path.exists():
            console.print("[yellow]Skipping seed: tests/golden/seeds.yaml not found[/]")
            return
        from .rag.embedder import get_embedder
        from .rag.indexer import index_candidate

        kb = KnowledgeBase()
        embedder = get_embedder()
        gs = GoldenSet.load(seed_path)
        for item in gs.items:
            cand = item.as_candidate()
            if kb.has_candidate(cand.uid):
                continue
            index_candidate(cand, kb, embedder)
        console.print(f"[green]Seeded {len(gs.items)} golden items into KB[/]")


@app.command()
def run(
    source: list[str] = typer.Option(
        None, "--source", help="Restrict to source(s): arxiv | github_trending | huggingface"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-evaluate even items already in KB (bypass Triage dedup + profile filter).",
    ),
) -> None:
    """Execute a single radar cycle (Scout -> Triage -> Analyst -> Evaluator -> Reporter)."""
    final = run_cycle(sources=source, force=force)
    rep = final.report
    if rep is None:
        console.print("[red]Cycle finished without a report.[/]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Cycle done[/] — strong:{len(rep.strong_recommend)} "
        f"watch:{len(rep.watch)} monitor:{len(rep.monitor)}"
    )
    console.print(f"reports written to: {get_settings().reports_dir}")


@app.command("report")
def show_report(
    period: str = typer.Option("7d", help="e.g. 7d, 30d"),
    rebuild: bool = typer.Option(
        False, "--rebuild",
        help="Also re-render reports/latest.{html,md} from existing KB (no source fetch, no LLM).",
    ),
) -> None:
    """Summarize stored scores in a period. Optionally re-render the HTML/MD report."""
    days = int(period.rstrip("dD"))
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    kb = KnowledgeBase()
    scores = kb.scores_in_period(start, end)
    table = Table(title=f"Scores in last {days}d", show_lines=False)
    for col in ("uid", "aggregate", "band", "needs_human"):
        table.add_column(col)
    for s in sorted(scores, key=lambda x: -x.aggregate)[:50]:
        table.add_row(s.candidate_uid, f"{s.aggregate:.2f}", s.band, str(s.needs_human))
    console.print(table)

    if rebuild:
        from .agents.reporter import reporter_node
        from .schemas import GraphState

        state = GraphState(cycle_id=f"rebuild-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}")
        reporter_node(state)
        console.print(
            f"[green]Rebuilt[/] {get_settings().reports_dir}/latest.html "
            f"(period last {days}d)"
        )


@app.command()
def feedback(
    uid: str = typer.Argument(..., help="Candidate uid (e.g. arxiv:2501.12345)"),
    tag: str = typer.Argument(..., help="adopt | watch | ignore"),
    note: str = typer.Option("", help="Optional comment."),
    user: str = typer.Option("cli", help="Who you are."),
) -> None:
    """Record a feedback row from the command line."""
    store = FeedbackStore()
    store.record(Feedback(candidate_uid=uid, tag=FeedbackTag(tag), note=note, user=user))
    console.print(f"[green]Recorded[/] {uid} -> {tag}")


@app.command("feedback-server")
def feedback_server(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
) -> None:
    """Run the local feedback FastAPI app (served reports + Adopt/Watch/Ignore endpoint)."""
    import uvicorn

    from .reporter.feedback_server import build_app

    app_obj = build_app()
    console.print(f"[green]Feedback server[/] http://{host}:{port}/latest")
    uvicorn.run(app_obj, host=host, port=port, log_level="info")


@app.command()
def schedule(
    scan_cron: str = typer.Option("0 9 * * MON", help="UTC cron for scan."),
    profile_cron: str = typer.Option("30 9 * * MON", help="UTC cron for profile auto-update."),
) -> None:
    """Run the APScheduler in the foreground."""
    from .scheduler import run_scheduler

    run_scheduler(scan_cron=scan_cron, profile_cron=profile_cron)


@app.command("update-profile")
def update_profile(
    lookback: int = typer.Option(30, help="Days of feedback to consider."),
) -> None:
    """One-off profile auto-update from recent adoptions."""
    res = ProfileUpdater().run(lookback_days=lookback)
    console.print(json.dumps(res, ensure_ascii=False, indent=2))


@app.command("ignore-stats")
def ignore_stats() -> None:
    """Inspect the IgnoreFilter — number of ignored items and centroid status."""
    from .feedback.ignore_filter import IgnoreFilter
    from .rag.embedder import get_embedder

    f = IgnoreFilter()
    embedder = get_embedder()
    active = f.is_active(embedder)
    console.print(f"ignored items in pool: [bold]{f.ignore_count}[/]")
    console.print(f"centroid active:       [bold]{active}[/]")
    if not active:
        console.print(
            "[yellow]No active ignore centroid.[/] Click 'Ignore' on items in the "
            "feedback UI, then re-run."
        )
        return
    if f.ignore_count <= 25:
        console.print("\nignored uids:")
        for u in sorted(f.ignored_uids):
            console.print(f"  - {u}")
    console.print(
        f"\nThreshold: {get_settings().triage_ignore_threshold}. "
        "Tune via RADAR_TRIAGE_IGNORE_THRESHOLD."
    )


@app.command()
def regression(
    seeds: Path = typer.Option(Path("tests/golden/seeds.yaml")),
) -> None:
    """Run the golden-set regression — re-evaluates each seed and compares to expected band/score."""
    from .agents.evaluator import _evaluate_one
    from .harness.critic import Critic
    from .harness.eval_spec import load_eval_specs
    from .llm import primary_llm
    from .memory.profile import LabProfile
    from .rag.embedder import get_embedder
    from .rag.retriever import Retriever

    if not seeds.exists():
        console.print(f"[red]No seeds file:[/] {seeds}")
        raise typer.Exit(code=2)

    gs = GoldenSet.load(seeds)
    kb = KnowledgeBase()
    specs = load_eval_specs()
    embedder = get_embedder()
    retriever = Retriever(kb, embedder)
    profile = LabProfile()
    llm = primary_llm()
    critic = Critic()

    def _eval(c):
        s = _evaluate_one(c, specs, retriever, kb, profile, llm, critic, cycle_id="regression")
        from .agents.evaluator import _band_of
        s.band = _band_of(s.aggregate, get_settings())
        return s

    rep = run_regression(gs, _eval)
    console.print(json.dumps({
        "ok": rep.ok,
        "drift_count": rep.drift_count,
        "passed": len(rep.passed_items),
        "failed_items": rep.failed_items,
    }, ensure_ascii=False, indent=2))
    raise typer.Exit(code=0 if rep.ok else 1)


if __name__ == "__main__":
    app()
