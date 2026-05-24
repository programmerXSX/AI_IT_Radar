"""Smoke tests — these do NOT require any LLM credentials.

They verify that the project imports cleanly, the schema validates, the SQLite KB
schema is created, and the harness/EvalSpec config parses. End-to-end LLM tests
live elsewhere because they cost money and require network access.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="airadar-test-"))
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp))
    # Reset the cached settings between tests.
    from ai_it_radar import settings as s
    s._settings = None
    yield
    # tmp left for inspection on failure


def test_imports():
    import ai_it_radar  # noqa
    from ai_it_radar import cli, graph, schemas, settings  # noqa
    from ai_it_radar.agents import (  # noqa
        analyst,
        evaluator,
        reporter,
        scout,
        triage,
    )
    from ai_it_radar.harness import critic, eval_spec, pairwise, regression  # noqa
    from ai_it_radar.memory import kb, profile, short_term  # noqa
    from ai_it_radar.rag import embedder, indexer, retriever, reranker  # noqa
    from ai_it_radar.sources import all_sources

    keys = set(all_sources().keys())
    assert keys == {"arxiv", "github_trending", "huggingface"}


def test_schemas_roundtrip():
    from ai_it_radar.schemas import (
        Candidate,
        CandidateKind,
        DimensionScore,
        Score,
        SourceKind,
    )

    c = Candidate(
        uid="arxiv:2501.99999",
        source=SourceKind.ARXIV,
        kind=CandidateKind.PAPER,
        title="Test paper",
        url="https://arxiv.org/abs/2501.99999",
        summary="x",
        content="y",
    )
    assert c.uid == "arxiv:2501.99999"
    assert c.source == SourceKind.ARXIV

    s = Score(
        candidate_uid=c.uid,
        dimensions=[DimensionScore(dimension_id="novelty", score=3, confidence=0.5,
                                   rationale="r", quote="q")],
        aggregate=3.0,
    )
    assert s.dimensions[0].score == 3


def test_kb_initializes():
    from ai_it_radar.memory.kb import KnowledgeBase

    kb = KnowledgeBase()
    assert kb is not None
    # Tables should exist.
    with kb._conn() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {"candidates", "analyses", "scores", "feedback", "eval_traces", "cycles"} <= names


def test_eval_specs_parse():
    # Point config_dir at the in-repo config (we do not isolate it).
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    from ai_it_radar.harness.eval_spec import load_eval_specs

    specs = load_eval_specs()
    ids = {s.id for s in specs}
    assert {"novelty", "maturity", "fit", "reproduction_cost", "risk"} <= ids
    for s in specs:
        # Rubric MUST cover 0..5
        assert set(s.rubric.keys()) >= set(range(0, 6))


def test_golden_seeds_load():
    repo_root = Path(__file__).resolve().parents[1]
    from ai_it_radar.harness.regression import GoldenSet

    gs = GoldenSet.load(repo_root / "tests" / "golden" / "seeds.yaml")
    assert len(gs.items) >= 20
    for item in gs.items:
        assert item.expected_band in {"strong_recommend", "watch", "monitor"}


def test_lab_profile_loads():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    from ai_it_radar.memory.profile import LabProfile

    p = LabProfile()
    anchors = p.anchors()
    assert len(anchors) >= 4
    assert all(a.is_anchor for a in anchors)
    assert "blockchain" in p.exclude_keywords()


def test_graph_builds():
    """The LangGraph topology compiles end-to-end (no nodes invoked)."""
    if "langgraph" not in sys.modules:
        try:
            import langgraph  # noqa
        except ImportError:
            pytest.skip("langgraph not installed")
    from ai_it_radar.graph import _build_graph_uncompiled

    g = _build_graph_uncompiled()
    nodes = set(g.nodes)
    assert {"scout", "triage", "analyst", "evaluator", "reporter"} <= nodes
