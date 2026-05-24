"""Manual smoke test for the configured LLM / Critic / Embedding endpoints.

Run with:
    uv run python tests/manual/check_keys.py

This script is NOT collected by pytest (lives outside the testpaths). Use it
when changing `.env` to verify all three providers respond before wasting a
real radar cycle on a misconfigured key.

Each step costs at most a few cents (2 LLM calls + 1 embedding).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow direct invocation from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rich.console import Console

console = Console()


def check_embedding() -> None:
    console.rule("[bold]Embedding (DashScope)")
    from ai_it_radar.rag.embedder import get_embedder

    embedder = get_embedder()
    vec = embedder.embed_query("LangGraph multi-agent radar")
    docs = embedder.embed_documents(
        ["agent orchestration", "vector retrieval", "evaluation harness"]
    )
    console.print(f"[green]OK[/]  query dim = {len(vec)}, first 4 = {vec[:4]}")
    console.print(f"[green]OK[/]  batch returned {len(docs)} vectors of dim {len(docs[0])}")


def check_primary_llm() -> None:
    console.rule("[bold]Primary LLM (DeepSeek)")
    from ai_it_radar.llm import llm_json, primary_llm

    prompt = (
        "Reply with STRICT JSON and nothing else. "
        'The JSON must have exactly these keys: ping (string "pong"), '
        "answer (integer 42)."
    )
    result = llm_json(primary_llm(), prompt)
    assert result.get("ping") == "pong", f"unexpected ping: {result}"
    assert int(result.get("answer", 0)) == 42, f"unexpected answer: {result}"
    console.print(f"[green]OK[/]  parsed = {result}")


def check_critic_llm() -> None:
    console.rule("[bold]Critic LLM (Qwen via DashScope)")
    from ai_it_radar.llm import critic_llm, llm_json

    prompt = (
        "Reply with STRICT JSON and nothing else. "
        'JSON keys: agree (boolean true), reason (string "ok").'
    )
    result = llm_json(critic_llm(), prompt)
    assert result.get("agree") is True, f"unexpected agree: {result}"
    console.print(f"[green]OK[/]  parsed = {result}")


def main() -> int:
    failures: list[str] = []
    for name, fn in (
        ("embedding", check_embedding),
        ("primary_llm", check_primary_llm),
        ("critic_llm", check_critic_llm),
    ):
        try:
            fn()
        except Exception as e:
            failures.append(name)
            console.print(f"[red]FAIL[/] {name}: {type(e).__name__}: {e}")

    console.rule()
    if failures:
        console.print(f"[red]Failed:[/] {', '.join(failures)}")
        return 1
    console.print("[bold green]All three providers OK.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
