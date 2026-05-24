"""Local FastAPI server that receives Adopt / Watch / Ignore clicks from the HTML report."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..feedback.store import FeedbackStore
from ..schemas import Feedback, FeedbackTag
from ..settings import get_settings

log = logging.getLogger(__name__)


class FeedbackPayload(BaseModel):
    candidate_uid: str
    tag: str
    note: str = ""
    user: str = "decision-maker"


def build_app() -> FastAPI:
    app = FastAPI(title="AI IT Radar — Feedback")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         # local-only; loosen since reports may open as file://
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = FeedbackStore()
    settings = get_settings()

    @app.get("/")
    def index() -> JSONResponse:
        return JSONResponse({
            "service": "ai-it-radar feedback",
            "endpoints": ["/feedback (POST)", "/feedback/{uid} (GET)", "/latest"],
        })

    @app.post("/feedback")
    def post_feedback(payload: FeedbackPayload):
        try:
            tag = FeedbackTag(payload.tag)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid tag: {payload.tag}")
        fb = Feedback(
            candidate_uid=payload.candidate_uid,
            tag=tag,
            note=payload.note,
            user=payload.user,
        )
        store.record(fb)
        log.info("feedback recorded: %s -> %s", payload.candidate_uid, tag.value)
        return {"ok": True}

    @app.get("/feedback/{uid}")
    def list_feedback(uid: str):
        return [fb.model_dump() for fb in store.for_candidate(uid)]

    # Serve the latest report (so users can browse from the same origin as the API).
    reports_dir: Path = settings.reports_dir
    if reports_dir.exists():
        app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

        @app.get("/latest")
        def latest():
            target = reports_dir / "latest.html"
            if not target.exists():
                raise HTTPException(status_code=404, detail="no report yet — run `radar run` first")
            return FileResponse(str(target))

    return app
