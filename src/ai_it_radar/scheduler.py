"""APScheduler wiring — runs `run_cycle` and `ProfileUpdater.run` on a schedule."""
from __future__ import annotations

import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

from .feedback.profile_updater import ProfileUpdater
from .graph import run_cycle

log = logging.getLogger(__name__)


def run_scheduler(*, scan_cron: str = "0 9 * * MON", profile_cron: str = "30 9 * * MON") -> None:
    """Start the APScheduler in the foreground.

    Defaults: weekly scan Monday 09:00, profile auto-update Monday 09:30.
    Use Ctrl-C to stop.
    """
    sched = BackgroundScheduler(timezone="UTC")

    sched.add_job(_safe_run_cycle, "cron", **_parse_cron(scan_cron), id="scan_cycle",
                  max_instances=1, coalesce=True)
    sched.add_job(_safe_profile_update, "cron", **_parse_cron(profile_cron),
                  id="profile_update", max_instances=1, coalesce=True)

    sched.start()
    log.info("scheduler started: scan=%s profile_update=%s", scan_cron, profile_cron)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        sched.shutdown(wait=False)
        log.info("scheduler stopped")


def _parse_cron(expr: str) -> dict:
    minute, hour, day, month, day_of_week = expr.split()
    return {
        "minute": minute, "hour": hour, "day": day,
        "month": month, "day_of_week": day_of_week,
    }


def _safe_run_cycle() -> None:
    try:
        run_cycle()
    except Exception as e:
        log.exception("scheduled run_cycle failed: %s", e)


def _safe_profile_update() -> None:
    try:
        ProfileUpdater().run()
    except Exception as e:
        log.exception("scheduled profile update failed: %s", e)
