from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .schemas import DataQualityIssue, Severity
from .storage import Store


class OperationalScheduler:
    """Local readiness checks; it never submits brokerage orders."""

    def __init__(self, store: Store):
        self.store = store
        self.scheduler = AsyncIOScheduler(timezone="Asia/Singapore")

    def _freshness_check(self, cadence: str) -> None:
        if self.store.latest_available_date("prices") is None:
            self.store.record_issue(
                DataQualityIssue(
                    severity=Severity.WARNING,
                    dataset="prices",
                    code=f"{cadence.upper()}_READINESS",
                    message=f"{cadence.title()} workflow is waiting for a price import; no order was created.",
                )
            )

    def start(self) -> None:
        self.scheduler.add_job(
            self._freshness_check,
            CronTrigger(hour=18, minute=0),
            args=["daily"],
            id="daily-risk-readiness",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._freshness_check,
            CronTrigger(day_of_week="fri", hour=18, minute=15),
            args=["weekly"],
            id="weekly-adjustment-readiness",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._freshness_check,
            CronTrigger(day=1, hour=18, minute=30),
            args=["monthly"],
            id="monthly-ciq-readiness",
            replace_existing=True,
        )
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
