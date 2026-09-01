from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .schemas import JobRecord, JobStatus
from .storage import Store


class JobManager:
    def __init__(self, store: Store):
        self.store = store
        self.tasks: set[asyncio.Task] = set()

    def submit(self, kind: str, work: Callable[[JobRecord], Awaitable[str | None]]) -> JobRecord:
        job = JobRecord(kind=kind)
        self.store.upsert_job(job)

        async def runner() -> None:
            job.status = JobStatus.RUNNING
            job.progress = 0.05
            job.message = "Running"
            self.store.upsert_job(job)
            try:
                job.result_ref = await work(job)
                job.status = JobStatus.SUCCEEDED
                job.progress = 1.0
                job.message = "Completed"
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.message = "Failed"
                job.error = f"{type(exc).__name__}: {exc}"
            self.store.upsert_job(job)

        task = asyncio.create_task(runner())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return job
