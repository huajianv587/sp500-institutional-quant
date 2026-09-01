from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .agents import EvidencePacketBuilder, ResearchGraph
from .config import Settings
from .storage import Store


class AgentStudyRunner:
    """Runs the frozen recent-month debate/no-debate ablation on ranked cases."""

    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings

    async def run(
        self, months: int = 24, progress: Callable[[float, str], None] | None = None
    ) -> dict[str, Any]:
        available = self.store.query_df(
            """
            SELECT DISTINCT as_of_date FROM factor_observations
            ORDER BY as_of_date DESC LIMIT ?
            """,
            [months],
        )
        dates = sorted(pd.to_datetime(available["as_of_date"]).dt.date.tolist())
        if len(dates) < months:
            raise ValueError(
                f"Run the base walk-forward backtest first; {months} factor months required, found {len(dates)}"
            )
        graph = ResearchGraph(self.store, self.settings)
        builder = EvidencePacketBuilder(self.store)
        completed = 0
        total = months * 10 * 2
        counts = {"with_debate": 0, "without_debate": 0}
        for as_of_date in dates:
            frame = self.store.query_df(
                """
                SELECT company_id, ticker, factor_score, ml_score, ensemble_score
                FROM factor_observations WHERE as_of_date = ?
                """,
                [as_of_date],
            )
            frame["ml_score"] = frame["ml_score"].fillna(0.0)
            frame["ensemble_score"] = frame["ensemble_score"].fillna(frame["factor_score"])
            frame["disagreement"] = abs(
                frame["factor_score"].rank(pct=True) - frame["ml_score"].rank(pct=True)
            )
            selected = (
                pd.concat([frame.nlargest(5, "ensemble_score"), frame.nlargest(5, "disagreement")])
                .drop_duplicates("company_id")
                .head(10)
            )
            for row in selected.itertuples():
                packet = builder.build(
                    str(row.company_id), as_of_date, float(row.ml_score), float(row.ensemble_score)
                )
                for variant, with_debate in (("without_debate", False), ("with_debate", True)):
                    decision = await graph.run(
                        packet,
                        with_debate=with_debate,
                        cache_namespace=f"agent-study:{variant}",
                    )
                    self.store.save_agent_study_decision(variant, decision)
                    counts[variant] += 1
                    completed += 1
                    if progress:
                        progress(completed / total, f"{as_of_date}: {completed}/{total} decisions")
        return {"months": months, "company_months": sum(counts.values()), "variants": counts}
