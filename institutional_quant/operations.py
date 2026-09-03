"""Deterministic daily, weekly and monthly operating workflows.

The service deliberately keeps the cadence boundary explicit: daily monitoring
never invokes the LLM committee, weekly changes are turnover constrained, and
the monthly workflow is the only path that runs the full research graph.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

from .agents import EvidencePacketBuilder, ResearchGraph
from .alpaca import AlpacaPaperClient
from .config import Settings
from .factors import FactorEngine
from .portfolio import PortfolioOptimizer
from .schemas import OperationResult, PaperTarget
from .storage import Store


def _cutoff(store: Store, requested: date | None) -> date:
    if requested is not None:
        return requested
    candidates = [
        store.latest_available_date(table)
        for table in ("fundamentals", "estimates", "market_returns")
    ]
    candidates = [value for value in candidates if value is not None]
    if not candidates:
        latest = store.price_coverage()[1]
        if latest is None:
            raise ValueError("No certified point-in-time data is available")
        return latest
    return min(max(candidates), date.today())


def _returns_overlay(store: Store, as_of_date: date) -> pd.DataFrame:
    frame = store.query_df(
        """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY company_id ORDER BY as_of_date DESC, effective_at DESC, ingested_at DESC
            ) AS rn
            FROM market_returns
            WHERE as_of_date <= ? AND effective_at <= ?
        )
        SELECT company_id, ticker, return_1d, return_1w, return_1m
        FROM ranked WHERE rn = 1
        """,
        [as_of_date, datetime.combine(as_of_date, time.max)],
    )
    if frame.empty:
        return pd.DataFrame(columns=["company_id", "short_horizon_score"])
    values = frame[["return_1d", "return_1w", "return_1m"]].apply(
        pd.to_numeric, errors="coerce"
    )
    # CIQ exports percentages.  Convert to fractional returns before averaging.
    values = values / 100.0
    raw = values.mean(axis=1, skipna=True)
    temp = frame[["company_id"]].copy()
    temp["raw_short_horizon"] = raw
    sectors = store.query_df(
        """SELECT company_id, sector FROM (
            SELECT company_id, sector, ROW_NUMBER() OVER (
                PARTITION BY company_id ORDER BY effective_at DESC
            ) AS rn FROM instruments WHERE effective_at <= ?
        ) WHERE rn = 1""",
        [datetime.combine(as_of_date, time.max)],
    )
    temp = temp.merge(sectors, on="company_id", how="left")
    temp["short_horizon_score"] = temp.groupby("sector", dropna=False)["raw_short_horizon"].transform(
        lambda series: (series - series.mean()) / (series.std(ddof=0) or 1.0)
    )
    return temp[["company_id", "short_horizon_score"]]


def _rank_snapshot(store: Store, as_of_date: date) -> pd.DataFrame:
    snapshot = FactorEngine(store).snapshot(as_of_date)
    if snapshot.empty:
        raise ValueError("Factor snapshot is empty; import point-in-time S&P 500 data first")
    snapshot["factor_rank"] = snapshot["factor_score"].rank(pct=True).fillna(0.0)
    # FactorEngine produces the deterministic factor snapshot.  Walk-forward
    # ML scores are persisted separately, so join only the latest score that
    # was already available at this cutoff (never a future observation).
    snapshot["ml_score"] = 0.0
    previous = store.query_df(
            """
            WITH latest AS (
              SELECT MAX(as_of_date) AS as_of_date
              FROM factor_observations
              WHERE as_of_date <= ?
            )
            SELECT company_id, ml_score
            FROM factor_observations
            WHERE as_of_date = (SELECT as_of_date FROM latest)
            """,
            [as_of_date],
    )
    if not previous.empty:
        snapshot = snapshot.drop(columns="ml_score").merge(previous, on="company_id", how="left")
        snapshot["ml_score"] = snapshot["ml_score"].fillna(0.0)
    snapshot["ml_score"] = pd.to_numeric(snapshot["ml_score"], errors="coerce").fillna(0.0)
    snapshot["ensemble_score"] = 0.5 * snapshot["factor_rank"] + 0.5 * snapshot["ml_score"]
    overlay = _returns_overlay(store, as_of_date)
    snapshot = snapshot.merge(overlay, on="company_id", how="left")
    snapshot["short_horizon_score"] = snapshot["short_horizon_score"].fillna(0.0)
    # Narrative inputs are deliberately capped at 10% of the rank score.
    snapshot["operational_score"] = snapshot["ensemble_score"] + 0.10 * snapshot["short_horizon_score"].clip(-1.0, 1.0)
    return snapshot.sort_values(["operational_score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def _risk_snapshot(store: Store, as_of_date: date) -> dict[str, Any]:
    portfolio = store.current_portfolio()
    prices = store.load_prices(as_of_date - timedelta(days=40), as_of_date, tickers=["SPY"])
    latest_price_date = None if prices.empty else str(prices["price_date"].max())
    return {
        "portfolio_exists": portfolio is not None,
        "holdings": len(portfolio.positions) if portfolio else 0,
        "latest_price_date": latest_price_date,
        "alerts": [] if latest_price_date else ["No recent price bars are available"],
    }


async def run_daily(
    store: Store,
    settings: Settings,
    as_of_date: date | None = None,
    client: AlpacaPaperClient | None = None,
) -> OperationResult:
    cutoff = _cutoff(store, as_of_date)
    ranked = _rank_snapshot(store, cutoff)
    paper_sync = await client.synchronize() if client is not None else None
    return OperationResult(
        operation_id=f"daily-{cutoff.isoformat()}",
        cadence="daily",
        as_of_date=cutoff,
        status="completed",
        message="Daily ranking and risk monitoring completed; no automatic rebalance was submitted.",
        result={
            "risk": _risk_snapshot(store, cutoff),
            "candidates": ranked.head(25)[["ticker", "sector", "operational_score", "short_horizon_score"]].to_dict(orient="records"),
            "agent_committee": False,
            "automatic_rebalance": False,
            "paper_sync": paper_sync,
        },
    )


async def run_weekly(store: Store, settings: Settings, as_of_date: date | None = None) -> OperationResult:
    cutoff = _cutoff(store, as_of_date)
    ranked = _rank_snapshot(store, cutoff)
    current = store.current_portfolio()
    if current is None:
        return OperationResult(
            operation_id=f"weekly-{cutoff.isoformat()}", cadence="weekly", as_of_date=cutoff,
            status="held", message="No existing portfolio; weekly adjustment held until monthly deployment.",
            result={"turnover_cap": 0.05, "portfolio": None},
        )
    prices = store.load_prices(cutoff - timedelta(days=430), cutoff)
    history = prices.pivot_table(index="price_date", columns="ticker", values="adjusted_close", aggfunc="last").pct_change().tail(252)
    recommendation = PortfolioOptimizer().optimize(
        ranked,
        history,
        cutoff,
        score_column="operational_score",
        current_weights={position.ticker: position.weight for position in current.positions},
        benchmark_sector_weights=ranked["sector"].value_counts(normalize=True).to_dict(),
        cadence="weekly",
        turnover_limit=0.05,
    )
    # The optimizer's hard monthly constraint is replaced with the weekly cap above.
    recommendation.cadence = "weekly"
    store.save_portfolio(recommendation)
    return OperationResult(
        operation_id=f"weekly-{cutoff.isoformat()}", cadence="weekly", as_of_date=cutoff,
        status="completed" if recommendation.status != "held" else "held",
        message="Weekly risk refresh and constrained adjustment completed; paper previews only.",
        result={"turnover_cap": 0.05, "portfolio": recommendation.model_dump(mode="json")},
    )


async def run_monthly(store: Store, settings: Settings, as_of_date: date | None = None) -> OperationResult:
    cutoff = _cutoff(store, as_of_date)
    ranked = _rank_snapshot(store, cutoff)
    selected = ranked.head(10)
    builder = EvidencePacketBuilder(store)
    graph = ResearchGraph(store, settings)
    semaphore = asyncio.Semaphore(2)
    async def run_case(row: Any) -> dict[str, Any]:
        packet = builder.build(str(row.company_id), cutoff, float(getattr(row, "ml_score", 0.0)))
        async with semaphore:
            decision = await graph.run(packet, with_debate=True)
        store.save_consensus(decision)
        return {"packet": packet.model_dump(mode="json"), "decision": decision.model_dump(mode="json")}

    # Cases are independent evidence packets.  Running them concurrently keeps
    # a ten-name monthly committee practical while each case still has its own
    # immutable cache key and provenance record.
    cases = list(await asyncio.gather(*(run_case(row) for row in selected.itertuples(index=False))))
    decisions = [case["decision"] for case in cases]
    from .api import _build_portfolio  # local import avoids a module cycle
    recommendation = await asyncio.to_thread(_build_portfolio, store, cutoff, decisions)
    recommendation.cadence = "monthly"
    store.save_portfolio(recommendation)
    return OperationResult(
        operation_id=f"monthly-{cutoff.isoformat()}", cadence="monthly", as_of_date=cutoff,
        status="completed" if recommendation.status != "held" else "held",
        message="Monthly factor, ML, multi-agent debate and portfolio deployment completed.",
        result={"selected_cases": len(cases), "cases": cases, "portfolio": recommendation.model_dump(mode="json")},
    )


async def prepare_one_share_order(store: Store, settings: Settings, client: AlpacaPaperClient, as_of_date: date | None = None) -> dict[str, Any]:
    cutoff = _cutoff(store, as_of_date)
    ranked = _rank_snapshot(store, cutoff)
    candidate = ranked.iloc[0]
    previews = await client.preview([PaperTarget(symbol=str(candidate.ticker), target_weight=0.0, quantity=1.0)])
    return {"candidate": {"ticker": str(candidate.ticker), "sector": str(candidate.sector), "score": float(candidate.operational_score)}, "previews": [item.model_dump(mode="json") for item in previews]}


async def run_full_cycle(store: Store, settings: Settings, client: AlpacaPaperClient, *, submit_paper_order: bool = False, as_of_date: date | None = None) -> OperationResult:
    cutoff = _cutoff(store, as_of_date)
    daily = await run_daily(store, settings, cutoff, client=client)
    weekly = await run_weekly(store, settings, cutoff)
    monthly = await run_monthly(store, settings, cutoff)
    order = await prepare_one_share_order(store, settings, client, cutoff)
    # A full-cycle run may prepare a preview, but approval stays a separate
    # checkpoint.  This prevents an asynchronous job from placing an order
    # merely because a request body contained a boolean flag.
    submitted: list[dict[str, Any]] = []
    fill_sync = await client.synchronize()
    return OperationResult(
        operation_id=f"full-cycle-{cutoff.isoformat()}", cadence="full-cycle", as_of_date=cutoff,
        status="awaiting_approval" if order["previews"] else "completed",
        message="Daily → weekly → monthly → paper preview completed; explicit approval is required before submission.",
        result={"daily": daily.model_dump(mode="json"), "weekly": weekly.model_dump(mode="json"), "monthly": monthly.model_dump(mode="json"), "paper_order": order, "submitted_orders": submitted, "fill_sync": fill_sync, "approval_required": bool(order["previews"]), "submit_requested_but_deferred": bool(submit_paper_order)},
    )
