"""Research-only portfolio rebalancing advice.

This module joins a *user-requested* broker snapshot with the certified local
factor/ML snapshot.  It deliberately produces a staged research plan only:
there is no Alpaca client here and therefore no route from this output to an
order submission.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .operations import _cutoff, _rank_snapshot
from .portfolio import PortfolioOptimizer
from .storage import Store

MAX_NAME_WEIGHT = 0.05
MONTHLY_TURNOVER_BUDGET = 0.20


def _number(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _format_metric(value: Any, *, percent: bool = False) -> str | None:
    numeric = _number(value)
    if numeric is None:
        return None
    return f"{numeric * 100:.1f}%" if percent else f"{numeric:.2f}"


def _latest_reference_prices(store: Store, as_of_date: date) -> tuple[dict[str, float], str | None]:
    prices = store.load_prices(as_of_date - timedelta(days=430), as_of_date)
    if prices.empty:
        return {}, None
    prices = prices.dropna(subset=["adjusted_close"]).sort_values("price_date")
    latest = prices.drop_duplicates("ticker", keep="last")
    return (
        {
            str(row.ticker): float(row.adjusted_close)
            for row in latest.itertuples(index=False)
            if float(row.adjusted_close) > 0
        },
        str(prices["price_date"].max()),
    )


def _target_portfolio(
    ranked: pd.DataFrame, store: Store, as_of_date: date
) -> tuple[pd.DataFrame, list[str]]:
    """Construct a fresh, constrained target without treating broker positions as targets."""
    candidates = ranked.dropna(subset=["operational_score"]).copy()
    history = store.load_prices(as_of_date - timedelta(days=430), as_of_date)
    returns = (
        history.pivot_table(
            index="price_date", columns="ticker", values="adjusted_close", aggfunc="last"
        )
        .pct_change()
        .tail(252)
    )
    benchmark = candidates["sector"].fillna("Unknown").value_counts(normalize=True).to_dict()
    try:
        recommendation = PortfolioOptimizer().optimize(
            candidates,
            returns,
            as_of_date,
            score_column="operational_score",
            benchmark_sector_weights=benchmark,
            min_positions=20,
            max_positions=30,
            cadence="monthly",
        )
        target = pd.DataFrame([position.model_dump() for position in recommendation.positions])
        if len(target) >= 20 and not target.empty:
            return target, list(recommendation.warnings)
    except (ValueError, ArithmeticError) as exc:
        return _equal_weight_target(candidates), [f"Optimizer fallback: {exc}"]
    return _equal_weight_target(candidates), [
        "Optimizer fallback: a diversified 20-name, 5% maximum-weight target was used."
    ]


def _equal_weight_target(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = candidates.sort_values(
        ["operational_score", "ticker"], ascending=[False, True]
    ).head(20)
    return (
        selected[["company_id", "ticker", "sector", "operational_score"]]
        .rename(columns={"operational_score": "score"})
        .assign(weight=MAX_NAME_WEIGHT)
    )


def _evidence(row: pd.Series, *, model_available: bool) -> list[dict[str, str]]:
    values: list[tuple[str, Any, bool]] = [
        ("Composite factor score", row.get("factor_score"), False),
        ("ML score", row.get("ml_score"), False),
        ("Value factor", row.get("factor_value"), False),
        ("Quality factor", row.get("factor_quality"), False),
        ("Growth factor", row.get("factor_growth"), False),
        ("Revision factor", row.get("factor_revisions"), False),
        ("P/E", row.get("price_to_earnings"), False),
        ("P/B", row.get("price_to_book"), False),
        ("TEV / EBITDA", row.get("tev_ebitda"), False),
        ("ROIC", row.get("roic"), True),
        ("Gross margin", row.get("gross_margin"), True),
        ("Revenue growth", row.get("revenue_growth"), True),
        ("EPS revision (1M)", row.get("eps_revision_1m"), True),
        ("Annualized volatility", row.get("volatility_252d"), True),
    ]
    output: list[dict[str, str]] = []
    for label, value, percent in values:
        if label == "ML score" and not model_available:
            continue
        formatted = _format_metric(value, percent=percent)
        if formatted is not None:
            output.append({"label": label, "value": formatted})
    return output


def _allocate_notional(actions: list[dict[str, Any]], equity: float) -> None:
    """Fit desired buys and sells into the explicit 20% monthly one-way budget."""
    budget = equity * MONTHLY_TURNOVER_BUDGET
    sells = [action for action in actions if action["desired_delta_value"] < -1e-6]
    buys = [action for action in actions if action["desired_delta_value"] > 1e-6]
    sells.sort(key=lambda action: abs(action["desired_delta_value"]), reverse=True)
    buys.sort(key=lambda action: (action["operational_score"], action["rank"]), reverse=True)

    remaining = budget
    for action in sells:
        amount = min(abs(action["desired_delta_value"]), remaining)
        action["proposed_delta_value"] = -amount
        remaining -= amount
    sale_proceeds = budget - remaining

    # Reinvest only staged sale proceeds.  This keeps the plan cash-neutral even
    # when the brokerage account has margin or a cash balance that is not supplied.
    remaining = sale_proceeds
    for action in buys:
        amount = min(action["desired_delta_value"], remaining)
        action["proposed_delta_value"] = amount
        remaining -= amount
    for action in actions:
        action.setdefault("proposed_delta_value", 0.0)


def build_rebalance_advice(
    ranked: pd.DataFrame,
    target: pd.DataFrame,
    holdings: list[dict[str, float | str]],
    equity: float,
    reference_prices: dict[str, float],
    *,
    as_of_date: date,
    reference_price_date: str | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build an explainable, staged advice object from already-computed inputs."""
    if equity <= 0:
        raise ValueError("Account equity must be greater than zero")
    warnings = list(warnings or [])
    ranked = ranked.copy()
    ranked["ticker"] = ranked["ticker"].astype(str).str.upper()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    target = target.copy()
    target["ticker"] = target["ticker"].astype(str).str.upper()
    target_weights = target.set_index("ticker")["weight"].astype(float).to_dict()
    target_sectors = target.set_index("ticker")["sector"].fillna("Unknown").astype(str).to_dict()

    holdings_by_ticker = {str(item["symbol"]).upper(): item for item in holdings}
    score_rows = ranked.drop_duplicates("ticker").set_index("ticker", drop=False)
    actions: list[dict[str, Any]] = []
    for ticker in sorted(set(holdings_by_ticker) | set(target_weights)):
        holding = holdings_by_ticker.get(ticker, {})
        row = score_rows.loc[ticker] if ticker in score_rows.index else pd.Series(dtype=object)
        current_value = float(holding.get("market_value", 0.0) or 0.0)
        current_quantity = float(holding.get("qty", 0.0) or 0.0)
        current_weight = current_value / equity
        target_weight = float(target_weights.get(ticker, 0.0))
        target_value = equity * target_weight
        desired_delta_value = target_value - current_value
        reference_price = reference_prices.get(ticker) or _number(holding.get("current_price"))
        if reference_price is None or reference_price <= 0:
            warnings.append(
                f"{ticker}: no usable local reference price; share quantity is unavailable."
            )
            reference_price = 0.0
        operational_score = _number(row.get("operational_score")) or 0.0
        rank = int(_number(row.get("rank")) or 9999)
        sector = str(row.get("sector") or target_sectors.get(ticker) or "Unclassified")
        actions.append(
            {
                "ticker": ticker,
                "sector": sector,
                "rank": rank,
                "operational_score": operational_score,
                "factor_score": _number(row.get("factor_score")),
                "ml_score": _number(row.get("ml_score")),
                "short_horizon_score": _number(row.get("short_horizon_score")),
                "current_quantity": current_quantity,
                "current_value": current_value,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "reference_price": reference_price,
                "desired_delta_value": desired_delta_value,
                "evidence": _evidence(
                    row, model_available=bool(np.abs(ranked["ml_score"]).sum() > 1e-9)
                )
                if not row.empty
                else [],
            }
        )

    _allocate_notional(actions, equity)
    for action in actions:
        delta = action["proposed_delta_value"]
        if abs(delta) < max(1.0, equity * 0.0001):
            continue
        if delta < 0:
            action["action"] = "Reduce"
            action["reason"] = (
                f"Current weight {action['current_weight']:.1%} is above the constrained target "
                f"of {action['target_weight']:.1%}; this is the next staged risk reduction."
            )
        elif action["current_quantity"] > 0:
            action["action"] = "Add"
            action["reason"] = (
                f"Current weight {action['current_weight']:.1%} is below the model target "
                f"of {action['target_weight']:.1%}; the score and sector constraints support a staged addition."
            )
        else:
            action["action"] = "Initiate"
            action["reason"] = (
                f"Rank #{action['rank']} in the certified sector-neutral snapshot; the name fills a "
                f"{action['target_weight']:.1%} constrained target allocation."
            )
        action["proposed_shares"] = (
            abs(delta) / action["reference_price"] if action["reference_price"] > 0 else None
        )
        action["proposed_notional"] = abs(delta)
        action["target_quantity"] = (
            equity * action["target_weight"] / action["reference_price"]
            if action["reference_price"] > 0
            else None
        )
    actions = [action for action in actions if "action" in action]
    actions.sort(
        key=lambda action: (
            0 if action["action"] == "Reduce" else 1,
            -action["proposed_notional"],
            action["rank"],
        )
    )

    current_sectors: dict[str, float] = {}
    for ticker, holding in holdings_by_ticker.items():
        sector = (
            str(score_rows.loc[ticker].get("sector"))
            if ticker in score_rows.index
            else "Unclassified"
        )
        current_sectors[sector] = (
            current_sectors.get(sector, 0.0)
            + float(holding.get("market_value", 0.0) or 0.0) / equity
        )
    target_sector_weights = target.groupby("sector", dropna=False)["weight"].sum().to_dict()
    sectors = [
        {
            "sector": str(sector),
            "current_weight": float(current_sectors.get(sector, 0.0)),
            "target_weight": float(target_sector_weights.get(sector, 0.0)),
            "difference": float(target_sector_weights.get(sector, 0.0))
            - float(current_sectors.get(sector, 0.0)),
        }
        for sector in sorted(set(current_sectors) | set(target_sector_weights))
    ]
    model_available = bool(np.abs(ranked["ml_score"]).sum() > 1e-9)
    if not model_available:
        warnings.append(
            "No non-zero walk-forward ML scores were available at this cutoff; ranking is factor-led."
        )
    warnings.extend(
        [
            "Research proposal only. It is not an order, quote, or individualized investment instruction.",
            "Reference prices are the latest locally stored adjusted closes and may differ from executable market prices.",
            "The staged plan limits one-way monthly turnover to 20% of account equity and caps every target name at 5%.",
        ]
    )
    return {
        "as_of_date": as_of_date.isoformat(),
        "reference_price_date": reference_price_date,
        "account_equity": equity,
        "model_signal": "factor + ML + short-horizon overlay"
        if model_available
        else "factor + short-horizon overlay",
        "monthly_turnover_budget": MONTHLY_TURNOVER_BUDGET,
        "max_name_weight": MAX_NAME_WEIGHT,
        "target_holdings": int(len(target)),
        "current_sector_weights": sectors,
        "actions": actions,
        "warnings": list(dict.fromkeys(warnings)),
    }


def generate_rebalance_advice(
    store: Store, holdings: list[dict[str, float | str]], equity: float
) -> dict[str, Any]:
    """Build advice from a broker snapshot supplied by an explicit UI action."""
    cutoff = _cutoff(store, None)
    ranked = _rank_snapshot(store, cutoff)
    target, warnings = _target_portfolio(ranked, store, cutoff)
    reference_prices, reference_price_date = _latest_reference_prices(store, cutoff)
    return build_rebalance_advice(
        ranked,
        target,
        holdings,
        equity,
        reference_prices,
        as_of_date=cutoff,
        reference_price_date=reference_price_date,
        warnings=warnings,
    )
