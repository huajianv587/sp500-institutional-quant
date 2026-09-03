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
SECTOR_ACTIVE_LIMIT = 0.08
DEFENSIVE_FACTOR_RANK_PERCENTILE = 0.75
DEFENSIVE_ML_SCORE = 0.25


def _number(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _format_metric(value: Any, *, percent: bool = False) -> str | None:
    numeric = _number(value)
    if numeric is None:
        return None
    return f"{numeric * 100:.1f}%" if percent else f"{numeric:.2f}"


EVIDENCE_METADATA: dict[str, dict[str, str]] = {
    "Composite factor score": {
        "calculation": "Arithmetic mean of the available factor-family scores. Eligibility requires at least 4 of 6 families and 2 institutional families.",
        "source": "Certified point-in-time fundamentals, estimates, and adjusted-price history.",
        "benchmark": "Each component is winsorized and sector-neutralized against the eligible S&P 500 universe.",
    },
    "ML score": {
        "calculation": "Average percentile rank from walk-forward ElasticNet and gradient-boosting predictions of next-month excess return.",
        "source": "Previously available factor observations and realized historical returns only.",
        "benchmark": "0.00 to 1.00 cross-sectional percentile within the certified S&P 500 snapshot.",
    },
    "Value factor": {
        "calculation": "Mean of sector-neutral z-scores for earnings yield, free-cash-flow yield, and EBITDA-to-enterprise-value.",
        "source": "Certified point-in-time fundamental and valuation inputs.",
        "benchmark": "Same-sector S&P 500 peers after 2.5% / 97.5% winsorization.",
    },
    "Quality factor": {
        "calculation": "Mean of sector-neutral z-scores for ROIC, gross profitability, accruals (inverted), and net debt / EBITDA (inverted).",
        "source": "Certified point-in-time fundamental inputs.",
        "benchmark": "Same-sector S&P 500 peers after 2.5% / 97.5% winsorization.",
    },
    "Growth factor": {
        "calculation": "Mean of sector-neutral z-scores for revenue growth, EPS growth, and margin change.",
        "source": "Certified point-in-time fundamental inputs.",
        "benchmark": "Same-sector S&P 500 peers after 2.5% / 97.5% winsorization.",
    },
    "Revision factor": {
        "calculation": "Mean of sector-neutral z-scores for 1-month and 3-month EPS revisions plus estimate surprise.",
        "source": "Certified point-in-time consensus-estimate history.",
        "benchmark": "Same-sector S&P 500 peers after 2.5% / 97.5% winsorization.",
    },
    "P/E": {
        "calculation": "Reported price-to-earnings multiple; lower is not automatically better without growth and quality context.",
        "source": "Certified point-in-time valuation input.",
        "benchmark": "Interpret alongside sector peers and the value-factor composite.",
    },
    "P/B": {
        "calculation": "Reported price-to-book multiple; its usefulness varies materially by sector.",
        "source": "Certified point-in-time valuation input.",
        "benchmark": "Interpret alongside sector peers and the value-factor composite.",
    },
    "TEV / EBITDA": {
        "calculation": "Reported enterprise-value-to-EBITDA multiple; lower is not automatically better without quality and growth context.",
        "source": "Certified point-in-time valuation input.",
        "benchmark": "Interpret alongside sector peers and the value-factor composite.",
    },
    "ROIC": {
        "calculation": "Reported return on invested capital.",
        "source": "Certified point-in-time fundamental input.",
        "benchmark": "Interpret alongside same-sector S&P 500 peers and the quality-factor composite.",
    },
    "Gross margin": {
        "calculation": "Reported gross profit divided by revenue.",
        "source": "Certified point-in-time fundamental input.",
        "benchmark": "Interpret alongside same-sector S&P 500 peers and the quality-factor composite.",
    },
    "Revenue growth": {
        "calculation": "Reported period-over-period revenue growth.",
        "source": "Certified point-in-time fundamental input.",
        "benchmark": "Interpret alongside same-sector S&P 500 peers and the growth-factor composite.",
    },
    "EPS revision (1M)": {
        "calculation": "Point-in-time change in consensus EPS estimate over the prior month.",
        "source": "Certified consensus-estimate history.",
        "benchmark": "Interpret alongside same-sector S&P 500 peers and the revision-factor composite.",
    },
    "Annualized volatility": {
        "calculation": "Standard deviation of daily returns over up to 252 sessions, annualized by square-root-of-252.",
        "source": "Certified adjusted-price history.",
        "benchmark": "Interpret alongside same-sector S&P 500 peers and the low-risk factor.",
    },
}

def _evidence_confidence(row: pd.Series, label: str, model_available: bool) -> tuple[str, str]:
    family_count = int(_number(row.get("factor_family_count")) or 0)
    institutional_count = int(_number(row.get("institutional_factor_family_count")) or 0)
    coverage = f"Coverage: {family_count}/6 factor families, {institutional_count}/4 institutional families."
    if label == "ML score":
        status = "Moderate" if model_available else "Unavailable"
        return status, "Walk-forward model output is a rank, not a probability of positive return."
    if label.endswith("factor") or label == "Composite factor score":
        status = "High" if family_count >= 5 and institutional_count >= 3 else "Moderate"
        return status, f"{coverage} This grades input completeness, not investment outcome certainty."
    return "Moderate", "A single observed metric is evidence, not a forecast or a stand-alone trade signal."


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


def _evidence(
    row: pd.Series, *, model_available: bool, as_of_date: date
) -> list[dict[str, str]]:
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
            metadata = EVIDENCE_METADATA[label]
            confidence, confidence_note = _evidence_confidence(row, label, model_available)
            output.append(
                {
                    "label": label,
                    "value": formatted,
                    "calculation": metadata["calculation"],
                    "source": metadata["source"],
                    "benchmark": metadata["benchmark"],
                    "confidence": confidence,
                    "confidence_note": confidence_note,
                    "available_at": as_of_date.isoformat(),
                    "provenance_hash": str(row.get("source_snapshot_hash") or "Not recorded"),
                }
            )
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
    model_available = bool(np.abs(ranked["ml_score"]).sum() > 1e-9)
    target_sector_weights = target.groupby("sector", dropna=False)["weight"].sum().to_dict()
    current_sectors: dict[str, float] = {}
    holding_sector_counts: dict[str, int] = {}
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
        holding_sector_counts[sector] = holding_sector_counts.get(sector, 0) + 1
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
        rank_value = _number(row.get("rank"))
        rank = int(rank_value) if rank_value is not None else None
        sector = str(row.get("sector") or target_sectors.get(ticker) or "Unclassified")
        actions.append(
            {
                "ticker": ticker,
                "sector": sector,
                "rank": rank,
                "operational_score": operational_score,
                "factor_score": _number(row.get("factor_score")),
                "factor_family_count": int(_number(row.get("factor_family_count")) or 0),
                "institutional_factor_family_count": int(
                    _number(row.get("institutional_factor_family_count")) or 0
                ),
                "ml_score": _number(row.get("ml_score")),
                "short_horizon_score": _number(row.get("short_horizon_score")),
                "current_quantity": current_quantity,
                "current_value": current_value,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "risk_limit_weight": current_weight,
                "reference_price": reference_price,
                "desired_delta_value": desired_delta_value,
                "evidence": _evidence(row, model_available=model_available, as_of_date=as_of_date)
                if not row.empty
                else [],
                "risk_details": [],
            }
        )

    # The model target is a source of *new idea* selection.  Existing holdings
    # are trimmed only when their own portfolio-risk gate fires.  Falling out of
    # a fresh target is not, by itself, a bearish conclusion.
    for action in actions:
        if action["current_quantity"] <= 0:
            action["bucket"] = "purchase"
            continue
        sector = action["sector"]
        flags: list[str] = []
        risk_limit = action["current_weight"]
        if action["current_weight"] > MAX_NAME_WEIGHT + 1e-8:
            flag = f"Position concentration {action['current_weight']:.1%} exceeds the {MAX_NAME_WEIGHT:.0%} name limit"
            flags.append(flag)
            action["risk_details"].append(
                {
                    "label": "Position concentration",
                    "value": f"{action['current_weight']:.1%}",
                    "calculation": "Position market value divided by synchronized paper-account equity.",
                    "source": "User-requested Alpaca paper position and account snapshot.",
                    "benchmark": f"Portfolio policy maximum: {MAX_NAME_WEIGHT:.0%} per name.",
                    "confidence": "High",
                    "confidence_note": "This is an arithmetic exposure check, not a return forecast.",
                    "available_at": as_of_date.isoformat(),
                }
            )
            risk_limit = min(risk_limit, MAX_NAME_WEIGHT)
        sector_overage = current_sectors.get(sector, 0.0) - (
            float(target_sector_weights.get(sector, 0.0)) + SECTOR_ACTIVE_LIMIT
        )
        if sector_overage > 1e-8:
            flag = f"{sector} exposure exceeds its model target plus the {SECTOR_ACTIVE_LIMIT:.0%} active-risk band"
            flags.append(flag)
            action["risk_details"].append(
                {
                    "label": "Sector exposure",
                    "value": f"{current_sectors.get(sector, 0.0):.1%}",
                    "calculation": "Sum of current position market values in the sector divided by account equity.",
                    "source": "Synchronized paper holdings, certified sector mapping, and constrained model target.",
                    "benchmark": f"Model sector target {target_sector_weights.get(sector, 0.0):.1%} plus {SECTOR_ACTIVE_LIMIT:.0%} active-risk band.",
                    "confidence": "High",
                    "confidence_note": "This is an arithmetic exposure check, not a return forecast.",
                    "available_at": as_of_date.isoformat(),
                }
            )
            sector_limit = action["current_weight"] - sector_overage / holding_sector_counts[sector]
            risk_limit = min(risk_limit, max(0.0, sector_limit))
        rank_percentile = (
            action["rank"] / max(len(ranked), 1)
            if action["rank"] is not None
            else None
        )
        factor_weak = (
            rank_percentile is not None
            and rank_percentile > DEFENSIVE_FACTOR_RANK_PERCENTILE
        )
        ml_weak = (
            model_available
            and action["ml_score"] is not None
            and action["ml_score"] < DEFENSIVE_ML_SCORE
        )
        if factor_weak:
            flag = f"Composite factor rank #{action['rank']} is in the lowest quarter of the certified universe"
            flags.append(flag)
            action["risk_details"].append(
                {
                    "label": "Weak composite factor rank",
                    "value": f"#{action['rank']}",
                    "calculation": "Descending rank of the certified composite factor score across the eligible S&P 500 universe.",
                    "source": "Certified point-in-time factor snapshot.",
                    "benchmark": "Defensive threshold: lowest 25% of eligible S&P 500 names.",
                    "confidence": "Moderate",
                    "confidence_note": "Coverage and cross-sectional strength, not a probability of loss.",
                    "available_at": as_of_date.isoformat(),
                }
            )
        if ml_weak:
            flag = f"ML score {(action['ml_score'] or 0.0):.2f} is below the defensive threshold"
            flags.append(flag)
            action["risk_details"].append(
                {
                    "label": "Weak ML score",
                    "value": f"{action['ml_score'] or 0.0:.2f}",
                    "calculation": "Walk-forward ensemble percentile for next-month excess-return prediction.",
                    "source": "Previously available factor observations and realized historical returns only.",
                    "benchmark": f"Defensive threshold: below {DEFENSIVE_ML_SCORE:.2f} percentile score.",
                    "confidence": "Moderate",
                    "confidence_note": "Model output is a rank, not a probability of loss or a short signal.",
                    "available_at": as_of_date.isoformat(),
                }
            )
        if factor_weak or ml_weak:
            # A weak holding is only trimmed to a defensive half-size, never
            # converted into a short or an automatic exit.
            risk_limit = min(risk_limit, MAX_NAME_WEIGHT / 2)
        action["risk_flags"] = flags
        action["risk_limit_weight"] = risk_limit
        if flags and risk_limit < action["current_weight"] - 1e-8:
            action["bucket"] = "risk"
            action["desired_delta_value"] = equity * risk_limit - action["current_value"]
        elif action["target_weight"] > action["current_weight"] + 1e-8:
            action["bucket"] = "purchase"
            action["desired_delta_value"] = (
                equity * action["target_weight"] - action["current_value"]
            )
        else:
            action["bucket"] = "none"
            action["desired_delta_value"] = 0.0

    _allocate_notional(actions, equity)
    for action in actions:
        delta = action["proposed_delta_value"]
        if abs(delta) < max(1.0, equity * 0.0001):
            continue
        if delta < 0:
            action["action"] = "Reduce"
            action["reason"] = (
                "Portfolio-risk reduction, not a bearish call: "
                + "; ".join(action["risk_flags"])
                + f". This staged change moves the holding toward a {action['risk_limit_weight']:.1%} risk limit."
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
    if not model_available:
        warnings.append(
            "No non-zero walk-forward ML scores were available at this cutoff; ranking is factor-led."
        )
    warnings.extend(
        [
            "Research proposal only. It is not an order, quote, or individualized investment instruction.",
            "Reference prices are the latest locally stored adjusted closes and may differ from executable market prices.",
            "The staged plan limits one-way monthly turnover to 20% of account equity and caps every target name at 5%.",
            "A company missing from the buy target is not a bearish call. Existing holdings are reduced only when their portfolio-risk gate is triggered.",
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
        "risk_reductions": [action for action in actions if action["action"] == "Reduce"],
        "purchase_candidates": [action for action in actions if action["action"] != "Reduce"],
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
