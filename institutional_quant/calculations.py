from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > 1e-12)
    return numerator / denominator


def _inverse_multiple_with_unit_calibration(
    numerator: pd.Series,
    denominator: pd.Series,
    reported_multiple: pd.Series,
) -> pd.Series:
    """Prefer reported multiples and calibrate raw amount scales for fallbacks.

    Capital IQ screen exports commonly mix statement values in $000 with market
    values in $M.  Where both forms exist, their overlap reveals the deterministic
    scale factor; missing reported multiples can then use a comparable raw ratio.
    """
    raw = _safe_divide(numerator, denominator)
    reported = _safe_divide(pd.Series(1.0, index=raw.index), reported_multiple)
    overlap = pd.concat([raw.rename("raw"), reported.rename("reported")], axis=1).dropna()
    overlap = overlap.loc[overlap["reported"].abs() > 1e-12]
    if len(overlap) >= 3:
        scale = (overlap["raw"] / overlap["reported"]).abs().median()
        if np.isfinite(scale) and scale > 10:
            raw = raw / scale
    return reported.fillna(raw)


def derive_estimate_revision_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn Capital IQ analyst revision counts into bounded breadth factors.

    Capital IQ supplies deterministic counts of analysts revising normalized EPS
    upward or downward over one- and three-month windows.  Breadth is the net
    direction divided by the larger of current analyst coverage and total
    revisers.  The latter matters for three-month windows where one analyst can
    revise more than once.  This keeps the LLM out of the arithmetic and makes
    the resulting [-1, 1] signal auditable without relying on clipping alone.
    """
    output = frame.copy()

    def series(name: str) -> pd.Series:
        return pd.to_numeric(
            output.get(name, pd.Series(np.nan, index=output.index)), errors="coerce"
        )

    analyst_count = series("eps_analyst_count_1m")
    for window in ("1m", "3m"):
        upward = series(f"eps_up_revisions_{window}")
        downward = series(f"eps_down_revisions_{window}")
        revisers = upward + downward
        denominator = pd.concat(
            [analyst_count.where(analyst_count > 0), revisers.where(revisers > 0)],
            axis=1,
        ).max(axis=1, skipna=True)
        denominator = denominator.where(denominator > 0)
        breadth = _safe_divide(upward - downward, denominator).clip(-1.0, 1.0)
        output[f"eps_revision_{window}"] = breadth
    return output


def derive_fundamental_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, auditable ratios; exported ratio fields remain valid fallbacks."""
    output = frame.copy()

    def series(name: str) -> pd.Series:
        return pd.to_numeric(
            output.get(name, pd.Series(np.nan, index=output.index)), errors="coerce"
        )

    earnings_yield = _inverse_multiple_with_unit_calibration(
        series("net_income"), series("market_cap"), series("price_to_earnings")
    )
    ebitda_to_ev = _inverse_multiple_with_unit_calibration(
        series("ebitda"), series("enterprise_value"), series("tev_ebitda")
    )
    derived = {
        "earnings_yield": earnings_yield,
        "fcf_yield": _safe_divide(series("free_cash_flow"), series("market_cap")),
        "ebitda_to_ev": ebitda_to_ev,
        "roic": _safe_divide(series("nopat"), series("invested_capital")),
        "gross_profitability": _safe_divide(series("gross_profit"), series("total_assets")),
        "accruals": _safe_divide(
            series("net_income") - series("operating_cash_flow"), series("total_assets")
        ),
        "net_debt_ebitda": _safe_divide(series("net_debt"), series("ebitda")),
        "revenue_growth": _safe_divide(series("revenue"), series("revenue_prior_year")) - 1,
        "eps_growth": _safe_divide(series("eps"), series("eps_prior_year")).abs() - 1,
        "margin_change": series("operating_margin") - series("operating_margin_prior_year"),
    }
    for name, values in derived.items():
        if name not in output:
            output[name] = values
        else:
            output[name] = pd.to_numeric(output[name], errors="coerce").fillna(values)
    return output
