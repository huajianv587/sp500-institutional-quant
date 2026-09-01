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
        column = f"eps_revision_{window}"
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(breadth)
        else:
            output[column] = breadth
    return output


def derive_point_in_time_estimate_changes(
    history: pd.DataFrame, as_of_date: object
) -> pd.DataFrame:
    """Derive 1m/3m EPS consensus changes without mixing fiscal targets.

    The latest consensus available at the signal date is compared with the
    latest observation on or before each historical cutoff for the exact same
    company and fiscal period.  Dividing by the absolute prior estimate keeps
    an improvement from a negative EPS estimate directionally positive.
    """
    output_columns = ["company_id", "eps_revision_1m", "eps_revision_3m"]
    required = {"company_id", "fiscal_period", "as_of_date", "effective_at", "value"}
    if history.empty or not required.issubset(history.columns):
        return pd.DataFrame(columns=output_columns)

    working = history.copy()
    working = working.loc[working["company_id"].notna()].copy()
    working["company_id"] = working["company_id"].astype(str)
    working["fiscal_period"] = pd.to_datetime(working["fiscal_period"], errors="coerce")
    working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce")
    working["effective_at"] = pd.to_datetime(working["effective_at"], errors="coerce")
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    signal_date = pd.Timestamp(as_of_date).normalize()
    working = working.loc[
        working["company_id"].notna()
        & working["fiscal_period"].notna()
        & working["as_of_date"].notna()
        & working["effective_at"].notna()
        & working["value"].notna()
        & (working["as_of_date"] <= signal_date)
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=output_columns)

    order = ["company_id", "as_of_date", "effective_at"]
    if "ingested_at" in working.columns:
        working["ingested_at"] = pd.to_datetime(working["ingested_at"], errors="coerce")
        order.append("ingested_at")
    working = working.sort_values(order).drop_duplicates(
        ["company_id", "fiscal_period", "as_of_date"], keep="last"
    )

    rows: list[dict[str, float | str]] = []
    for company_id, company_history in working.groupby("company_id", sort=False):
        current = company_history.iloc[-1]
        same_target = company_history.loc[
            company_history["fiscal_period"] == current["fiscal_period"]
        ]
        row: dict[str, float | str] = {"company_id": str(company_id)}
        for months in (1, 3):
            cutoff = (signal_date.to_period("M") - months).end_time.normalize()
            prior = same_target.loc[same_target["as_of_date"] <= cutoff]
            revision = np.nan
            if not prior.empty:
                prior_value = float(prior.iloc[-1]["value"])
                if abs(prior_value) > 1e-12:
                    revision = (float(current["value"]) - prior_value) / abs(prior_value)
            row[f"eps_revision_{months}m"] = revision
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


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
