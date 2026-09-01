from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > 1e-12)
    return numerator / denominator


def derive_fundamental_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, auditable ratios; exported ratio fields remain valid fallbacks."""
    output = frame.copy()

    def series(name: str) -> pd.Series:
        return pd.to_numeric(
            output.get(name, pd.Series(np.nan, index=output.index)), errors="coerce"
        )

    derived = {
        "earnings_yield": _safe_divide(series("net_income"), series("market_cap")),
        "fcf_yield": _safe_divide(series("free_cash_flow"), series("market_cap")),
        "ebitda_to_ev": _safe_divide(series("ebitda"), series("enterprise_value")),
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
