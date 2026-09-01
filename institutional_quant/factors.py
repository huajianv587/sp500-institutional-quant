from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from .calculations import derive_fundamental_features
from .storage import Store

FACTOR_FAMILIES: dict[str, dict[str, float]] = {
    "value": {"earnings_yield": 1.0, "fcf_yield": 1.0, "ebitda_to_ev": 1.0},
    "quality": {
        "roic": 1.0,
        "gross_profitability": 1.0,
        "accruals": -1.0,
        "net_debt_ebitda": -1.0,
    },
    "growth": {"revenue_growth": 1.0, "eps_growth": 1.0, "margin_change": 1.0},
    "revisions": {
        "eps_revision_1m": 1.0,
        "eps_revision_3m": 1.0,
        "estimate_surprise": 1.0,
    },
    "momentum": {"momentum_12_1": 1.0, "momentum_6_1": 1.0},
    "low_risk": {"volatility_252d": -1.0, "beta_252d": -1.0},
}

INSTITUTIONAL_FACTOR_FAMILIES = ("value", "quality", "growth", "revisions")
MIN_FACTOR_FAMILIES = 4
MIN_INSTITUTIONAL_FACTOR_FAMILIES = 2


def _winsorized_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lower, upper = valid.quantile([0.025, 0.975])
    clipped = numeric.clip(lower, upper)
    standard_deviation = clipped.std(ddof=0)
    if not standard_deviation or np.isnan(standard_deviation):
        return pd.Series(0.0, index=series.index)
    return (clipped - clipped.mean()) / standard_deviation


def _sector_neutral_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    result = pd.Series(index=frame.index, dtype=float)
    for _, index in frame.groupby("sector", dropna=False).groups.items():
        if len(index) >= 3:
            result.loc[index] = _winsorized_zscore(frame.loc[index, column])
        else:
            result.loc[index] = _winsorized_zscore(frame[column]).loc[index]
    return result


def _combine_factor_families(frame: pd.DataFrame) -> pd.DataFrame:
    family_columns = [f"factor_{family}" for family in FACTOR_FAMILIES]
    institutional_columns = [f"factor_{family}" for family in INSTITUTIONAL_FACTOR_FAMILIES]
    frame["factor_family_count"] = frame[family_columns].notna().sum(axis=1)
    frame["institutional_factor_family_count"] = frame[institutional_columns].notna().sum(axis=1)
    eligible = (frame["factor_family_count"] >= MIN_FACTOR_FAMILIES) & (
        frame["institutional_factor_family_count"] >= MIN_INSTITUTIONAL_FACTOR_FAMILIES
    )
    frame["factor_score"] = frame[family_columns].mean(axis=1, skipna=True).where(eligible)
    return frame


class FactorEngine:
    def __init__(self, store: Store):
        self.store = store

    def _universe(self, as_of_date: date) -> pd.DataFrame:
        cutoff = datetime.combine(as_of_date, time.max)
        return self.store.query_df(
            """
            WITH latest_instrument AS (
                SELECT company_id, ticker, company_name, sector,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_id ORDER BY effective_at DESC
                       ) AS rn
                FROM instruments WHERE effective_at <= ?
            )
            SELECT m.company_id, m.ticker, COALESCE(i.sector, 'Unknown') AS sector,
                   COALESCE(i.company_name, m.ticker) AS company_name
            FROM index_membership m
            LEFT JOIN latest_instrument i ON i.company_id = m.company_id AND i.rn = 1
            WHERE m.index_code = 'SP500'
              AND m.member_from <= ?
              AND (m.member_to IS NULL OR m.member_to >= ?)
              AND m.effective_at <= ?
            ORDER BY m.company_id
            """,
            [cutoff, as_of_date, as_of_date, cutoff],
        )

    def _latest_metrics(self, table: str, as_of_date: date) -> tuple[pd.DataFrame, list[str]]:
        if table not in {"fundamentals", "estimates"}:
            raise ValueError(table)
        cutoff = datetime.combine(as_of_date, time.max)
        period_column = "period_end" if table == "fundamentals" else "fiscal_period"
        frame = self.store.query_df(
            f"""
            WITH ranked AS (
                SELECT company_id, metric, value, source_file_id, effective_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_id, metric
                           ORDER BY {period_column} DESC, effective_at DESC
                       ) AS rn
                FROM {table}
                WHERE effective_at <= ?
            )
            SELECT company_id, metric, value, source_file_id, effective_at
            FROM ranked WHERE rn = 1
            """,
            [cutoff],
        )
        if frame.empty:
            return pd.DataFrame(), []
        sources = sorted(set(frame["source_file_id"].astype(str)))
        wide = frame.pivot(index="company_id", columns="metric", values="value")
        wide.columns = [str(column).strip().lower() for column in wide.columns]
        return wide.reset_index(), sources

    def _market_features(
        self, universe: pd.DataFrame, as_of_date: date
    ) -> tuple[pd.DataFrame, list[str]]:
        start = as_of_date - timedelta(days=430)
        company_ids = universe["company_id"].astype(str).drop_duplicates().tolist()
        prices = self.store.load_prices(
            start,
            as_of_date,
            company_ids=company_ids,
            tickers=["SPY"],
        )
        if not prices.empty:
            prices = prices.loc[
                pd.to_datetime(prices["effective_at"])
                <= pd.Timestamp(datetime.combine(as_of_date, time.max))
            ][["company_id", "ticker", "price_date", "adjusted_close", "source_file_id"]]
        if prices.empty:
            return pd.DataFrame(columns=["company_id"]), []
        sources = sorted(set(prices["source_file_id"].astype(str)))
        output: list[dict[str, float | str]] = []
        spy = prices.loc[prices["ticker"] == "SPY"].set_index("price_date")["adjusted_close"]
        spy_returns = spy.pct_change()
        for company_id, group in prices.loc[prices["ticker"] != "SPY"].groupby("company_id"):
            series = group.drop_duplicates("price_date").set_index("price_date")["adjusted_close"]
            returns = series.pct_change().dropna()
            momentum_12_1 = np.nan
            momentum_6_1 = np.nan
            if len(series) >= 22:
                one_month_ago = series.iloc[-22]
                if len(series) >= 253 and series.iloc[-253] > 0:
                    momentum_12_1 = one_month_ago / series.iloc[-253] - 1
                if len(series) >= 127 and series.iloc[-127] > 0:
                    momentum_6_1 = one_month_ago / series.iloc[-127] - 1
            volatility = (
                returns.tail(252).std(ddof=0) * np.sqrt(252) if len(returns) >= 40 else np.nan
            )
            aligned = pd.concat(
                [returns.rename("asset"), spy_returns.rename("spy")], axis=1
            ).dropna()
            beta = np.nan
            if len(aligned) >= 40 and aligned["spy"].var(ddof=0) > 0:
                beta = aligned.cov(ddof=0).loc["asset", "spy"] / aligned["spy"].var(ddof=0)
            output.append(
                {
                    "company_id": company_id,
                    "momentum_12_1": momentum_12_1,
                    "momentum_6_1": momentum_6_1,
                    "volatility_252d": volatility,
                    "beta_252d": beta,
                }
            )
        return pd.DataFrame(output), sources

    def snapshot(self, as_of_date: date) -> pd.DataFrame:
        universe = self._universe(as_of_date)
        if universe.empty:
            raise ValueError(f"No historical S&P 500 membership for {as_of_date}")
        fundamentals, fundamental_sources = self._latest_metrics("fundamentals", as_of_date)
        estimates, estimate_sources = self._latest_metrics("estimates", as_of_date)
        market, price_sources = self._market_features(universe, as_of_date)
        frame = universe.merge(fundamentals, on="company_id", how="left")
        frame = frame.merge(estimates, on="company_id", how="left", suffixes=("", "_estimate"))
        frame = frame.merge(market, on="company_id", how="left")
        frame = derive_fundamental_features(frame)

        all_features = sorted(
            {feature for features in FACTOR_FAMILIES.values() for feature in features}
        )
        for feature in all_features:
            if feature not in frame.columns:
                frame[feature] = np.nan
            frame[f"z_{feature}"] = _sector_neutral_zscore(frame, feature)

        for family, definitions in FACTOR_FAMILIES.items():
            components = [
                frame[f"z_{feature}"] * direction for feature, direction in definitions.items()
            ]
            frame[f"factor_{family}"] = pd.concat(components, axis=1).mean(axis=1, skipna=True)
        frame = _combine_factor_families(frame)

        source_ids = sorted(set(fundamental_sources + estimate_sources + price_sources))
        snapshot_payload = {
            "as_of_date": as_of_date.isoformat(),
            "source_ids": source_ids,
            "companies": sorted(frame["company_id"].astype(str)),
            "families": FACTOR_FAMILIES,
        }
        frame["source_snapshot_hash"] = hashlib.sha256(
            json.dumps(snapshot_payload, sort_keys=True).encode()
        ).hexdigest()
        frame["as_of_date"] = as_of_date
        return frame

    def persist(self, frame: pd.DataFrame) -> None:
        records = []
        raw_features = sorted(
            {feature for family in FACTOR_FAMILIES.values() for feature in family}
        )
        for row in frame.to_dict(orient="records"):
            feature_json = {
                key: (None if pd.isna(row.get(key)) else float(row[key])) for key in raw_features
            }
            records.append(
                {
                    "company_id": row["company_id"],
                    "ticker": row["ticker"],
                    "as_of_date": row["as_of_date"],
                    "sector": row["sector"],
                    "feature_json": json.dumps(feature_json),
                    "factor_score": float(row["factor_score"]),
                    "elastic_score": row.get("elastic_score"),
                    "tree_score": row.get("tree_score"),
                    "ml_score": row.get("ml_score"),
                    "ensemble_score": row.get("ensemble_score"),
                    "next_month_excess_return": row.get("next_month_excess_return"),
                    "source_snapshot_hash": row["source_snapshot_hash"],
                }
            )
        self.store.insert_frame("factor_observations", pd.DataFrame(records))
