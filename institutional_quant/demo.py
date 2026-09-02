from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .ingestion import CapitalIQImporter
from .schemas import DatasetKind
from .storage import DuckDBStore

SECTORS = [
    "Information Technology",
    "Financials",
    "Health Care",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]


def build_synthetic_demo(database_path: Path, raw_data_dir: Path) -> DuckDBStore:
    """Generate licensed-data-free fixtures for engineering verification only."""
    # The demo is an acceptance fixture, not a durable research database.  Rebuild
    # it on every invocation so the documented two-command smoke test is
    # repeatable after schema or fixture changes.
    for suffix in ("", ".wal"):
        candidate = Path(f"{database_path}{suffix}")
        if candidate.exists():
            candidate.unlink()
    settings = Settings(
        database_backend="duckdb",
        database_path=database_path,
        raw_data_dir=raw_data_dir,
        report_dir=database_path.parent / "reports",
        ciq_cloud_storage_confirmed=False,
    )
    settings.ensure_directories()
    store = DuckDBStore(database_path)
    store.initialize()
    importer = CapitalIQImporter(store, settings)
    rng = np.random.default_rng(17)
    companies = [f"SYN{i:03d}" for i in range(35)]
    tickers = [f"Q{i:02d}" for i in range(35)]
    instruments = pd.DataFrame(
        {
            "company_id": companies,
            "ticker": tickers,
            "company_name": [f"Synthetic Company {i}" for i in range(35)],
            "sector": [SECTORS[i % len(SECTORS)] for i in range(35)],
            "currency": "USD",
            "effective_at": "2016-01-01T00:00:00Z",
            "as_of_date": date(2016, 1, 1),
        }
    )
    membership = pd.DataFrame(
        {
            "company_id": companies,
            "ticker": tickers,
            "index_code": "SP500",
            "member_from": date(2016, 1, 1),
            "member_to": None,
            "effective_at": "2016-01-01T00:00:00Z",
            "as_of_date": date(2016, 1, 1),
        }
    )
    days = pd.bdate_range("2017-01-03", "2026-09-01")
    market_shocks = rng.normal(0.0003, 0.009, len(days))
    price_rows = []
    for index, (company_id, ticker) in enumerate(zip(companies, tickers, strict=True)):
        quality = (index % 7 - 3) / 10000
        returns = 0.0001 + quality + 0.8 * market_shocks + rng.normal(0, 0.011, len(days))
        close = 30 * np.exp(np.cumsum(returns))
        for day, value in zip(days, close, strict=True):
            price_rows.append(
                {
                    "company_id": company_id,
                    "ticker": ticker,
                    "price_date": day.date(),
                    "open": value * (1 + rng.normal(0, 0.001)),
                    "high": value * 1.01,
                    "low": value * 0.99,
                    "close": value,
                    "adjusted_close": value,
                    "volume": 1_000_000 + index * 10_000,
                    "source": "synthetic",
                    "effective_at": f"{day.date()}T23:00:00Z",
                    "as_of_date": day.date(),
                }
            )
    spy = 100 * np.exp(np.cumsum(market_shocks))
    for day, value in zip(days, spy, strict=True):
        price_rows.append(
            {
                "company_id": "BENCHMARK_SPY",
                "ticker": "SPY",
                "price_date": day.date(),
                "open": value,
                "high": value * 1.005,
                "low": value * 0.995,
                "close": value,
                "adjusted_close": value,
                "volume": 10_000_000,
                "source": "synthetic",
                "effective_at": f"{day.date()}T23:00:00Z",
                "as_of_date": day.date(),
            }
        )
    fundamental_rows = []
    estimate_rows = []
    periods = pd.date_range("2016-03-31", "2026-06-30", freq="QE")
    for index, (company_id, ticker) in enumerate(zip(companies, tickers, strict=True)):
        base_revenue = 2_000 + 80 * index
        previous_revenue = None
        previous_eps = None
        previous_margin = None
        for period_number, period in enumerate(periods):
            trend = 1 + (0.018 + index % 5 * 0.002) * period_number
            revenue = base_revenue * trend * (1 + rng.normal(0, 0.025))
            eps = (1.2 + 0.04 * index) * trend * (1 + rng.normal(0, 0.03))
            margin = 0.10 + (index % 8) * 0.012 + rng.normal(0, 0.004)
            values = {
                "revenue": revenue,
                "eps": eps,
                "operating_margin": margin,
                "revenue_prior_year": previous_revenue,
                "eps_prior_year": previous_eps,
                "operating_margin_prior_year": previous_margin,
                "net_income": revenue * margin * 0.75,
                "market_cap": revenue * (4 + index % 5),
                "free_cash_flow": revenue * margin * 0.62,
                "ebitda": revenue * (margin + 0.05),
                "enterprise_value": revenue * (4.5 + index % 5),
                "nopat": revenue * margin * 0.78,
                "invested_capital": revenue * 1.8,
                "gross_profit": revenue * (0.28 + index % 6 * 0.025),
                "total_assets": revenue * 2.1,
                "operating_cash_flow": revenue * margin * 0.88,
                "net_debt": revenue * (0.2 + index % 4 * 0.12),
            }
            effective = (period + pd.Timedelta(days=42)).date()
            # Capital IQ historical exports are frozen research snapshots.  Keep
            # the filing availability timestamp separate from the observation
            # cut-off used by the strategy.  The final partial quarter is frozen
            # at the study end rather than silently disappearing from coverage.
            snapshot = min(
                (pd.Timestamp(effective) + pd.offsets.QuarterEnd(0)).date(),
                date(2026, 8, 31),
            )
            for metric, value in values.items():
                if value is not None:
                    fundamental_rows.append(
                        {
                            "company_id": company_id,
                            "ticker": ticker,
                            "period_end": period.date(),
                            "period_type": "Quarterly",
                            "effective_at": f"{effective}T12:00:00Z",
                            "as_of_date": snapshot,
                            "metric": metric,
                            "value": value,
                            "unit": "USD",
                        }
                    )
            previous_revenue, previous_eps, previous_margin = revenue, eps, margin
        for month in pd.date_range("2017-01-31", "2026-08-31", freq="ME"):
            analyst_count = 18 + index % 17
            directional_bias = (index % 7) - 3
            up_1m = int(np.clip(rng.poisson(4 + max(directional_bias, 0)), 0, analyst_count))
            down_1m = int(np.clip(rng.poisson(4 + max(-directional_bias, 0)), 0, analyst_count))
            up_3m = int(np.clip(rng.poisson(12 + max(directional_bias, 0)), 0, 50))
            down_3m = int(np.clip(rng.poisson(12 + max(-directional_bias, 0)), 0, 50))
            for metric, value in {
                "eps_estimate": (1.2 + 0.04 * index)
                * (1 + (0.018 + index % 5 * 0.002) * (len(periods) + month.year - 2026)),
                "eps_analyst_count_1m": analyst_count,
                "eps_up_revisions_1m": up_1m,
                "eps_down_revisions_1m": down_1m,
                "eps_up_revisions_3m": up_3m,
                "eps_down_revisions_3m": down_3m,
                "estimate_surprise": rng.normal(index % 5 * 0.003, 0.025),
            }.items():
                estimate_rows.append(
                    {
                        "company_id": company_id,
                        "ticker": ticker,
                        "fiscal_period": (month + pd.offsets.MonthEnd(3)).date(),
                        "effective_at": f"{month.date()}T20:00:00Z",
                        "valid_to": None,
                        "as_of_date": month.date(),
                        "metric": metric,
                        "value": value,
                        "unit": "ratio",
                    }
                )
    datasets = {
        DatasetKind.INSTRUMENTS: instruments,
        DatasetKind.INDEX_MEMBERSHIP: membership,
        DatasetKind.FUNDAMENTALS: pd.DataFrame(fundamental_rows),
        DatasetKind.ESTIMATES: pd.DataFrame(estimate_rows),
        DatasetKind.PRICES: pd.DataFrame(price_rows),
    }
    with tempfile.TemporaryDirectory() as temporary:
        for kind, frame in datasets.items():
            path = Path(temporary) / f"synthetic_{kind.value}.csv"
            frame.to_csv(path, index=False)
            importer.import_file(path, kind)
    return store
