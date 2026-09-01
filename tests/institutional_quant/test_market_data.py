from __future__ import annotations

import asyncio
from datetime import date, datetime

import httpx
import pandas as pd

from institutional_quant.config import Settings
from institutional_quant.market_data import (
    AlpacaHistoricalPriceSync,
    YahooHistoricalPriceSync,
    ticker_at_price_date,
)
from institutional_quant.storage import DuckDBStore, ParquetPriceLake


def test_ticker_at_price_date_rejects_recycled_symbol_and_tracks_rename():
    by_security = {
        ("CIQ:1", "PARA"): [(date(2022, 2, 14), date(2025, 8, 6))],
        ("CIQ:1", "PSKY"): [(date(2025, 8, 7), None)],
        ("CIQ:1", "VIAC"): [(date(2019, 12, 4), date(2022, 2, 13))],
    }
    by_company = {
        "CIQ:1": [
            ("VIAC", date(2019, 12, 4), date(2022, 2, 13)),
            ("PARA", date(2022, 2, 14), date(2025, 8, 6)),
            ("PSKY", date(2025, 8, 7), None),
        ]
    }

    assert (
        ticker_at_price_date("CIQ:1", "PARA", date(2023, 1, 3), by_security, by_company) == "PARA"
    )
    assert (
        ticker_at_price_date("CIQ:1", "PSKY", date(2025, 8, 7), by_security, by_company) == "PSKY"
    )
    assert ticker_at_price_date("CIQ:1", "PARA", date(2025, 8, 7), by_security, by_company) is None


def test_parquet_price_lake_merges_and_prefers_authoritative_source(tmp_path):
    lake = ParquetPriceLake(tmp_path / "market" / "prices.parquet")
    base = {
        "company_id": "CIQ:1",
        "ticker": "AAPL",
        "price_date": date(2025, 1, 2),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "adjusted_close": 100.0,
        "volume": 10.0,
        "effective_at": datetime(2025, 1, 2, 23, 59),
        "as_of_date": date(2025, 1, 3),
        "ingested_at": datetime(2025, 1, 3),
    }
    lake.persist(
        pd.DataFrame(
            [
                {
                    **base,
                    "source": "alpaca_iex_adjusted",
                    "source_file_id": "alpaca",
                },
                {
                    **base,
                    "adjusted_close": 150.0,
                    "source": "yahoo_adjusted",
                    "source_file_id": "yahoo",
                },
            ]
        )
    )
    lake.persist(
        pd.DataFrame(
            [
                {
                    **base,
                    "adjusted_close": 200.0,
                    "source": "capital_iq",
                    "source_file_id": "ciq",
                }
            ]
        )
    )

    prices = lake.load(date(2025, 1, 1), date(2025, 1, 3))
    assert len(prices) == 1
    assert prices.iloc[0]["adjusted_close"] == 200.0
    assert prices.iloc[0]["source"] == "capital_iq"
    assert lake.coverage() == (date(2025, 1, 2), date(2025, 1, 2))
    assert lake.sources() == ["alpaca_iex_adjusted", "capital_iq", "yahoo_adjusted"]


def test_alpaca_price_sync_archives_and_labels_adjusted_iex_data(tmp_path):
    store = DuckDBStore(tmp_path / "market.duckdb")
    store.initialize()
    store.insert_frame(
        "instruments",
        pd.DataFrame(
            [
                {
                    "company_id": "1",
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "sector": "Information Technology",
                    "currency": "USD",
                    "effective_at": datetime(2026, 8, 31, 23, 59),
                    "as_of_date": date(2026, 8, 31),
                    "source_file_id": "fixture",
                    "ingested_at": datetime(2026, 9, 1),
                }
            ]
        ),
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "bars": {
                    "AAPL": [
                        {
                            "t": "2021-09-01T04:00:00Z",
                            "o": 10.0,
                            "h": 11.0,
                            "l": 9.0,
                            "c": 10.5,
                            "v": 100,
                        }
                    ],
                    "SPY": [
                        {
                            "t": "2021-09-01T04:00:00Z",
                            "o": 20.0,
                            "h": 21.0,
                            "l": 19.0,
                            "c": 20.5,
                            "v": 200,
                        }
                    ],
                },
                "next_page_token": None,
            },
        )

    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "market.duckdb",
        raw_data_dir=tmp_path / "raw",
        alpaca_paper_key="paper-key",
        alpaca_paper_secret="paper-secret",
    )
    result = asyncio.run(
        AlpacaHistoricalPriceSync(
            store,
            settings,
            transport=httpx.MockTransport(handler),
        ).sync(
            date(2021, 9, 1),
            date(2021, 9, 1),
            sync_date=date(2026, 9, 1),
        )
    )

    assert result.imported_rows == 2
    assert result.issues == []
    assert requests[0].url.params["adjustment"] == "all"
    assert requests[0].url.params["feed"] == "iex"
    assert requests[0].headers["APCA-API-KEY-ID"] == "paper-key"
    prices = store.query_df("SELECT * FROM prices ORDER BY ticker")
    assert set(prices["source"]) == {"alpaca_iex_adjusted"}
    assert prices.loc[prices["ticker"] == "SPY", "company_id"].item() == "BENCHMARK:SPY"
    assert any((tmp_path / "raw" / "prices_alpaca").glob("*.jsonl"))


def test_price_queries_prefer_capital_iq_over_alpaca(tmp_path):
    store = DuckDBStore(tmp_path / "priority.duckdb")
    store.initialize()
    base = {
        "company_id": "1",
        "ticker": "AAPL",
        "price_date": date(2025, 1, 2),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "adjusted_close": 100.0,
        "volume": 10.0,
        "effective_at": datetime(2025, 1, 2, 23, 59),
        "as_of_date": date(2025, 1, 3),
        "source_file_id": "alpaca",
        "ingested_at": datetime(2025, 1, 3),
    }
    columns = [
        "company_id",
        "ticker",
        "price_date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
        "effective_at",
        "as_of_date",
        "source_file_id",
        "ingested_at",
    ]
    store.insert_frame(
        "prices",
        pd.DataFrame([{**base, "source": "alpaca_iex_adjusted"}])[columns],
    )
    store.insert_frame(
        "prices",
        pd.DataFrame(
            [
                {
                    **base,
                    "open": 200.0,
                    "close": 200.0,
                    "adjusted_close": 200.0,
                    "source": "capital_iq",
                    "source_file_id": "ciq",
                }
            ]
        )[columns],
    )

    from institutional_quant.backtest import BacktestEngine

    prices = BacktestEngine(store)._prices(date(2025, 1, 2), date(2025, 1, 2))
    assert len(prices) == 1
    assert prices.iloc[0]["adjusted_close"] == 200.0


def test_alpaca_price_sync_flags_adjusted_return_outliers(tmp_path):
    store = DuckDBStore(tmp_path / "outlier.duckdb")
    store.initialize()
    store.insert_frame(
        "instruments",
        pd.DataFrame(
            [
                {
                    "company_id": "1",
                    "ticker": "ONE",
                    "company_name": "One",
                    "sector": "Industrials",
                    "currency": "USD",
                    "effective_at": datetime(2021, 1, 1),
                    "as_of_date": date(2026, 8, 31),
                    "source_file_id": "fixture",
                    "ingested_at": datetime(2026, 9, 1),
                }
            ]
        ),
    )
    store.insert_frame(
        "index_membership",
        pd.DataFrame(
            [
                {
                    "company_id": "1",
                    "ticker": "ONE",
                    "index_code": "SP500",
                    "member_from": date(2021, 1, 1),
                    "member_to": None,
                    "effective_at": datetime(2021, 1, 1),
                    "as_of_date": date(2026, 8, 31),
                    "source_file_id": "fixture",
                    "ingested_at": datetime(2026, 9, 1),
                }
            ]
        ),
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "bars": {
                    "ONE": [
                        {
                            "t": "2021-09-01T04:00:00Z",
                            "o": 10,
                            "h": 10,
                            "l": 10,
                            "c": 10,
                            "v": 100,
                        },
                        {
                            "t": "2021-09-02T04:00:00Z",
                            "o": 100,
                            "h": 100,
                            "l": 100,
                            "c": 100,
                            "v": 100,
                        },
                    ],
                    "SPY": [],
                },
                "next_page_token": None,
            },
        )

    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "outlier.duckdb",
        raw_data_dir=tmp_path / "raw",
        alpaca_paper_key="paper-key",
        alpaca_paper_secret="paper-secret",
    )
    result = asyncio.run(
        AlpacaHistoricalPriceSync(store, settings, transport=httpx.MockTransport(handler)).sync(
            date(2021, 9, 1), date(2021, 9, 2)
        )
    )

    issue = next(item for item in result.issues if item.code == "ADJUSTED_PRICE_RETURN_OUTLIER")
    assert issue.severity.value == "error"
    assert "ONE" in issue.message


def test_yahoo_sync_applies_adjustment_factor_to_ohlc(tmp_path):
    store = DuckDBStore(tmp_path / "yahoo.duckdb")
    store.initialize()
    store.insert_frame(
        "instruments",
        pd.DataFrame(
            [
                {
                    "company_id": "1",
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "sector": "Information Technology",
                    "currency": "USD",
                    "effective_at": datetime(2021, 1, 1),
                    "as_of_date": date(2026, 8, 31),
                    "source_file_id": "fixture",
                    "ingested_at": datetime(2026, 9, 1),
                }
            ]
        ),
    )
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        raw_close = 20.0 if request.url.path.endswith("/SPY") else 10.0
        return httpx.Response(
            200,
            json={
                "chart": {
                    "result": [
                        {
                            "timestamp": [1630474200],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [raw_close * 0.9],
                                        "high": [raw_close * 1.1],
                                        "low": [raw_close * 0.8],
                                        "close": [raw_close],
                                        "volume": [100],
                                    }
                                ],
                                "adjclose": [{"adjclose": [raw_close * 0.5]}],
                            },
                        }
                    ],
                    "error": None,
                }
            },
        )

    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "yahoo.duckdb",
        raw_data_dir=tmp_path / "raw",
    )
    result = asyncio.run(
        YahooHistoricalPriceSync(store, settings, transport=httpx.MockTransport(handler)).sync(
            date(2021, 9, 1), date(2021, 9, 1), sync_date=date(2026, 9, 1)
        )
    )

    assert result.imported_rows == 2
    prices = store.query_df("SELECT * FROM prices ORDER BY ticker")
    apple = prices.loc[prices["ticker"] == "AAPL"].iloc[0]
    assert apple["source"] == "yahoo_adjusted"
    assert apple["open"] == 4.5
    assert apple["adjusted_close"] == 5.0
    assert any(path.endswith("/AAPL") for path in requested)
    assert any(path.endswith("/SPY") for path in requested)
    assert any((tmp_path / "raw" / "prices_yahoo").glob("*.jsonl"))


def test_yahoo_recycled_paramount_tickers_use_successor_history():
    assert YahooHistoricalPriceSync._yahoo_symbol("VIAC") == "PSKY"
    assert YahooHistoricalPriceSync._yahoo_symbol("PARA") == "PSKY"
    assert YahooHistoricalPriceSync._yahoo_symbol("BRK.B") == "BRK-B"
