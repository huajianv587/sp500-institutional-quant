from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pandas as pd

from institutional_quant.alpaca import AlpacaPaperClient
from institutional_quant.config import Settings
from institutional_quant.demo import build_synthetic_demo
from institutional_quant.ingestion import CapitalIQImporter
from institutional_quant.operations import run_daily, run_full_cycle, run_weekly
from institutional_quant.schemas import DatasetKind
from institutional_quant.storage import DuckDBStore


def test_market_return_export_maps_duplicate_price_change_columns(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "returns.duckdb")
    store.initialize()
    settings = Settings(database_backend="duckdb", database_path=store.path, raw_data_dir=tmp_path / "raw")
    source = tmp_path / "returns.csv"
    pd.DataFrame(
        {"Entity ID": ["C1"], "Ticker": ["AAA"], "Price Change (%)": [1.0], "Price Change (%).1": [2.0], "Price Change (%).2": [3.0]}
    ).to_csv(source, index=False)
    result = CapitalIQImporter(store, settings).import_file(
        source,
        DatasetKind.MARKET_RETURNS,
        current_snapshot_as_of=date(2026, 9, 3),
        current_snapshot_effective_at=datetime(2026, 9, 3, 12, 0),
    )
    assert result.imported_rows == 1
    row = store.query_df("SELECT * FROM market_returns").iloc[0]
    assert (row.return_1d, row.return_1w, row.return_1m) == (1.0, 2.0, 3.0)
    assert CapitalIQImporter(store, settings).import_file(
        source,
        DatasetKind.MARKET_RETURNS,
        current_snapshot_as_of=date(2026, 9, 3),
        current_snapshot_effective_at=datetime(2026, 9, 3, 12, 0),
    ).idempotent


def test_daily_and_weekly_operations_are_deterministic(tmp_path: Path) -> None:
    store = build_synthetic_demo(tmp_path / "demo.duckdb", tmp_path / "raw")
    settings = Settings(database_backend="duckdb", database_path=tmp_path / "demo.duckdb", deepseek_api_key="test")
    daily = asyncio.run(run_daily(store, settings, date(2026, 8, 31)))
    assert daily.result["agent_committee"] is False
    assert daily.result["automatic_rebalance"] is False
    assert len(daily.result["candidates"]) > 0
    weekly = asyncio.run(run_weekly(store, settings, date(2026, 8, 31)))
    assert weekly.status == "held"
    assert weekly.result["turnover_cap"] == 0.05


def test_full_cycle_stops_at_one_share_paper_approval_checkpoint(tmp_path: Path, monkeypatch) -> None:
    store = build_synthetic_demo(tmp_path / "demo.duckdb", tmp_path / "raw")
    settings = Settings(
        database_backend="duckdb", database_path=tmp_path / "demo.duckdb",
        deepseek_api_key="test", alpaca_paper_key="key", alpaca_paper_secret="secret",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={"equity": "100000", "cash": "100000", "status": "ACTIVE"})
        if request.url.path == "/v2/positions":
            return httpx.Response(200, json=[])
        if request.url.path == "/v2/stocks/trades/latest":
            symbols = request.url.params.get("symbols", "Q00")
            return httpx.Response(200, json={"trades": {symbol: {"p": 100.0} for symbol in symbols.split(",")}})
        if request.url.path == "/v2/orders" and request.method == "POST":
            payload = json.loads(request.content)
            return httpx.Response(200, json={"id": "paper-1", "status": "accepted", **payload})
        if request.url.path == "/v2/orders":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    client = AlpacaPaperClient(settings, transport=httpx.MockTransport(handler))
    async def fake_monthly(*args, **kwargs):
        from institutional_quant.schemas import OperationResult
        return OperationResult(operation_id="monthly", cadence="monthly", as_of_date=date(2026, 8, 31), status="held", message="mocked", result={})
    monkeypatch.setattr("institutional_quant.operations.run_monthly", fake_monthly)
    result = asyncio.run(run_full_cycle(store, settings, client, submit_paper_order=True, as_of_date=date(2026, 8, 31)))
    assert result.status == "awaiting_approval"
    assert result.result["approval_required"] is True
    assert result.result["submitted_orders"] == []
    assert result.result["paper_order"]["previews"][0]["qty"] == 1.0
