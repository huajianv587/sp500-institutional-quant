from __future__ import annotations

import httpx

from institutional_quant.config import Settings
from institutional_quant.preflight import run_preflight


def test_live_preflight_authenticates_paper_account(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_get(url, *, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return httpx.Response(
            200,
            json={"status": "ACTIVE", "trading_blocked": False},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    settings = Settings(
        database_backend="duckdb",
        alpaca_paper_key="paper-key",
        alpaca_paper_secret="paper-secret",
    )

    checks = run_preflight(settings, live=True)
    paper = next(check for check in checks if check.name == "Alpaca paper connection")

    assert paper.ready is True
    assert captured["url"] == "https://paper-api.alpaca.markets/v2/account"
    assert captured["headers"] == {
        "APCA-API-KEY-ID": "paper-key",
        "APCA-API-SECRET-KEY": "paper-secret",
    }


def test_live_preflight_reports_blocked_paper_account(monkeypatch) -> None:
    def fake_get(url, *, headers, timeout):
        return httpx.Response(
            200,
            json={"status": "ACTIVE", "trading_blocked": True},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    settings = Settings(
        database_backend="duckdb",
        alpaca_paper_key="paper-key",
        alpaca_paper_secret="paper-secret",
    )

    checks = run_preflight(settings, live=True)
    paper = next(check for check in checks if check.name == "Alpaca paper connection")

    assert paper.ready is False
    assert "trading_blocked=True" in paper.detail
