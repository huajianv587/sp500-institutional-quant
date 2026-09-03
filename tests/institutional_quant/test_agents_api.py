from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from types import SimpleNamespace

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from institutional_quant.agents import (
    DeepSeekResponsesClient,
    EvidencePacketBuilder,
    ResearchGraph,
    StructuredResponse,
    _strict_json_schema,
)
from institutional_quant.alpaca import AlpacaPaperClient
from institutional_quant.api import _bearish_research_candidates, _prepare_decision, create_app
from institutional_quant.config import Settings
from institutional_quant.rebalance import build_rebalance_advice
from institutional_quant.schemas import (
    AnalystView,
    BacktestResult,
    BacktestSpec,
    ConsensusDecision,
    DebateTurn,
    EvidenceClaim,
    EvidenceItem,
    EvidencePacket,
    PaperTarget,
    PortfolioPosition,
    PortfolioRecommendation,
    Rating,
    StrategyMetrics,
)
from institutional_quant.storage import DuckDBStore


class FakeStructuredClient:
    def __init__(self):
        self.calls = 0

    async def structured(self, *, role, output_type, **kwargs):
        self.calls += 1
        reference = "src:revenue"
        if output_type is AnalystView:
            value = AnalystView(
                role=role,
                stance_score=0.2,
                summary="Evidence-grounded view",
                claims=[EvidenceClaim(text="Revenue is available", evidence_refs=[reference])],
            )
        elif output_type is DebateTurn:
            value = DebateTurn(
                speaker="bull" if role.startswith("bull") else "bear",
                round_number=int(role[-1]),
                stance_score=0.2,
                argument="Grounded case",
                evidence_refs=[reference],
            )
        else:
            value = ConsensusDecision(
                company_id="C1",
                ticker="ONE",
                as_of_date=date(2026, 8, 31),
                rating=Rating.HOLD,
                score_adjustment=0.02,
                summary="Balanced evidence",
                supporting_evidence=[reference],
                dissent="Residual uncertainty",
                uncertainties=["Limited history"],
                analyst_median_score=0.2,
                model_alias="fake",
                prompt_version="fake",
                evidence_hash="fake",
            )
        return StructuredResponse(value, "fake-v1", "fingerprint", 10, 5, 3.0)


def packet() -> EvidencePacket:
    return EvidencePacket(
        company_id="C1",
        ticker="ONE",
        sector="Industrials",
        as_of_date=date(2026, 8, 31),
        factor_score=0.4,
        ml_score=0.5,
        ensemble_score=0.45,
        evidence=[
            EvidenceItem(
                evidence_id="src:revenue",
                label="Revenue",
                value=100,
                unit="USD",
                effective_at=datetime(2026, 8, 1),
                source_file_id="src",
                field="revenue",
            )
        ],
    )


def test_deepseek_strict_schema_requires_all_nested_properties() -> None:
    schema = _strict_json_schema(AnalystView.model_json_schema())

    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    claim = schema["$defs"]["EvidenceClaim"]
    assert set(claim["required"]) == set(claim["properties"])
    assert claim["additionalProperties"] is False


def test_prepared_decision_separates_long_bearish_and_neutral_ratings() -> None:
    base = {
        "ticker": "TEST",
        "summary": "Grounded conclusion",
        "supporting_evidence": ["src:revenue"],
    }

    assert _prepare_decision({**base, "rating": "Overweight"})["debate_bucket"] == "long"
    assert _prepare_decision({**base, "rating": "Sell"})["debate_bucket"] == "bearish"
    assert _prepare_decision({**base, "rating": "Underweight"})["debate_bucket"] == "bearish"
    assert _prepare_decision({**base, "rating": "Hold"})["debate_bucket"] == "neutral"


def test_bearish_research_screen_requires_weak_factor_and_multiple_adverse_signals(
    monkeypatch,
) -> None:
    class FakeFactors:
        def __init__(self, _store):
            pass

        def snapshot(self, _as_of_date):
            return pd.DataFrame(
                [
                    {
                        "company_id": "bear",
                        "ticker": "BEAR",
                        "factor_score": -2.0,
                        "price_to_earnings": 100.0,
                        "revenue_growth": -0.2,
                        "eps_revision_1m": -0.1,
                        "volatility_252d": 0.8,
                    },
                    {"company_id": "one", "ticker": "ONE", "factor_score": 0.0, "price_to_earnings": 10.0, "volatility_252d": 0.2},
                    {"company_id": "two", "ticker": "TWO", "factor_score": 1.0, "price_to_earnings": 11.0, "volatility_252d": 0.21},
                    {"company_id": "three", "ticker": "THREE", "factor_score": 2.0, "price_to_earnings": 12.0, "volatility_252d": 0.22},
                ]
            )

    class FakeStore:
        def query_df(self, *_args, **_kwargs):
            return pd.DataFrame()

    monkeypatch.setattr("institutional_quant.api.FactorEngine", FakeFactors)

    screened = _bearish_research_candidates(FakeStore(), date(2026, 9, 1))

    assert screened["ticker"].tolist() == ["BEAR"]
    assert screened.iloc[0]["bearish_signal_count"] >= 2


def test_responses_client_ignores_reasoning_text() -> None:
    payload = {
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "hidden work"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": '{"ok":true}'}],
            },
        ]
    }

    assert DeepSeekResponsesClient._output_text(payload) == '{"ok":true}'


def test_responses_client_retries_semantic_validation_once() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        summary = "invalid" if calls == 1 else "valid"
        body = {
            "role": "fundamental",
            "stance_score": 0.0,
            "summary": summary,
            "claims": [{"text": "Grounded claim", "evidence_refs": ["src:revenue"]}],
            "catalysts": [],
            "risks": [],
        }
        return httpx.Response(
            200,
            json={
                "model": "fake-v1",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": json.dumps(body)}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = DeepSeekResponsesClient(
        Settings(database_backend="duckdb", deepseek_api_key="test"),
        transport=httpx.MockTransport(handler),
    )

    def validator(value: AnalystView) -> None:
        if value.summary != "valid":
            raise ValueError("semantic validation failed")

    result = asyncio.run(
        client.structured(
            role="fundamental",
            system_prompt="test",
            user_payload={},
            output_type=AnalystView,
            model="fake",
            reasoning_effort="high",
            validator=validator,
        )
    )

    assert result.value.summary == "valid"
    assert calls == 2


def test_evidence_packet_excludes_future_snapshot_date(tmp_path) -> None:
    store = DuckDBStore(tmp_path / "evidence.duckdb")
    store.initialize()
    store.insert_frame(
        "fundamentals",
        pd.DataFrame(
            [
                {
                    "company_id": "C1",
                    "ticker": "ONE",
                    "period_end": date(2026, 6, 30),
                    "period_type": "Quarterly",
                    "effective_at": datetime(2026, 8, 1),
                    "as_of_date": date(2026, 8, 31),
                    "metric": "revenue",
                    "value": 100.0,
                    "unit": "USD",
                    "source_file_id": "visible",
                    "ingested_at": datetime(2026, 9, 1),
                },
                {
                    "company_id": "C1",
                    "ticker": "ONE",
                    "period_end": date(2026, 6, 30),
                    "period_type": "Quarterly",
                    "effective_at": datetime(2026, 8, 1),
                    "as_of_date": date(2026, 9, 2),
                    "metric": "future_metric",
                    "value": 999.0,
                    "unit": "USD",
                    "source_file_id": "future",
                    "ingested_at": datetime(2026, 9, 2),
                },
            ]
        ),
    )
    builder = EvidencePacketBuilder(store)
    builder.factors = SimpleNamespace(
        snapshot=lambda _as_of: pd.DataFrame(
            [
                {
                    "company_id": "C1",
                    "ticker": "ONE",
                    "sector": "Industrials",
                    "factor_score": 0.2,
                }
            ]
        )
    )

    result = builder.build("C1", date(2026, 9, 1))

    assert [item.field for item in result.evidence] == ["revenue"]


def test_langgraph_debate_and_cache(tmp_path) -> None:
    store = DuckDBStore(tmp_path / "agent.duckdb")
    store.initialize()
    fake = FakeStructuredClient()
    settings = Settings(database_backend="duckdb", deepseek_api_key="test")
    graph = ResearchGraph(store, settings, client=fake)
    decision = asyncio.run(graph.run(packet(), with_debate=True))
    assert decision.rating is Rating.HOLD
    assert decision.evidence_hash == packet().provenance_hash
    assert fake.calls == 9
    asyncio.run(graph.run(packet(), with_debate=True))
    assert fake.calls == 9


def test_dashboard_and_health_use_duckdb_fixture(tmp_path) -> None:
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api.duckdb",
        raw_data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        page = client.get("/data-status")
        assert health.status_code == 200
        assert health.json()["paper_trading_only"] is True
        assert page.status_code == 200
        assert "Data Status" in page.text
        assert "Five-year research gate is not ready" in page.text
        assert "Upload a Capital IQ export" in page.text
        assert 'action="/api/v1/imports/ciq"' in page.text
        assert "Automatic import" in page.text


def test_debate_page_defaults_to_long_only_and_exposes_bearish_research(tmp_path) -> None:
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api-debate.duckdb",
        raw_data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
    )
    with TestClient(create_app(settings)) as client:
        page = client.get("/debate")

    assert page.status_code == 200
    assert "Long only" in page.text
    assert "Run bearish screen" in page.text
    assert "not short orders" in page.text
    assert "News is not inferred" in page.text


def test_bearish_screen_requires_explicit_external_processing_gate(tmp_path) -> None:
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api-bearish-gate.duckdb",
        raw_data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/debate/bearish-screen")

    assert response.status_code == 403
    assert "CIQ_EXTERNAL_PROCESSING_CONFIRMED" in response.json()["detail"]


def test_current_market_returns_upload_infers_missing_snapshot_timestamps(tmp_path) -> None:
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api-import.duckdb",
        raw_data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
    )
    export = (
        "Entity ID,Ticker,Price Change (%),Price Change (%).1,Price Change (%).2\n"
        "C1,AAA,1.0,2.0,3.0\n"
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/imports/ciq",
            data={"dataset": "auto"},
            files={
                "file": (
                    "ciq_sp500_returns_1d_1w_1m_2026-09-03.csv",
                    export,
                    "text/csv",
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["imported_rows"] == 1


def test_held_portfolio_is_rendered_as_reference_not_a_trade_recommendation(tmp_path) -> None:
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api-portfolio.duckdb",
        raw_data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
    )
    with TestClient(create_app(settings)) as client:
        client.app.state.store.save_portfolio(
            PortfolioRecommendation(
                as_of_date=date(2026, 9, 3),
                cadence="weekly",
                one_way_turnover=0.0,
                status="held",
                warnings=["Optimizer infeasible; previous target retained."],
                positions=[
                    PortfolioPosition(
                        company_id="C1",
                        ticker="ONE",
                        sector="Unknown",
                        weight=0.05,
                        score=0.0,
                    )
                ],
            )
        )
        page = client.get("/portfolio")

    assert page.status_code == 200
    assert "No new portfolio action is recommended" in page.text
    assert "Prior model target (reference only)" in page.text
    assert "Not recalculated" in page.text


def test_backtest_page_explains_crossing_lines_with_risk_metrics(tmp_path) -> None:
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api-backtest.duckdb",
        raw_data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
    )
    metric_defaults = {
        "annualized_volatility": 0.14,
        "sortino_zero_rf": 1.1,
        "beta": 0.9,
        "information_ratio": 0.3,
        "average_one_way_turnover": 0.08,
        "monthly_hit_rate": 0.6,
        "observations": 3,
    }
    with TestClient(create_app(settings)) as client:
        client.app.state.store.save_backtest(
            BacktestResult(
                spec=BacktestSpec(transaction_cost_bps=10),
                certified_point_in_time=True,
                certification_notes=[],
                metrics=[
                    StrategyMetrics(strategy="SPY buy-and-hold", cagr=0.10, sharpe_zero_rf=0.7, max_drawdown=-0.20, **metric_defaults),
                    StrategyMetrics(strategy="factor_only", cagr=0.14, sharpe_zero_rf=0.9, max_drawdown=-0.24, **metric_defaults),
                    StrategyMetrics(strategy="ml_only", cagr=0.12, sharpe_zero_rf=1.1, max_drawdown=-0.16, **metric_defaults),
                ],
                monthly_returns=[
                    {"date": "2026-01-31", "spy": 0.01, "factor_only": 0.03, "ml_only": 0.02},
                    {"date": "2026-02-28", "spy": -0.01, "factor_only": 0.02, "ml_only": -0.01},
                    {"date": "2026-03-31", "spy": 0.02, "factor_only": 0.01, "ml_only": 0.03},
                ],
                factor_ic=[],
                statistical_tests=[
                    {"strategy": "factor_only", "annualized_alpha": 0.04, "alpha_t_stat": 1.4},
                    {"strategy": "ml_only", "annualized_alpha": 0.02, "alpha_t_stat": 0.9},
                ],
            )
        )
        page = client.get("/backtest")

    assert page.status_code == 200
    assert "Conclusion summary" in page.text
    assert "No single strategy dominated" in page.text
    assert "Strategy comparison" in page.text
    assert "Lines crossing is normal" in page.text
    assert "Factor-only" in page.text


def test_my_holdings_keeps_watchlist_and_paper_sync_read_only(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={"status": "ACTIVE", "equity": "12000", "cash": "4000"})
        if request.url.path == "/v2/positions":
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "AAPL",
                        "qty": "3",
                        "current_price": "200",
                        "market_value": "600",
                        "unrealized_pl": "20",
                    }
                ],
            )
        if request.url.path == "/v2/orders":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api-holdings.duckdb",
        raw_data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
        alpaca_paper_key="key",
        alpaca_paper_secret="secret",
    )
    with TestClient(create_app(settings)) as client:
        page = client.get("/my-holdings")
        assert page.status_code == 200
        assert "Model targets and brokerage holdings are separate" in page.text
        assert "No brokerage data loaded" in page.text
        assert "Generate advice from synced holdings" in page.text
        assert "Research proposal · paper only · no order authority" in page.text
        assert "Risk reductions from current holdings" in page.text
        assert "New and additional research candidates" in page.text

        created = client.post("/api/v1/watchlist", json={"ticker": "aapl", "note": "Review earnings"})
        assert created.status_code == 201
        assert created.json()["item"]["ticker"] == "AAPL"
        updated = client.post("/api/v1/watchlist", json={"ticker": "AAPL", "note": "Review guidance"})
        assert updated.status_code == 201
        assert client.get("/api/v1/watchlist").json()["items"] == [
            {"ticker": "AAPL", "note": "Review guidance", "created_at": created.json()["item"]["created_at"]}
        ]

        client.app.state.alpaca = AlpacaPaperClient(
            settings, transport=httpx.MockTransport(handler)
        )
        synchronized = client.get("/api/v1/holdings/alpaca-paper")
        assert synchronized.status_code == 200
        assert synchronized.json()["paper_only"] is True
        assert synchronized.json()["positions"][0]["symbol"] == "AAPL"

        deleted = client.delete("/api/v1/watchlist/AAPL")
        assert deleted.status_code == 200
        assert client.get("/api/v1/watchlist").json()["items"] == []


def test_rebalance_advice_stages_concentration_reduction_and_explains_evidence() -> None:
    ranked = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "sector": "Information Technology",
                "operational_score": 0.91,
                "factor_score": 1.2,
                "ml_score": 0.7,
                "factor_value": 0.3,
                "factor_quality": 0.8,
                "price_to_earnings": 20.0,
                "roic": 0.23,
            },
            {
                "ticker": "MSFT",
                "sector": "Information Technology",
                "operational_score": 0.35,
                "factor_score": 0.2,
                "ml_score": 0.2,
                "factor_value": -0.1,
            },
        ]
    )
    target = pd.DataFrame(
        [
            {"company_id": "AAPL", "ticker": "AAPL", "sector": "Information Technology", "weight": 0.05, "score": 0.91},
            {"company_id": "MSFT", "ticker": "MSFT", "sector": "Information Technology", "weight": 0.05, "score": 0.35},
        ]
    )
    advice = build_rebalance_advice(
        ranked,
        target,
        [{"symbol": "MSFT", "qty": 8.0, "current_price": 100.0, "market_value": 800.0}],
        1_000.0,
        {"AAPL": 50.0, "MSFT": 100.0},
        as_of_date=date(2026, 9, 1),
        reference_price_date="2026-09-01",
    )

    reductions = [action for action in advice["actions"] if action["action"] == "Reduce"]
    additions = [action for action in advice["actions"] if action["action"] == "Initiate"]
    assert reductions[0]["ticker"] == "MSFT"
    assert reductions[0]["proposed_notional"] <= 200.0
    assert reductions[0]["risk_limit_weight"] == 0.025
    assert reductions[0]["risk_flags"]
    assert reductions[0]["risk_details"][0]["calculation"]
    assert reductions[0]["risk_details"][0]["benchmark"] == "Portfolio policy maximum: 5% per name."
    assert additions[0]["ticker"] == "AAPL"
    assert additions[0]["proposed_shares"] == 1.0
    assert {item["label"] for item in additions[0]["evidence"]} >= {"P/E", "ROIC"}
    roic = next(item for item in additions[0]["evidence"] if item["label"] == "ROIC")
    assert roic["calculation"] == "Reported return on invested capital."
    assert roic["source"] == "Certified point-in-time fundamental input."
    assert roic["benchmark"]
    assert roic["confidence"] == "Moderate"
    assert roic["available_at"] == "2026-09-01"
    assert advice["max_name_weight"] == 0.05
    assert any("not an order" in warning for warning in advice["warnings"])
    assert advice["risk_reductions"] == reductions
    assert advice["purchase_candidates"] == additions


def test_unselected_holding_is_not_reduced_without_a_portfolio_risk_trigger() -> None:
    ranked = pd.DataFrame(
        [
            {"ticker": "KEEP", "sector": "Utilities", "operational_score": 0.9, "factor_score": 0.9, "ml_score": 0.8},
            {"ticker": "BUY", "sector": "Health Care", "operational_score": 0.8, "factor_score": 0.8, "ml_score": 0.7},
        ]
    )
    target = pd.DataFrame(
        [{"company_id": "BUY", "ticker": "BUY", "sector": "Health Care", "weight": 0.05, "score": 0.8}]
    )
    advice = build_rebalance_advice(
        ranked,
        target,
        [{"symbol": "KEEP", "qty": 4.0, "current_price": 10.0, "market_value": 40.0}],
        1_000.0,
        {"KEEP": 10.0, "BUY": 20.0},
        as_of_date=date(2026, 9, 1),
        reference_price_date="2026-09-01",
    )

    assert advice["risk_reductions"] == []
    assert not any(action["ticker"] == "KEEP" for action in advice["actions"])


def test_holding_without_a_current_score_is_not_assumed_to_be_a_sell_signal() -> None:
    ranked = pd.DataFrame(
        [
            {
                "ticker": "BUY",
                "sector": "Health Care",
                "rank": 1,
                "operational_score": 0.9,
                "factor_score": 1.0,
                "ml_score": 0.8,
            }
        ]
    )
    target = pd.DataFrame(
        [
            {
                "company_id": "BUY",
                "ticker": "BUY",
                "sector": "Health Care",
                "weight": 0.05,
                "score": 0.9,
            }
        ]
    )

    advice = build_rebalance_advice(
        ranked,
        target,
        [{"symbol": "UNSCORED", "qty": 20.0, "current_price": 20.0, "market_value": 400.0}],
        10_000.0,
        {"BUY": 50.0, "UNSCORED": 20.0},
        as_of_date=date(2026, 9, 1),
        reference_price_date="2026-09-01",
    )

    assert advice["risk_reductions"] == []
    assert not any(action["ticker"] == "UNSCORED" for action in advice["actions"])


def test_rebalance_advice_endpoint_uses_submitted_snapshot_without_broker_call(tmp_path, monkeypatch) -> None:
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "api-advice.duckdb",
        raw_data_dir=tmp_path / "raw",
    )

    def fake_advice(store, positions, equity):
        assert positions == [{"symbol": "AAPL", "qty": 2.0, "current_price": 100.0, "market_value": 200.0}]
        assert equity == 1_000.0
        return {"as_of_date": "2026-09-01", "actions": [], "warnings": []}

    monkeypatch.setattr("institutional_quant.api.generate_rebalance_advice", fake_advice)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/holdings/rebalance-advice",
            json={
                "equity": 1_000,
                "positions": [{"symbol": "aapl", "qty": 2, "current_price": 100, "market_value": 200}],
            },
        )

    assert response.status_code == 200
    assert response.json()["as_of_date"] == "2026-09-01"


def test_alpaca_preview_requires_unchanged_explicit_approval() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={"equity": "100000"})
        if request.url.path == "/v2/positions":
            return httpx.Response(200, json=[])
        if request.url.path == "/v2/stocks/trades/latest":
            return httpx.Response(200, json={"trades": {"AAPL": {"p": 200}}})
        if request.url.path == "/v2/orders":
            payload = json.loads(request.content)
            return httpx.Response(200, json={"id": "paper-order", **payload})
        return httpx.Response(404)

    settings = Settings(
        database_backend="duckdb", alpaca_paper_key="key", alpaca_paper_secret="secret"
    )
    client = AlpacaPaperClient(settings, transport=httpx.MockTransport(handler))
    previews = asyncio.run(client.preview([PaperTarget(symbol="AAPL", target_weight=0.05)]))
    assert previews[0].estimated_notional == 5000
    try:
        asyncio.run(client.submit(previews, approved=False))
    except ValueError as exc:
        assert "approval" in str(exc)
    else:
        raise AssertionError("Submission without approval must fail")
    orders = asyncio.run(client.submit(previews, approved=True))
    assert orders[0]["id"] == "paper-order"
