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
from institutional_quant.api import create_app
from institutional_quant.config import Settings
from institutional_quant.schemas import (
    AnalystView,
    ConsensusDecision,
    DebateTurn,
    EvidenceClaim,
    EvidenceItem,
    EvidencePacket,
    PaperTarget,
    Rating,
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
