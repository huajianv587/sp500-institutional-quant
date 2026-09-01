from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time
from typing import Any, TypedDict, TypeVar

import httpx
import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from .config import Settings
from .factors import FactorEngine
from .schemas import (
    AnalystView,
    ConsensusDecision,
    DebateTurn,
    EvidenceItem,
    EvidencePacket,
    Rating,
)
from .storage import Store

PROMPT_VERSION = "institutional-research-v1"
T = TypeVar("T", bound=BaseModel)


class ConsensusDraft(BaseModel):
    rating: Rating
    score_adjustment: float
    summary: str
    supporting_evidence: list[str]
    dissent: str
    uncertainties: list[str]
    analyst_median_score: float


class ResearchState(TypedDict, total=False):
    packet: EvidencePacket
    with_debate: bool
    cache_namespace: str
    analysts: list[AnalystView]
    turns: list[DebateTurn]
    decision: ConsensusDecision


@dataclass
class StructuredResponse:
    value: BaseModel
    model_version: str | None
    system_fingerprint: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float


class DeepSeekResponsesClient:
    """Strict structured-output client with one malformed-output retry."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for live Agent runs")
        self.settings = settings
        self.transport = transport

    @staticmethod
    def _output_text(payload: dict[str, Any]) -> str:
        if payload.get("output_text"):
            return str(payload["output_text"])
        pieces: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("text"):
                    pieces.append(str(content["text"]))
        if pieces:
            return "".join(pieces)
        choices = payload.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
        raise ValueError("DeepSeek returned no final structured output")

    async def structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        output_type: type[T],
        model: str,
        reasoning_effort: str,
    ) -> StructuredResponse:
        schema = output_type.model_json_schema()
        request = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "input": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, separators=(",", ":"), default=str),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": output_type.__name__,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        errors: list[str] = []
        async with httpx.AsyncClient(
            base_url=self.settings.deepseek_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
            timeout=180,
            transport=self.transport,
        ) as client:
            for _attempt in range(2):
                started = time.perf_counter()
                response = await client.post("/responses", json=request)
                response.raise_for_status()
                payload = response.json()
                latency = (time.perf_counter() - started) * 1000
                try:
                    parsed = output_type.model_validate_json(self._output_text(payload))
                    usage = payload.get("usage", {})
                    return StructuredResponse(
                        value=parsed,
                        model_version=payload.get("model"),
                        system_fingerprint=payload.get("system_fingerprint"),
                        input_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0))),
                        output_tokens=int(
                            usage.get("output_tokens", usage.get("completion_tokens", 0))
                        ),
                        latency_ms=latency,
                    )
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(str(exc))
                    request["input"].append(
                        {
                            "role": "user",
                            "content": "Return one complete JSON object that exactly satisfies the schema.",
                        }
                    )
        raise ValueError(
            f"Invalid structured output after one retry for {role}: {' | '.join(errors)}"
        )


class EvidencePacketBuilder:
    def __init__(self, store: Store):
        self.store = store
        self.factors = FactorEngine(store)

    def build(
        self,
        company_id: str,
        as_of_date: date,
        ml_score: float = 0.0,
        ensemble_score: float | None = None,
    ) -> EvidencePacket:
        snapshot = self.factors.snapshot(as_of_date)
        company = snapshot.loc[snapshot["company_id"].astype(str) == str(company_id)]
        if company.empty:
            raise ValueError(f"{company_id} is not in the point-in-time universe")
        row = company.iloc[0]
        cutoff = datetime.combine(as_of_date, datetime_time.max)
        evidence = self.store.query_df(
            """
            WITH observations AS (
                SELECT company_id, metric AS field, value, unit, effective_at, source_file_id,
                       ROW_NUMBER() OVER (PARTITION BY company_id, metric ORDER BY period_end DESC, effective_at DESC) AS rn
                FROM fundamentals WHERE company_id = ? AND effective_at <= ?
                UNION ALL
                SELECT company_id, metric AS field, value, unit, effective_at, source_file_id,
                       ROW_NUMBER() OVER (PARTITION BY company_id, metric ORDER BY fiscal_period DESC, effective_at DESC) AS rn
                FROM estimates WHERE company_id = ? AND effective_at <= ?
            )
            SELECT field, value, unit, effective_at, source_file_id
            FROM observations WHERE rn = 1 ORDER BY field
            """,
            [company_id, cutoff, company_id, cutoff],
        )
        items = [
            EvidenceItem(
                evidence_id=f"{source.source_file_id}:{source.field}",
                label=str(source.field).replace("_", " ").title(),
                value=None if pd.isna(source.value) else float(source.value),
                unit=None if pd.isna(source.unit) else str(source.unit),
                effective_at=source.effective_at,
                source_file_id=str(source.source_file_id),
                field=str(source.field),
            )
            for source in evidence.itertuples()
        ]
        if not items:
            raise ValueError("No attributable fundamentals or estimates are available")
        factor_score = float(row["factor_score"])
        return EvidencePacket(
            company_id=str(company_id),
            ticker=str(row["ticker"]),
            sector=str(row["sector"]),
            as_of_date=as_of_date,
            factor_score=factor_score,
            ml_score=float(ml_score),
            ensemble_score=(
                float(ensemble_score)
                if ensemble_score is not None
                else 0.5 * factor_score + 0.5 * float(ml_score)
            ),
            evidence=items,
        )


class ResearchGraph:
    analyst_roles = {
        "fundamental": "Assess financial quality, growth durability, cash conversion, and balance-sheet strength.",
        "valuation": "Assess valuation using only supplied inputs. Do not invent prices, multiples, or forecasts.",
        "estimates_peer": "Assess estimate direction, surprises, and peer-relative evidence.",
        "risk": "Assess downside, cyclicality, leverage, estimate risk, and data uncertainty.",
    }

    def __init__(
        self,
        store: Store,
        settings: Settings,
        client: DeepSeekResponsesClient | None = None,
        analyst_reasoning: str = "high",
        decision_reasoning: str = "max",
        apply_benchmark_routing: bool = True,
    ):
        self.store = store
        selected = store.list_model_benchmarks(20) if apply_benchmark_routing else []
        decision_route = next((item for item in selected if item.selected_for == "decision"), None)
        supporting_route = next(
            (item for item in selected if item.selected_for == "supporting"), None
        )
        self.settings = settings.model_copy(
            update={
                "deepseek_analyst_model": (
                    supporting_route.model if supporting_route else settings.deepseek_analyst_model
                ),
                "deepseek_decision_model": (
                    decision_route.model if decision_route else settings.deepseek_decision_model
                ),
            }
        )
        self.client = client or DeepSeekResponsesClient(settings)
        self.analyst_reasoning = (
            supporting_route.reasoning_effort if supporting_route else analyst_reasoning
        )
        self.decision_reasoning = (
            decision_route.reasoning_effort if decision_route else decision_reasoning
        )
        builder = StateGraph(ResearchState)
        builder.add_node("independent_analysts", self._analyst_node)
        builder.add_node("bull_bear_debate", self._debate_node)
        builder.add_node("consensus_judge", self._judge_node)
        builder.add_edge(START, "independent_analysts")
        builder.add_conditional_edges(
            "independent_analysts",
            lambda state: "bull_bear_debate" if state["with_debate"] else "consensus_judge",
            {"bull_bear_debate": "bull_bear_debate", "consensus_judge": "consensus_judge"},
        )
        builder.add_edge("bull_bear_debate", "consensus_judge")
        builder.add_edge("consensus_judge", END)
        self.graph = builder.compile()

    @staticmethod
    def _cache_key(
        role: str,
        model: str,
        reasoning: str,
        packet: EvidencePacket,
        cache_namespace: str = "",
    ) -> str:
        return hashlib.sha256(
            f"{cache_namespace}|{role}|{model}|{reasoning}|{PROMPT_VERSION}|{packet.provenance_hash}".encode()
        ).hexdigest()

    @staticmethod
    def _validate_references(model: BaseModel, packet: EvidencePacket) -> None:
        valid = {item.evidence_id for item in packet.evidence}
        document = model.model_dump(mode="json")
        references: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    if key in {"evidence_refs", "supporting_evidence"}:
                        references.extend(nested)
                    else:
                        walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)

        walk(document)
        unsupported = sorted(set(references) - valid)
        if unsupported:
            raise ValueError(f"Unsupported evidence references: {unsupported}")

    async def _cached_call(
        self,
        *,
        role: str,
        prompt: str,
        payload: dict[str, Any],
        output_type: type[T],
        model: str,
        reasoning: str,
        packet: EvidencePacket,
        cache_namespace: str = "",
    ) -> T:
        cache_key = self._cache_key(role, model, reasoning, packet, cache_namespace)
        cached = self.store.cache_get(cache_key)
        if cached:
            raw = cached["response_json"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            return output_type.model_validate(raw)
        response = await self.client.structured(
            role=role,
            system_prompt=prompt,
            user_payload=payload,
            output_type=output_type,
            model=model,
            reasoning_effort=reasoning,
        )
        self._validate_references(response.value, packet)
        self.store.cache_put(
            {
                "cache_key": cache_key,
                "role": role,
                "model_alias": model,
                "model_version": response.model_version,
                "system_fingerprint": response.system_fingerprint,
                "reasoning_effort": reasoning,
                "prompt_version": PROMPT_VERSION,
                "evidence_hash": packet.provenance_hash,
                "response_json": response.value.model_dump(mode="json"),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
            }
        )
        return response.value  # type: ignore[return-value]

    async def _analyst_node(self, state: ResearchState) -> ResearchState:
        packet = state["packet"]
        base = packet.model_dump(mode="json")
        common = (
            "You are an institutional equity researcher. Use only the supplied EvidencePacket. "
            "Every factual or numerical claim must cite valid evidence IDs. State uncertainty; never infer missing data."
        )
        analysts = await asyncio.gather(
            *[
                self._cached_call(
                    role=role,
                    prompt=f"{common} {instruction}",
                    payload=base,
                    output_type=AnalystView,
                    model=self.settings.deepseek_analyst_model,
                    reasoning=self.analyst_reasoning if role != "risk" else self.decision_reasoning,
                    packet=packet,
                    cache_namespace=state.get("cache_namespace", ""),
                )
                for role, instruction in self.analyst_roles.items()
            ]
        )
        return {"analysts": list(analysts)}

    async def _debate_node(self, state: ResearchState) -> ResearchState:
        packet = state["packet"]
        base = packet.model_dump(mode="json")
        common = (
            "You are an institutional equity researcher. Use only the supplied EvidencePacket. "
            "Every factual or numerical claim must cite valid evidence IDs. State uncertainty; never infer missing data."
        )
        turns: list[DebateTurn] = []
        for round_number in (1, 2):
            previous = [turn.model_dump(mode="json") for turn in turns]
            round_turns = await asyncio.gather(
                *[
                    self._cached_call(
                        role=f"{speaker}_round_{round_number}",
                        prompt=(
                            f"{common} Argue the strongest evidence-grounded {speaker} case. "
                            f"This is debate round {round_number}; address the opposing case when present."
                        ),
                        payload={
                            "packet": base,
                            "analysts": [
                                view.model_dump(mode="json") for view in state["analysts"]
                            ],
                            "prior_turns": previous,
                        },
                        output_type=DebateTurn,
                        model=self.settings.deepseek_decision_model,
                        reasoning=self.decision_reasoning,
                        packet=packet,
                        cache_namespace=state.get("cache_namespace", ""),
                    )
                    for speaker in ("bull", "bear")
                ]
            )
            turns.extend(round_turns)
        return {"turns": turns}

    async def _judge_node(self, state: ResearchState) -> ResearchState:
        packet = state["packet"]
        base = packet.model_dump(mode="json")
        common = (
            "You are an institutional equity researcher. Use only the supplied EvidencePacket. "
            "Every factual or numerical claim must cite valid evidence IDs. State uncertainty; never infer missing data."
        )
        turns = state.get("turns", [])
        consensus_role = (
            "consensus_with_debate" if state["with_debate"] else "consensus_without_debate"
        )
        draft = await self._cached_call(
            role=consensus_role,
            prompt=(
                f"{common} Act as an independent consensus judge. Debate rhetoric cannot outweigh evidence. "
                "The score_adjustment is bounded to plus or minus 0.10. Preserve dissent and uncertainties."
            ),
            payload={
                "packet": base,
                "analysts": [view.model_dump(mode="json") for view in state["analysts"]],
                "debate": [turn.model_dump(mode="json") for turn in turns],
            },
            output_type=ConsensusDraft,
            model=self.settings.deepseek_decision_model,
            reasoning=self.decision_reasoning,
            packet=packet,
            cache_namespace=state.get("cache_namespace", ""),
        )
        metadata = (
            self.store.cache_get(
                self._cache_key(
                    consensus_role,
                    self.settings.deepseek_decision_model,
                    self.decision_reasoning,
                    packet,
                    state.get("cache_namespace", ""),
                )
            )
            or {}
        )
        decision = ConsensusDecision(
            company_id=packet.company_id,
            ticker=packet.ticker,
            as_of_date=packet.as_of_date,
            rating=draft.rating,
            score_adjustment=float(np.clip(draft.score_adjustment, -0.10, 0.10)),
            summary=draft.summary,
            supporting_evidence=draft.supporting_evidence,
            dissent=draft.dissent,
            uncertainties=draft.uncertainties,
            analyst_median_score=draft.analyst_median_score,
            model_alias=self.settings.deepseek_decision_model,
            model_version=metadata.get("model_version"),
            system_fingerprint=metadata.get("system_fingerprint"),
            prompt_version=PROMPT_VERSION,
            evidence_hash=packet.provenance_hash,
        )
        self.store.save_consensus(decision)
        return {"decision": decision}

    async def run(
        self, packet: EvidencePacket, with_debate: bool = True, cache_namespace: str = ""
    ) -> ConsensusDecision:
        state = await self.graph.ainvoke(
            {"packet": packet, "with_debate": with_debate, "cache_namespace": cache_namespace}
        )
        return state["decision"]


def contains_uncited_number(text: str) -> bool:
    """Used by benchmark/tests to flag unsupported numerical prose."""
    return bool(re.search(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", text))
