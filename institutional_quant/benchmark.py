from __future__ import annotations

from collections import Counter
from statistics import median

from .agents import ResearchGraph
from .config import Settings
from .schemas import EvidencePacket, ModelBenchmarkResult, ModelConfig
from .storage import Store


class ModelBenchmark:
    """Quality-first frozen-case benchmark; cost and latency never outrank correctness."""

    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings

    async def evaluate(
        self,
        packets: list[EvidencePacket],
        configurations: list[ModelConfig],
        repeats: int = 3,
    ) -> list[ModelBenchmarkResult]:
        if not packets:
            raise ValueError("At least one frozen EvidencePacket is required")
        results: list[ModelBenchmarkResult] = []
        for configuration in configurations:
            routed = self.settings.model_copy(
                update={
                    "deepseek_analyst_model": configuration.model,
                    "deepseek_decision_model": configuration.model,
                }
            )
            graph = ResearchGraph(
                self.store,
                routed,
                analyst_reasoning=configuration.reasoning_effort,
                decision_reasoning=configuration.reasoning_effort,
                apply_benchmark_routing=False,
            )
            successful = 0
            reference_total = 0
            reference_valid = 0
            ratings: dict[str, list[str]] = {}
            unsupported = 0
            numerical = 0
            attempts = 0
            before = self.store.query_df(
                "SELECT COALESCE(SUM(input_tokens),0) AS i, COALESCE(SUM(output_tokens),0) AS o FROM agent_cache"
            ).iloc[0]
            for packet in packets:
                valid_ids = {item.evidence_id for item in packet.evidence}
                for repeat in range(repeats):
                    attempts += 1
                    try:
                        decision = await graph.run(
                            packet,
                            with_debate=True,
                            cache_namespace=f"benchmark:{configuration.model}:{configuration.reasoning_effort}:{repeat}",
                        )
                    except Exception:
                        continue
                    successful += 1
                    ratings.setdefault(packet.provenance_hash, []).append(decision.rating.value)
                    reference_total += len(decision.supporting_evidence)
                    reference_valid += sum(ref in valid_ids for ref in decision.supporting_evidence)
                    unsupported += sum(ref not in valid_ids for ref in decision.supporting_evidence)
                    numerical += int(-0.10 <= decision.score_adjustment <= 0.10)
            cache = self.store.query_df(
                "SELECT latency_ms FROM agent_cache WHERE model_alias = ? AND reasoning_effort = ?",
                [configuration.model, configuration.reasoning_effort],
            )
            latencies = (
                cache["latency_ms"].dropna().astype(float).tolist() if not cache.empty else []
            )
            after = self.store.query_df(
                "SELECT COALESCE(SUM(input_tokens),0) AS i, COALESCE(SUM(output_tokens),0) AS o FROM agent_cache"
            ).iloc[0]
            stability = []
            for values in ratings.values():
                count = Counter(values).most_common(1)[0][1]
                stability.append(count / len(values))
            result = ModelBenchmarkResult(
                model=configuration.model,
                reasoning_effort=configuration.reasoning_effort,
                cases=len(packets),
                schema_success_rate=successful / attempts if attempts else 0,
                evidence_coverage=reference_valid / reference_total if reference_total else 0,
                unsupported_claim_rate=unsupported / reference_total if reference_total else 0,
                rating_stability=sum(stability) / len(stability) if stability else 0,
                numerical_consistency=numerical / successful if successful else 0,
                median_latency_ms=float(median(latencies)) if latencies else 0,
                input_tokens=max(0, int(after["i"] - before["i"])),
                output_tokens=max(0, int(after["o"] - before["o"])),
                role_scope=configuration.role_scope,
            )
            results.append(result)

        def quality(item: ModelBenchmarkResult) -> float:
            return (
                item.schema_success_rate * 0.25
                + item.evidence_coverage * 0.25
                + (1 - item.unsupported_claim_rate) * 0.20
                + item.rating_stability * 0.15
                + item.numerical_consistency * 0.15
            )

        decision_candidates = [item for item in results if item.role_scope == "decision"] or results
        decision = max(decision_candidates, key=quality)
        decision.selected = True
        decision.selected_for = "decision"
        supporting_candidates = [item for item in results if item.role_scope == "supporting"]
        if supporting_candidates:
            best_supporting_quality = max(map(quality, supporting_candidates))
            near_best = [
                item
                for item in supporting_candidates
                if quality(item) >= best_supporting_quality - 0.02
            ]
            supporting = min(
                near_best,
                key=lambda item: (
                    item.input_tokens + item.output_tokens,
                    item.median_latency_ms,
                ),
            )
            supporting.selected = True
            supporting.selected_for = "supporting"
        for result in results:
            self.store.save_model_benchmark(result)
        return results
