from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .backtest import BacktestEngine, derive_cost_sensitivity
from .reports import write_backtest_report
from .schemas import BacktestResult, BacktestSpec
from .storage import Store


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CaseStudyRunner:
    """Freezes primary/sensitivity/Agent-ablation results into one manifest."""

    def __init__(self, store: Store, output_dir: Path):
        self.store = store
        self.output_dir = output_dir

    def run(self, spec: BacktestSpec | None = None) -> dict[str, Any]:
        primary_spec = (spec or BacktestSpec()).model_copy(
            update={"transaction_cost_bps": 10.0, "agent_overlay": False}
        )
        primary = BacktestEngine(self.store).run(primary_spec)
        results: list[BacktestResult] = [
            derive_cost_sensitivity(primary, 5.0),
            primary,
            derive_cost_sensitivity(primary, 25.0),
        ]
        for result in results:
            self.store.save_backtest(result)
            write_backtest_report(result, self.output_dir)

        ablations: list[BacktestResult] = []
        coverage = self.store.query_df(
            """
            SELECT variant, COUNT(DISTINCT as_of_date) AS months
            FROM agent_study_decisions GROUP BY variant
            """
        )
        months = {row["variant"]: int(row["months"]) for row in coverage.to_dict(orient="records")}
        for variant in ("without_debate", "with_debate"):
            if months.get(variant, 0) >= 24:
                result = BacktestEngine(self.store).run(
                    primary_spec.model_copy(
                        update={"agent_overlay": True, "agent_variant": variant}
                    )
                )
                ablations.append(result)
                write_backtest_report(result, self.output_dir)

        sources = self.store.query_df(
            "SELECT dataset, original_name, sha256, row_count, imported_at FROM source_files ORDER BY dataset, sha256"
        ).to_dict(orient="records")
        fingerprints = self.store.query_df(
            """
            SELECT DISTINCT model_alias, model_version, system_fingerprint,
                            reasoning_effort, prompt_version
            FROM agent_cache ORDER BY model_alias, model_version, reasoning_effort
            """
        ).to_dict(orient="records")
        code_files = sorted(Path("institutional_quant").rglob("*.py")) + [
            Path("pyproject.toml"),
            Path("uv.lock"),
        ]
        code_hash = hashlib.sha256(
            "".join(f"{path}:{_sha256(path)}\n" for path in code_files if path.exists()).encode()
        ).hexdigest()
        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "code_hash": code_hash,
            "source_files": sources,
            "model_fingerprints": fingerprints,
            "primary_backtest_id": primary.backtest_id,
            "cost_sensitivity": {
                str(result.spec.transaction_cost_bps): result.backtest_id for result in results
            },
            "agent_ablations": {
                result.spec.agent_variant: result.backtest_id for result in ablations
            },
            "synthetic": any(str(row["original_name"]).startswith("synthetic_") for row in sources),
        }
        manifest_directory = self.output_dir / "manifests"
        manifest_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_directory / f"case-study-{primary.backtest_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return manifest
