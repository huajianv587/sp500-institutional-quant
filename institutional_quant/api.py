from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import httpx
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .agent_study import AgentStudyRunner
from .agents import EvidencePacketBuilder, ResearchGraph
from .alpaca import AlpacaPaperClient
from .backtest import BacktestEngine
from .benchmark import ModelBenchmark
from .config import Settings
from .factors import FactorEngine
from .ingestion import CapitalIQImporter, certify_point_in_time
from .jobs import JobManager
from .operations import (
    prepare_one_share_order,
    run_daily,
    run_full_cycle,
    run_monthly,
    run_weekly,
)
from .portfolio import PortfolioOptimizer
from .reports import write_backtest_report
from .scheduler import OperationalScheduler
from .schemas import (
    BacktestSpec,
    DatasetKind,
    JobRecord,
    ModelBenchmarkRequest,
    OperationRequest,
    PaperOrderPreviewRequest,
    PaperOrderSubmitRequest,
    ResearchRunRequest,
)
from .storage import Store, create_store

PACKAGE_DIR = Path(__file__).parent


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_EVIDENCE_TOKEN = re.compile(r"src_[0-9a-z]+:[a-z0-9_]+", re.IGNORECASE)
_EVIDENCE_BLOCK = re.compile(
    r"\[(?:\s*src_[^,\]\s]+:[^,\]\s]+\s*,?)+\]", re.IGNORECASE
)


def _field_label(field: str) -> str:
    """Turn an internal field name into a compact, reader-facing label."""
    labels = {
        "price_to_earnings": "P/E",
        "price_to_book": "P/B",
        "tev_ebitda": "TEV / EBITDA",
        "ebitda": "EBITDA",
        "ebitda_margin": "EBITDA Margin",
        "eps_estimate": "EPS Estimate",
        "eps_analyst_count_1m": "EPS Analysts (1M)",
        "eps_up_revisions_1m": "EPS Up Revisions (1M)",
        "eps_down_revisions_1m": "EPS Down Revisions (1M)",
        "eps_up_revisions_3m": "EPS Up Revisions (3M)",
        "eps_down_revisions_3m": "EPS Down Revisions (3M)",
        "roic": "ROIC",
        "return_on_equity": "ROE",
        "return_on_assets": "ROA",
        "sp_norm_eps_act_or_est": "S&P Norm EPS",
    }
    return labels.get(field, field.replace("_", " ").strip().title())


def _prepare_decision(value: Any) -> Any:
    """Add a clean display summary while preserving the audit evidence IDs."""
    if not isinstance(value, dict):
        return value

    evidence = value.get("supporting_evidence") or []
    evidence_refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in evidence:
        raw_text = str(raw).strip()
        source_id, separator, field = raw_text.partition(":")
        if not separator or not source_id or not field or raw_text in seen:
            continue
        seen.add(raw_text)
        evidence_refs.append(
            {
                "raw": raw_text,
                "source_id": source_id,
                "source_short": f"{source_id[:14]}…" if len(source_id) > 14 else source_id,
                "field": field,
                "label": _field_label(field),
            }
        )

    summary = str(value.get("summary") or "").strip()
    # Citation blocks are useful for audit but make the primary narrative hard to read.
    display_summary = _EVIDENCE_BLOCK.sub("", summary)
    display_summary = _EVIDENCE_TOKEN.sub("", display_summary)
    display_summary = re.sub(r"\[\s*,?\s*\]", "", display_summary)
    display_summary = re.sub(r"\s{2,}", " ", display_summary)
    display_summary = re.sub(r"\s+\)", ")", display_summary)
    display_summary = re.sub(r"\s+([,.;])", r"\1", display_summary).strip(" ,")

    prepared = dict(value)
    prepared["summary_display"] = display_summary or summary
    prepared["evidence_refs"] = evidence_refs
    prepared["evidence_count"] = len(evidence)
    return prepared


def _research_candidates(store: Store, as_of_date: date, requested: list[str]) -> pd.DataFrame:
    snapshot = FactorEngine(store).snapshot(as_of_date)
    snapshot["ml_score"] = 0.0
    previous = store.query_df(
        """
        WITH latest AS (
          SELECT MAX(as_of_date) AS as_of_date FROM factor_observations WHERE as_of_date <= ?
        )
        SELECT company_id, ml_score FROM factor_observations
        WHERE as_of_date = (SELECT as_of_date FROM latest)
        """,
        [as_of_date],
    )
    if not previous.empty:
        snapshot = snapshot.drop(columns="ml_score").merge(previous, on="company_id", how="left")
        snapshot["ml_score"] = snapshot["ml_score"].fillna(0.0)
    snapshot["ensemble_score"] = (
        0.5 * snapshot["factor_score"].rank(pct=True) + 0.5 * snapshot["ml_score"]
    )
    if requested:
        chosen = snapshot.loc[snapshot["company_id"].astype(str).isin(set(requested))]
        missing = sorted(set(requested) - set(chosen["company_id"].astype(str)))
        if missing:
            raise ValueError(f"Companies are not in the point-in-time universe: {missing}")
        return chosen
    current = store.current_portfolio()
    holdings = {position.company_id for position in current.positions} if current else set()
    top = snapshot.nlargest(5, "ensemble_score")
    deteriorating = snapshot.loc[snapshot["company_id"].isin(holdings)].nsmallest(
        3, "ensemble_score"
    )
    snapshot["disagreement"] = abs(
        snapshot["factor_score"].rank(pct=True) - snapshot["ml_score"].rank(pct=True)
    )
    disagreements = snapshot.nlargest(3, "disagreement")
    return pd.concat([top, deteriorating, disagreements]).drop_duplicates("company_id").head(10)


def _build_portfolio(store: Store, as_of_date: date, decisions: list[dict[str, Any]]):
    universe = FactorEngine(store).snapshot(as_of_date)
    universe["ensemble_score"] = universe["factor_score"].rank(pct=True)
    adjustment = {row["company_id"]: float(row["score_adjustment"]) for row in decisions}
    universe["agent_adjusted_score"] = universe.apply(
        lambda row: float(row["ensemble_score"]) + adjustment.get(str(row["company_id"]), 0.0),
        axis=1,
    )
    start = as_of_date - timedelta(days=430)
    prices = store.load_prices(start, as_of_date)
    if not prices.empty:
        cutoff = pd.Timestamp(datetime.combine(as_of_date, datetime.max.time()))
        prices = prices.loc[pd.to_datetime(prices["effective_at"]) <= cutoff][
            ["ticker", "price_date", "adjusted_close"]
        ].sort_values("price_date")
    history = (
        prices.pivot_table(
            index="price_date", columns="ticker", values="adjusted_close", aggfunc="last"
        )
        .pct_change()
        .tail(252)
    )
    current = store.current_portfolio()
    current_weights = (
        {position.ticker: position.weight for position in current.positions} if current else {}
    )
    recommendation = PortfolioOptimizer().optimize(
        universe,
        history,
        as_of_date,
        score_column="agent_adjusted_score",
        current_weights=current_weights,
        benchmark_sector_weights=universe["sector"].value_counts(normalize=True).to_dict(),
    )
    store.save_portfolio(recommendation)
    return recommendation


def create_app(settings: Settings | None = None) -> FastAPI:
    load_dotenv()
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    store = create_store(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store.initialize()
        application.state.store = store
        application.state.settings = settings
        application.state.jobs = JobManager(store)
        application.state.alpaca = AlpacaPaperClient(settings)
        application.state.scheduler = OperationalScheduler(store)
        if settings.enable_scheduler:
            application.state.scheduler.start()
        yield
        application.state.scheduler.shutdown()

    app = FastAPI(
        title="S&P 500 Institutional Multi-Agent Quant Platform",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")

    @app.get("/health")
    async def health():
        return {"status": "ok", "database": settings.database_backend, "paper_trading_only": True}

    @app.post("/api/v1/imports/ciq")
    async def import_ciq(
        file: Annotated[UploadFile, File()],
        dataset: Annotated[str, Form()] = "auto",
        current_snapshot_as_of: Annotated[date | None, Form()] = None,
        current_snapshot_effective_at: Annotated[datetime | None, Form()] = None,
        current_snapshot_timestamp_provenance: Annotated[str | None, Form()] = None,
    ):
        suffix = Path(file.filename or "export.csv").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            shutil.copyfileobj(file.file, temporary)
            path = Path(temporary.name)
        try:
            requested_dataset = dataset.strip().lower()
            resolved_dataset = (
                CapitalIQImporter.detect_dataset(path, file.filename)
                if requested_dataset == "auto"
                else DatasetKind(requested_dataset)
            )
            timestamp_provenance = current_snapshot_timestamp_provenance
            if resolved_dataset is DatasetKind.MARKET_RETURNS:
                filename = file.filename or ""
                inferred_as_of = False
                inferred_effective_at = False
                if current_snapshot_as_of is None:
                    match = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", filename)
                    if match:
                        current_snapshot_as_of = date(
                            int(match.group(1)), int(match.group(2)), int(match.group(3))
                        )
                        inferred_as_of = True
                if current_snapshot_effective_at is None:
                    current_snapshot_effective_at = datetime.now().astimezone().replace(tzinfo=None)
                    inferred_effective_at = True
                if inferred_as_of and inferred_effective_at:
                    timestamp_provenance = "server_inferred_filename_date_and_upload_timestamp"
                elif inferred_as_of:
                    timestamp_provenance = "server_inferred_filename_date"
                elif inferred_effective_at:
                    timestamp_provenance = "server_inferred_upload_timestamp"
            result = CapitalIQImporter(store, settings).import_file(
                path,
                resolved_dataset,
                current_snapshot_as_of=current_snapshot_as_of,
                current_snapshot_effective_at=current_snapshot_effective_at,
                current_snapshot_timestamp_provenance=timestamp_provenance,
            )
            return result
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)

    @app.get("/api/v1/data-quality")
    async def data_quality():
        certified, notes = certify_point_in_time(
            store, date(2021, 9, 1), min(date.today(), date(2026, 8, 31))
        )
        return {
            "certified_point_in_time": certified,
            "certification_notes": notes,
            "sources": store.source_status(),
            "issues": store.list_issues(),
            "gates": {
                "external_model_processing": settings.ciq_external_processing_confirmed,
                "cloud_storage": settings.ciq_cloud_storage_confirmed,
            },
        }

    @app.post("/api/v1/model-benchmarks", status_code=202)
    async def run_benchmark(request: ModelBenchmarkRequest):
        if not settings.ciq_external_processing_confirmed:
            raise HTTPException(
                status_code=403, detail="CIQ_EXTERNAL_PROCESSING_CONFIRMED must be true"
            )

        async def work(job: JobRecord) -> str:
            job.message = "Running frozen model cases"
            store.upsert_job(job)
            results = await ModelBenchmark(store, settings).evaluate(
                request.packets, request.configurations, request.repeats
            )
            return results[0].benchmark_id

        return app.state.jobs.submit("model_benchmark", work)

    @app.post("/api/v1/research-runs", status_code=202)
    async def research_run(request: ResearchRunRequest):
        if not settings.ciq_external_processing_confirmed:
            raise HTTPException(
                status_code=403, detail="CIQ_EXTERNAL_PROCESSING_CONFIRMED must be true"
            )

        async def work(job: JobRecord) -> str:
            if request.historical_months:

                def progress(value: float, message: str) -> None:
                    job.progress = value
                    job.message = message
                    store.upsert_job(job)

                result = await AgentStudyRunner(store, settings).run(
                    request.historical_months, progress
                )
                stored = {
                    "run_id": job.job_id,
                    "as_of_date": request.as_of_date.isoformat(),
                    "agent_study": result,
                }
                store.save_research_run(job.job_id, request.as_of_date, stored)
                return job.job_id
            selected = _research_candidates(store, request.as_of_date, request.company_ids)
            builder = EvidencePacketBuilder(store)
            graph = ResearchGraph(store, settings)
            payloads = []
            for index, row in enumerate(selected.itertuples(), start=1):
                packet = builder.build(str(row.company_id), request.as_of_date, float(row.ml_score))
                decision = await graph.run(packet, with_debate=request.with_debate)
                payloads.append(
                    {
                        "packet": packet.model_dump(mode="json"),
                        "decision": decision.model_dump(mode="json"),
                    }
                )
                job.progress = 0.1 + 0.7 * index / len(selected)
                job.message = f"Completed {index}/{len(selected)} research cases"
                store.upsert_job(job)
            decisions = [payload["decision"] for payload in payloads]
            portfolio = await asyncio.to_thread(
                _build_portfolio, store, request.as_of_date, decisions
            )
            result = {
                "run_id": job.job_id,
                "as_of_date": request.as_of_date.isoformat(),
                "with_debate": request.with_debate,
                "cases": payloads,
                "portfolio": portfolio.model_dump(mode="json"),
            }
            store.save_research_run(job.job_id, request.as_of_date, result)
            return job.job_id

        return app.state.jobs.submit("research_run", work)

    @app.get("/api/v1/research-runs/{run_id}")
    async def get_research_run(run_id: str):
        result = store.get_research_run(run_id)
        if result is None:
            job = store.get_job(run_id)
            if job:
                return job
            raise HTTPException(status_code=404, detail="Research run not found")
        return result

    @app.post("/api/v1/backtests", status_code=202)
    async def run_backtest(spec: BacktestSpec):
        async def work(job: JobRecord) -> str:
            result = await asyncio.to_thread(BacktestEngine(store).run, spec)
            await asyncio.to_thread(write_backtest_report, result, settings.report_dir)
            return result.backtest_id

        return app.state.jobs.submit("backtest", work)

    @app.get("/api/v1/backtests/{backtest_id}")
    async def get_backtest(backtest_id: str):
        result = store.get_backtest(backtest_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return result

    @app.get("/api/v1/portfolio/current")
    async def current_portfolio():
        portfolio = store.current_portfolio()
        if portfolio is None:
            raise HTTPException(status_code=404, detail="No portfolio recommendation exists")
        return portfolio

    @app.post("/api/v1/paper/orders/preview")
    async def paper_preview(request: PaperOrderPreviewRequest):
        try:
            return await app.state.alpaca.preview(request.targets)
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/paper/orders/submit")
    async def paper_submit(request: PaperOrderSubmitRequest):
        try:
            return await app.state.alpaca.submit(request.orders, request.approved)
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/paper/orders/demo-preview")
    async def paper_demo_preview():
        try:
            return await prepare_one_share_order(store, settings, app.state.alpaca)
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/paper/sync")
    async def paper_sync():
        try:
            return await app.state.alpaca.synchronize()
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def submit_operation(kind: str, request: OperationRequest):
        async def work(job: JobRecord) -> str:
            if kind == "daily":
                result = await run_daily(store, settings, request.as_of_date, client=app.state.alpaca)
            elif kind == "weekly":
                result = await run_weekly(store, settings, request.as_of_date)
            elif kind == "monthly":
                result = await run_monthly(store, settings, request.as_of_date)
            else:
                result = await run_full_cycle(
                    store,
                    settings,
                    app.state.alpaca,
                    submit_paper_order=False,
                    as_of_date=request.as_of_date,
                )
            job.message = result.message
            job.progress = 1.0
            store.upsert_job(job)
            return json.dumps(result.model_dump(mode="json"), default=str)

        return app.state.jobs.submit(f"operation_{kind}", work)

    @app.post("/api/v1/operations/daily", status_code=202)
    async def operation_daily(request: OperationRequest | None = None):
        request = request or OperationRequest()
        return submit_operation("daily", request)

    @app.post("/api/v1/operations/weekly", status_code=202)
    async def operation_weekly(request: OperationRequest | None = None):
        request = request or OperationRequest()
        return submit_operation("weekly", request)

    @app.post("/api/v1/operations/monthly", status_code=202)
    async def operation_monthly(request: OperationRequest | None = None):
        request = request or OperationRequest()
        return submit_operation("monthly", request)

    @app.post("/api/v1/operations/full-cycle", status_code=202)
    async def operation_full_cycle(request: OperationRequest | None = None):
        request = request or OperationRequest()
        return submit_operation("full-cycle", request)

    @app.get("/api/v1/operations/{job_id}")
    async def get_operation(job_id: str):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Operation not found")
        payload = job.model_dump(mode="json")
        if job.result_ref:
            try:
                payload["operation"] = json.loads(job.result_ref)
            except json.JSONDecodeError:
                payload["operation"] = {"result_ref": job.result_ref}
        return payload

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def page_context(slug: str) -> dict[str, Any]:
        context: dict[str, Any] = {
            "slug": slug,
            "database_backend": settings.database_backend,
            "paper_only": True,
            "sources": [],
            "issues": [],
            "portfolio": None,
            "backtests": [],
            "benchmarks": [],
            "research_runs": [],
            "decisions": [],
            "factors": [],
            "factor_readiness": None,
            "certified": False,
            "certification_notes": [],
            "chart": "",
        }
        if slug == "data-status":
            context["sources"] = store.source_status()
            context["issues"] = store.list_issues(25)
            certified, notes = certify_point_in_time(
                store, date(2021, 9, 1), min(date.today(), date(2026, 8, 31))
            )
            context["certified"] = certified
            context["certification_notes"] = notes
        elif slug == "factor-lab":
            available_factor_dates = [
                value
                for value in (
                    store.latest_available_date("prices"),
                    store.latest_available_date("fundamentals"),
                    store.latest_available_date("estimates"),
                )
                if value is not None
            ]
            factor_date = (
                min(max(available_factor_dates), date.today())
                if available_factor_dates
                else None
            )
            if factor_date:
                try:
                    snapshot = FactorEngine(store).snapshot(factor_date)
                    family_columns = [
                        "factor_value",
                        "factor_quality",
                        "factor_growth",
                        "factor_revisions",
                        "factor_momentum",
                        "factor_low_risk",
                    ]
                    context["factor_readiness"] = {
                        "as_of_date": factor_date,
                        "companies": len(snapshot),
                        "investable": int(snapshot["factor_score"].notna().sum()),
                        "family_coverage": {
                            column.removeprefix("factor_"): int(snapshot[column].notna().sum())
                            for column in family_columns
                        },
                    }
                    factors = snapshot.dropna(subset=["factor_score"]).nlargest(
                        25, "factor_score"
                    )
                    context["factors"] = factors[
                        ["ticker", "sector", "factor_score"]
                    ].to_dict(orient="records")
                except ValueError:
                    pass
        elif slug == "research-runs":
            runs = store.query_df(
                "SELECT run_id, as_of_date, created_at FROM research_runs "
                "ORDER BY created_at DESC LIMIT 20"
            )
            context["research_runs"] = runs.to_dict(orient="records")
            context["benchmarks"] = store.list_model_benchmarks(10)
        elif slug == "debate":
            decisions = store.query_df(
                "SELECT decision_json FROM consensus_decisions "
                "ORDER BY created_at DESC LIMIT 20"
            )
            context["decisions"] = [
                _prepare_decision(_json_value(value))
                for value in decisions.get("decision_json", [])
            ]
        elif slug == "portfolio":
            context["portfolio"] = store.current_portfolio()
        elif slug == "backtest":
            context["backtests"] = store.list_backtests(10)
        if slug == "backtest" and context["backtests"]:
            latest = context["backtests"][0]
            frame = pd.DataFrame(latest.monthly_returns)
            if not frame.empty:
                figure = go.Figure()
                for column in [
                    "spy",
                    "equal_weight_universe",
                    "factor_only",
                    "ml_only",
                    "factor_ml_ensemble",
                ]:
                    if column in frame:
                        figure.add_trace(
                            go.Scatter(
                                x=frame["date"], y=(1 + frame[column]).cumprod(), name=column
                            )
                        )
                figure.update_layout(
                    template="plotly_white",
                    height=380,
                    margin={"l": 30, "r": 20, "t": 30, "b": 30},
                    yaxis_title="Growth of $1",
                )
                context["chart"] = figure.to_html(full_html=False, include_plotlyjs=True)
        return context

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        context = await asyncio.to_thread(page_context, "data-status")
        return templates.TemplateResponse(
            request=request, name="page.html", context=context
        )

    for path, slug in [
        ("/data-status", "data-status"),
        ("/factor-lab", "factor-lab"),
        ("/research-runs", "research-runs"),
        ("/debate", "debate"),
        ("/portfolio", "portfolio"),
        ("/backtest", "backtest"),
        ("/paper-trading", "paper-trading"),
    ]:

        async def render(request: Request, page_slug: str = slug):
            context = await asyncio.to_thread(page_context, page_slug)
            return templates.TemplateResponse(
                request=request, name="page.html", context=context
            )

        app.add_api_route(path, render, response_class=HTMLResponse, methods=["GET"], name=slug)

    return app
