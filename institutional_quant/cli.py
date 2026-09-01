from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from dotenv import load_dotenv

from .backtest import BacktestEngine
from .case_study import CaseStudyRunner
from .config import Settings
from .demo import build_synthetic_demo
from .factors import FactorEngine
from .index_data import PublicSP500MembershipSync
from .ingestion import CapitalIQImporter, historical_universe_gaps
from .market_data import AlpacaHistoricalPriceSync, YahooHistoricalPriceSync
from .preflight import run_preflight
from .reports import write_backtest_report
from .schemas import BacktestSpec, DatasetKind
from .storage import create_store

app = typer.Typer(help="Institutional multi-agent quant platform")


@app.command()
def serve() -> None:
    """Start the localhost FastAPI and HTMX dashboard."""
    load_dotenv()
    settings = Settings.from_env()
    uvicorn.run(
        "institutional_quant.api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=False,
    )


@app.command("init-db")
def init_db() -> None:
    """Create the isolated institutional_quant schema in Supabase."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    typer.echo(f"Initialized {settings.database_backend}")


@app.command()
def preflight(live: bool = False) -> None:
    """Check configuration without printing secrets; --live tests configured services."""
    load_dotenv()
    settings = Settings.from_env()
    checks = run_preflight(settings, live=live)
    for check in checks:
        typer.echo(f"{'PASS' if check.ready else 'WAIT'}  {check.name}: {check.detail}")


@app.command("import-ciq")
def import_ciq(
    dataset: DatasetKind,
    path: Path,
    current_snapshot_as_of: Annotated[
        str | None,
        typer.Option(help="Explicit as-of date for a non-historical current snapshot"),
    ] = None,
    current_snapshot_effective_at: Annotated[
        str | None,
        typer.Option(help="Observed-at timestamp for a non-historical current snapshot"),
    ] = None,
) -> None:
    """Validate, archive and ingest one Capital IQ CSV/XLSX export."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    parsed_as_of = date.fromisoformat(current_snapshot_as_of) if current_snapshot_as_of else None
    parsed_effective_at = (
        datetime.fromisoformat(current_snapshot_effective_at)
        if current_snapshot_effective_at
        else None
    )
    result = CapitalIQImporter(store, settings).import_file(
        path,
        dataset,
        current_snapshot_as_of=parsed_as_of,
        current_snapshot_effective_at=parsed_effective_at,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("factor-snapshot")
def factor_snapshot(
    as_of: Annotated[str, typer.Option("--as-of", help="Research cut-off date")],
    top: int = 25,
    persist: bool = False,
) -> None:
    """Calculate an auditable factor cross-section for one research date."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    engine = FactorEngine(store)
    snapshot = engine.snapshot(date.fromisoformat(as_of))
    eligible = snapshot.dropna(subset=["factor_score"]).copy()
    if persist and not eligible.empty:
        engine.persist(eligible)
    family_columns = [
        "factor_value",
        "factor_quality",
        "factor_growth",
        "factor_revisions",
        "factor_momentum",
        "factor_low_risk",
    ]
    rankings = eligible.nlargest(top, "factor_score")
    ranking_columns = [
        "company_id",
        "ticker",
        "company_name",
        "sector",
        "factor_score",
        *family_columns,
    ]
    ranking_records = (
        rankings[ranking_columns]
        .astype(object)
        .where(rankings[ranking_columns].notna(), None)
        .to_dict(orient="records")
    )
    typer.echo(
        json.dumps(
            {
                "as_of_date": as_of,
                "companies": len(snapshot),
                "investable": len(eligible),
                "persisted": bool(persist and not eligible.empty),
                "family_coverage": {
                    column.removeprefix("factor_"): int(snapshot[column].notna().sum())
                    for column in family_columns
                },
                "top": ranking_records,
            },
            indent=2,
            default=str,
            allow_nan=False,
        )
    )


@app.command("synthetic-demo")
def synthetic_demo(
    database: Path = Path("data/demo/institutional_quant.duckdb"),
    raw_dir: Path = Path("data/demo/raw"),
) -> None:
    """Create a licensed-data-free local fixture database."""
    store = build_synthetic_demo(database, raw_dir)
    typer.echo(json.dumps(store.source_status(), default=str, indent=2))


@app.command("sync-alpaca-prices")
def sync_alpaca_prices(
    start: str = "2021-09-01",
    end: str = "2026-08-31",
    batch_size: int = 50,
) -> None:
    """Download immutable, fully adjusted Alpaca IEX bars into the local price lake."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    result = asyncio.run(
        AlpacaHistoricalPriceSync(store, settings).sync(
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            batch_size=batch_size,
        )
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("sync-yahoo-prices")
def sync_yahoo_prices(
    start: str = "2017-04-01",
    end: str = "2021-09-01",
    concurrency: int = 8,
) -> None:
    """Download immutable adjusted Yahoo chart bars as a labelled substitute."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    result = asyncio.run(
        YahooHistoricalPriceSync(store, settings).sync(
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            concurrency=concurrency,
        )
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("sync-public-sp500-membership")
def sync_public_sp500_membership(
    start: str = "2021-09-01",
    end: str = "2026-08-31",
    refresh_identities: Annotated[
        bool,
        typer.Option(
            "--refresh-identities",
            help="Rebuild public intervals after importing new Capital IQ company IDs",
        ),
    ] = False,
) -> None:
    """Load a pinned, explicitly non-CIQ point-in-time S&P 500 reconstruction."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    result = PublicSP500MembershipSync(store, settings).sync(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        refresh_identity_map=refresh_identities,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("ciq-gap-manifest")
def ciq_gap_manifest(
    start: str = "2021-09-01",
    end: str = "2026-08-31",
    output: Path = Path("data/exports/ciq/historical_universe_gap_manifest.csv"),
) -> None:
    """Export historical constituents that still need Capital IQ coverage."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    gaps = historical_universe_gaps(
        store,
        date.fromisoformat(start),
        date.fromisoformat(end),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(output, index=False)
    typer.echo(
        json.dumps(
            {
                "output": str(output.resolve()),
                "rows": len(gaps),
                "unique_tickers": int(gaps["ticker"].nunique()) if not gaps.empty else 0,
                "needs_fundamentals": int(gaps["needs_fundamentals"].sum())
                if not gaps.empty
                else 0,
                "needs_estimates": int(gaps["needs_estimates"].sum()) if not gaps.empty else 0,
                "needs_instrument_identity": int(gaps["needs_instrument_identity"].sum())
                if not gaps.empty
                else 0,
            },
            indent=2,
        )
    )


@app.command("backtest-demo")
def backtest_demo(database: Path = Path("data/demo/institutional_quant.duckdb")) -> None:
    """Run the full 2021-09 through 2026-08 walk-forward synthetic study."""
    settings = Settings(database_backend="duckdb", database_path=database)
    store = create_store(settings)
    store.initialize()
    result = BacktestEngine(store).run(BacktestSpec())
    write_backtest_report(result, Path("output"))
    typer.echo(result.model_dump_json(indent=2))


@app.command("case-study")
def case_study() -> None:
    """Run and freeze primary, 5/25 bps sensitivity, and available Agent ablations."""
    load_dotenv()
    settings = Settings.from_env()
    store = create_store(settings)
    store.initialize()
    manifest = CaseStudyRunner(store, settings.report_dir).run()
    typer.echo(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    app()
