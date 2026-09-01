from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import duckdb
import numpy as np
import pandas as pd

from .schemas import (
    BacktestResult,
    ConsensusDecision,
    DataQualityIssue,
    JobRecord,
    PortfolioRecommendation,
)

PRICE_COLUMNS = [
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


class ParquetPriceLake:
    """Derived local analytical cache for high-volume daily market data."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    @staticmethod
    def _quoted_path(path: Path) -> str:
        return str(path).replace("'", "''")

    def persist(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        incoming = frame[PRICE_COLUMNS].drop_duplicates(
            ["company_id", "price_date", "source"], keep="last"
        )
        temporary = self.path.with_suffix(".tmp.parquet")
        with self._lock, duckdb.connect() as connection:
            connection.execute("PRAGMA threads=4")
            connection.register("incoming_prices", incoming)
            existing = (
                f"SELECT * FROM read_parquet('{self._quoted_path(self.path)}') UNION ALL "
                if self.path.exists()
                else ""
            )
            connection.execute(
                f"""
                COPY (
                    SELECT {",".join(PRICE_COLUMNS)} FROM (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY company_id, price_date, source
                            ORDER BY ingested_at DESC, source_file_id DESC
                        ) AS rn
                        FROM ({existing}SELECT * FROM incoming_prices)
                    ) WHERE rn = 1
                ) TO '{self._quoted_path(temporary)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            connection.unregister("incoming_prices")
        temporary.replace(self.path)

    def load(
        self,
        start: date,
        end: date,
        *,
        company_ids: list[str] | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=PRICE_COLUMNS)
        filters = ["price_date BETWEEN ? AND ?"]
        parameters: list[Any] = [start, end]
        identity_filters: list[str] = []
        if company_ids:
            identity_filters.append("company_id IN (" + ",".join("?" for _ in company_ids) + ")")
            parameters.extend(company_ids)
        if tickers:
            identity_filters.append("ticker IN (" + ",".join("?" for _ in tickers) + ")")
            parameters.extend(tickers)
        if identity_filters:
            filters.append("(" + " OR ".join(identity_filters) + ")")
        with self._lock, duckdb.connect() as connection:
            return connection.execute(
                f"""
                SELECT {",".join(PRICE_COLUMNS)} FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY company_id, price_date
                        ORDER BY CASE source
                            WHEN 'capital_iq' THEN 0
                            WHEN 'yahoo_adjusted' THEN 1
                            WHEN 'alpaca_iex_adjusted' THEN 2
                            ELSE 3
                        END,
                        ingested_at DESC,
                        source_file_id DESC
                    ) AS rn
                    FROM read_parquet(?)
                    WHERE {" AND ".join(filters)}
                ) WHERE rn = 1
                ORDER BY price_date, ticker
                """,
                [str(self.path), *parameters],
            ).df()

    def coverage(self) -> tuple[date | None, date | None]:
        if not self.path.exists():
            return None, None
        with self._lock, duckdb.connect() as connection:
            row = connection.execute(
                "SELECT MIN(price_date), MAX(price_date) FROM read_parquet(?)",
                [str(self.path)],
            ).fetchone()
        return row[0], row[1]

    def sources(self) -> list[str]:
        if not self.path.exists():
            return []
        with self._lock, duckdb.connect() as connection:
            return [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT source FROM read_parquet(?) ORDER BY source",
                    [str(self.path)],
                ).fetchall()
            ]


DDL = r"""
CREATE TABLE IF NOT EXISTS source_files (
    source_file_id VARCHAR PRIMARY KEY,
    dataset VARCHAR NOT NULL,
    original_name VARCHAR NOT NULL,
    archived_path VARCHAR NOT NULL,
    sha256 VARCHAR NOT NULL UNIQUE,
    row_count BIGINT NOT NULL,
    imported_at TIMESTAMP NOT NULL,
    metadata_json JSON
);

CREATE TABLE IF NOT EXISTS data_quality_issues (
    issue_id VARCHAR PRIMARY KEY,
    severity VARCHAR NOT NULL,
    dataset VARCHAR NOT NULL,
    code VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    source_file_id VARCHAR,
    row_number BIGINT,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS instruments (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    company_name VARCHAR NOT NULL,
    sector VARCHAR NOT NULL,
    currency VARCHAR NOT NULL,
    effective_at TIMESTAMP NOT NULL,
    as_of_date DATE NOT NULL,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, effective_at)
);

CREATE TABLE IF NOT EXISTS index_membership (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    index_code VARCHAR NOT NULL,
    member_from DATE NOT NULL,
    member_to DATE,
    effective_at TIMESTAMP NOT NULL,
    as_of_date DATE NOT NULL,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, index_code, member_from)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    period_end DATE NOT NULL,
    period_type VARCHAR NOT NULL,
    effective_at TIMESTAMP NOT NULL,
    as_of_date DATE NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    unit VARCHAR,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, period_end, period_type, effective_at, metric)
);

CREATE TABLE IF NOT EXISTS estimates (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    fiscal_period DATE NOT NULL,
    effective_at TIMESTAMP NOT NULL,
    valid_to TIMESTAMP,
    as_of_date DATE NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    unit VARCHAR,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, fiscal_period, effective_at, metric)
);

CREATE TABLE IF NOT EXISTS prices (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    price_date DATE NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    adjusted_close DOUBLE NOT NULL,
    volume DOUBLE,
    source VARCHAR NOT NULL,
    effective_at TIMESTAMP NOT NULL,
    as_of_date DATE NOT NULL,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, price_date, source)
);

CREATE TABLE IF NOT EXISTS ownership (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    effective_at TIMESTAMP NOT NULL,
    as_of_date DATE NOT NULL,
    institutional_pct DOUBLE,
    institutional_change DOUBLE,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, effective_at)
);

CREATE TABLE IF NOT EXISTS insider_transactions (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    transaction_date DATE NOT NULL,
    effective_at TIMESTAMP NOT NULL,
    as_of_date DATE NOT NULL,
    transaction_type VARCHAR NOT NULL,
    shares DOUBLE NOT NULL,
    value DOUBLE,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, transaction_date, transaction_type, shares)
);

CREATE TABLE IF NOT EXISTS factor_observations (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    sector VARCHAR NOT NULL,
    feature_json JSON NOT NULL,
    factor_score DOUBLE NOT NULL,
    elastic_score DOUBLE,
    tree_score DOUBLE,
    ml_score DOUBLE,
    ensemble_score DOUBLE,
    next_month_excess_return DOUBLE,
    source_snapshot_hash VARCHAR NOT NULL,
    PRIMARY KEY (company_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS agent_cache (
    cache_key VARCHAR PRIMARY KEY,
    role VARCHAR NOT NULL,
    model_alias VARCHAR NOT NULL,
    model_version VARCHAR,
    system_fingerprint VARCHAR,
    reasoning_effort VARCHAR NOT NULL,
    prompt_version VARCHAR NOT NULL,
    evidence_hash VARCHAR NOT NULL,
    response_json JSON NOT NULL,
    input_tokens BIGINT,
    output_tokens BIGINT,
    latency_ms DOUBLE,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS consensus_decisions (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    decision_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id VARCHAR PRIMARY KEY,
    as_of_date DATE NOT NULL,
    result_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_study_decisions (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    variant VARCHAR NOT NULL,
    decision_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, as_of_date, variant)
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id VARCHAR PRIMARY KEY,
    as_of_date DATE NOT NULL,
    cadence VARCHAR NOT NULL,
    recommendation_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_id VARCHAR PRIMARY KEY,
    result_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS model_benchmarks (
    benchmark_id VARCHAR PRIMARY KEY,
    result_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    progress DOUBLE NOT NULL,
    message VARCHAR NOT NULL,
    result_ref VARCHAR,
    error VARCHAR,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
"""


class DuckDBStore:
    """Small single-writer repository around the analytical DuckDB file."""

    def __init__(self, path: str | Path):
        self.is_cloud = False
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def connect(self, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            connection = duckdb.connect(str(self.path), read_only=read_only)
            try:
                yield connection
            finally:
                connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(DDL)

    def execute(self, sql: str, parameters: list[Any] | tuple[Any, ...] | None = None) -> None:
        with self.connect() as connection:
            connection.execute(sql, parameters or [])

    def query_df(
        self, sql: str, parameters: list[Any] | tuple[Any, ...] | None = None
    ) -> pd.DataFrame:
        with self.connect() as connection:
            return connection.execute(sql, parameters or []).df()

    def insert_frame(self, table: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        safe_tables = {
            "instruments",
            "index_membership",
            "fundamentals",
            "estimates",
            "prices",
            "ownership",
            "insider_transactions",
            "factor_observations",
        }
        if table not in safe_tables:
            raise ValueError(f"unsupported insert table: {table}")
        with self.connect() as connection:
            connection.register("incoming_frame", frame)
            connection.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM incoming_frame")
            connection.unregister("incoming_frame")

    def load_prices(
        self,
        start: date,
        end: date,
        *,
        company_ids: list[str] | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        filters = ["price_date BETWEEN ? AND ?"]
        parameters: list[Any] = [start, end]
        identity_filters: list[str] = []
        if company_ids:
            identity_filters.append("company_id IN (" + ",".join("?" for _ in company_ids) + ")")
            parameters.extend(company_ids)
        if tickers:
            identity_filters.append("ticker IN (" + ",".join("?" for _ in tickers) + ")")
            parameters.extend(tickers)
        if identity_filters:
            filters.append("(" + " OR ".join(identity_filters) + ")")
        return self.query_df(
            f"""
            SELECT {",".join(PRICE_COLUMNS)} FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY company_id, price_date
                    ORDER BY CASE source
                        WHEN 'capital_iq' THEN 0
                        WHEN 'yahoo_adjusted' THEN 1
                        WHEN 'alpaca_iex_adjusted' THEN 2
                        ELSE 3
                    END,
                    ingested_at DESC,
                    source_file_id DESC
                ) AS rn
                FROM prices WHERE {" AND ".join(filters)}
            ) WHERE rn = 1
            ORDER BY price_date, ticker
            """,
            parameters,
        )

    def price_coverage(self) -> tuple[date | None, date | None]:
        frame = self.query_df(
            "SELECT MIN(price_date) AS first, MAX(price_date) AS last FROM prices"
        )
        if frame.empty or pd.isna(frame.iloc[0]["first"]):
            return None, None
        return (
            pd.Timestamp(frame.iloc[0]["first"]).date(),
            pd.Timestamp(frame.iloc[0]["last"]).date(),
        )

    def price_sources(self) -> list[str]:
        frame = self.query_df("SELECT DISTINCT source FROM prices ORDER BY source")
        return [str(value) for value in frame.get("source", [])]

    def source_by_hash(self, sha256: str) -> dict[str, Any] | None:
        frame = self.query_df("SELECT * FROM source_files WHERE sha256 = ?", [sha256])
        return None if frame.empty else frame.iloc[0].to_dict()

    def register_source_file(
        self,
        *,
        source_file_id: str,
        dataset: str,
        original_name: str,
        archived_path: str,
        sha256: str,
        row_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                source_file_id,
                dataset,
                original_name,
                archived_path,
                sha256,
                row_count,
                datetime.utcnow(),
                json.dumps(metadata or {}),
            ],
        )

    def record_issue(self, issue: DataQualityIssue) -> None:
        self.execute(
            "INSERT OR REPLACE INTO data_quality_issues VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                issue.issue_id,
                issue.severity.value,
                issue.dataset,
                issue.code,
                issue.message,
                issue.source_file_id,
                issue.row_number,
                issue.created_at,
            ],
        )

    def list_issues(self, limit: int = 200) -> list[dict[str, Any]]:
        frame = self.query_df(
            "SELECT * FROM data_quality_issues ORDER BY created_at DESC LIMIT ?", [limit]
        )
        return frame.to_dict(orient="records")

    def source_status(self) -> list[dict[str, Any]]:
        frame = self.query_df(
            """
            SELECT dataset, COUNT(*) AS files, SUM(row_count) AS rows,
                   MAX(imported_at) AS latest_import
            FROM source_files GROUP BY dataset ORDER BY dataset
            """
        )
        return frame.to_dict(orient="records")

    def upsert_job(self, job: JobRecord) -> None:
        job.updated_at = datetime.utcnow()
        self.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                job.job_id,
                job.kind,
                job.status.value,
                job.progress,
                job.message,
                job.result_ref,
                job.error,
                job.created_at,
                job.updated_at,
            ],
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        frame = self.query_df("SELECT * FROM jobs WHERE job_id = ?", [job_id])
        if frame.empty:
            return None
        return JobRecord.model_validate(frame.iloc[0].to_dict())

    def save_backtest(self, result: BacktestResult) -> None:
        self.execute(
            "INSERT OR REPLACE INTO backtest_runs VALUES (?, ?, ?)",
            [result.backtest_id, result.model_dump_json(), result.created_at],
        )

    def get_backtest(self, backtest_id: str) -> BacktestResult | None:
        frame = self.query_df(
            "SELECT result_json FROM backtest_runs WHERE backtest_id = ?", [backtest_id]
        )
        if frame.empty:
            return None
        return BacktestResult.model_validate_json(str(frame.iloc[0]["result_json"]))

    def list_backtests(self, limit: int = 20) -> list[BacktestResult]:
        frame = self.query_df(
            "SELECT result_json FROM backtest_runs ORDER BY created_at DESC LIMIT ?", [limit]
        )
        return [BacktestResult.model_validate_json(str(value)) for value in frame["result_json"]]

    def save_model_benchmark(self, result) -> None:
        self.execute(
            "INSERT OR REPLACE INTO model_benchmarks VALUES (?, ?, ?)",
            [result.benchmark_id, result.model_dump_json(), result.created_at],
        )

    def list_model_benchmarks(self, limit: int = 20):
        from .schemas import ModelBenchmarkResult

        frame = self.query_df(
            "SELECT result_json FROM model_benchmarks ORDER BY created_at DESC LIMIT ?", [limit]
        )
        return [
            ModelBenchmarkResult.model_validate_json(str(value)) for value in frame["result_json"]
        ]

    def save_portfolio(self, recommendation: PortfolioRecommendation) -> None:
        self.execute(
            "INSERT OR REPLACE INTO portfolios VALUES (?, ?, ?, ?, ?)",
            [
                recommendation.portfolio_id,
                recommendation.as_of_date,
                recommendation.cadence,
                recommendation.model_dump_json(),
                datetime.utcnow(),
            ],
        )

    def current_portfolio(self) -> PortfolioRecommendation | None:
        frame = self.query_df(
            "SELECT recommendation_json FROM portfolios ORDER BY as_of_date DESC, created_at DESC LIMIT 1"
        )
        if frame.empty:
            return None
        return PortfolioRecommendation.model_validate_json(
            str(frame.iloc[0]["recommendation_json"])
        )

    def save_consensus(self, decision: ConsensusDecision) -> None:
        self.execute(
            "INSERT OR REPLACE INTO consensus_decisions VALUES (?, ?, ?, ?, ?)",
            [
                decision.company_id,
                decision.ticker,
                decision.as_of_date,
                decision.model_dump_json(),
                datetime.utcnow(),
            ],
        )

    def save_research_run(self, run_id: str, as_of_date: date, result: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO research_runs VALUES (?, ?, ?, ?)",
            [run_id, as_of_date, json.dumps(result, default=str), datetime.utcnow()],
        )

    def get_research_run(self, run_id: str) -> dict[str, Any] | None:
        frame = self.query_df("SELECT result_json FROM research_runs WHERE run_id = ?", [run_id])
        return None if frame.empty else json.loads(str(frame.iloc[0]["result_json"]))

    def save_agent_study_decision(self, variant: str, decision: ConsensusDecision) -> None:
        self.execute(
            "INSERT OR REPLACE INTO agent_study_decisions VALUES (?, ?, ?, ?, ?, ?)",
            [
                decision.company_id,
                decision.ticker,
                decision.as_of_date,
                variant,
                decision.model_dump_json(),
                datetime.utcnow(),
            ],
        )

    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        frame = self.query_df("SELECT * FROM agent_cache WHERE cache_key = ?", [cache_key])
        return None if frame.empty else frame.iloc[0].to_dict()

    def cache_put(self, values: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT OR REPLACE INTO agent_cache VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                values["cache_key"],
                values["role"],
                values["model_alias"],
                values.get("model_version"),
                values.get("system_fingerprint"),
                values["reasoning_effort"],
                values["prompt_version"],
                values["evidence_hash"],
                json.dumps(values["response_json"]),
                values.get("input_tokens", 0),
                values.get("output_tokens", 0),
                values.get("latency_ms", 0.0),
                values.get("created_at", datetime.utcnow()),
            ],
        )

    def latest_available_date(self, table: str) -> date | None:
        if table == "prices":
            return self.price_coverage()[1]
        safe = {
            "fundamentals": "as_of_date",
            "estimates": "as_of_date",
            "prices": "price_date",
            "index_membership": "as_of_date",
        }
        if table not in safe:
            raise ValueError(table)
        frame = self.query_df(f"SELECT MAX({safe[table]}) AS latest FROM {table}")
        return None if frame.empty or pd.isna(frame.iloc[0]["latest"]) else frame.iloc[0]["latest"]


class Store(Protocol):
    is_cloud: bool

    def initialize(self) -> None: ...
    def execute(self, sql: str, parameters=None) -> None: ...
    def query_df(self, sql: str, parameters=None) -> pd.DataFrame: ...
    def insert_frame(self, table: str, frame: pd.DataFrame) -> None: ...
    def load_prices(
        self,
        start: date,
        end: date,
        *,
        company_ids: list[str] | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame: ...
    def price_coverage(self) -> tuple[date | None, date | None]: ...
    def price_sources(self) -> list[str]: ...
    def source_by_hash(self, sha256: str) -> dict[str, Any] | None: ...
    def register_source_file(self, **kwargs) -> None: ...
    def record_issue(self, issue: DataQualityIssue) -> None: ...
    def list_issues(self, limit: int = 200) -> list[dict[str, Any]]: ...
    def source_status(self) -> list[dict[str, Any]]: ...
    def upsert_job(self, job: JobRecord) -> None: ...
    def get_job(self, job_id: str) -> JobRecord | None: ...
    def save_backtest(self, result: BacktestResult) -> None: ...
    def get_backtest(self, backtest_id: str) -> BacktestResult | None: ...
    def list_backtests(self, limit: int = 20) -> list[BacktestResult]: ...
    def save_model_benchmark(self, result) -> None: ...
    def list_model_benchmarks(self, limit: int = 20): ...
    def save_portfolio(self, recommendation: PortfolioRecommendation) -> None: ...
    def current_portfolio(self) -> PortfolioRecommendation | None: ...
    def save_consensus(self, decision: ConsensusDecision) -> None: ...
    def save_research_run(self, run_id: str, as_of_date: date, result: dict[str, Any]) -> None: ...
    def get_research_run(self, run_id: str) -> dict[str, Any] | None: ...
    def save_agent_study_decision(self, variant: str, decision: ConsensusDecision) -> None: ...
    def cache_get(self, cache_key: str) -> dict[str, Any] | None: ...
    def cache_put(self, values: dict[str, Any]) -> None: ...


POSTGRES_DDL = DDL.replace(" JSON", " JSONB").replace("DOUBLE", "DOUBLE PRECISION")


POSTGRES_CONFLICT_KEYS: dict[str, tuple[str, ...]] = {
    "instruments": ("company_id", "effective_at"),
    "index_membership": ("company_id", "index_code", "member_from"),
    "fundamentals": ("company_id", "period_end", "period_type", "effective_at", "metric"),
    "estimates": ("company_id", "fiscal_period", "effective_at", "metric"),
    "prices": ("company_id", "price_date", "source"),
    "ownership": ("company_id", "effective_at"),
    "insider_transactions": (
        "company_id",
        "transaction_date",
        "transaction_type",
        "shares",
    ),
    "factor_observations": ("company_id", "as_of_date"),
}


class SupabasePostgresStore:
    """Supabase Postgres storage through a direct or session-pooler URL."""

    schema = "institutional_quant"

    def __init__(self, database_url: str, price_lake_path: Path):
        if not database_url.startswith(("postgres://", "postgresql://")):
            raise ValueError("SUPABASE_DB_URL must be a Postgres connection string")
        self.database_url = database_url
        self.price_lake = ParquetPriceLake(price_lake_path)
        self.is_cloud = True
        self._lock = threading.RLock()

    @staticmethod
    def _sql(sql: str) -> str:
        return sql.replace("?", "%s")

    @contextmanager
    def connect(self):
        import psycopg

        with self._lock:
            connection = psycopg.connect(self.database_url, prepare_threshold=None)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(f"SET search_path TO {self.schema}, public")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def initialize(self) -> None:
        import psycopg

        with (
            psycopg.connect(
                self.database_url, prepare_threshold=None, autocommit=True
            ) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cursor.execute(f"SET search_path TO {self.schema}, public")
            cursor.execute(POSTGRES_DDL)

    def execute(self, sql: str, parameters=None) -> None:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(self._sql(sql), parameters or [])

    def query_df(self, sql: str, parameters=None) -> pd.DataFrame:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(self._sql(sql), parameters or [])
            if cursor.description is None:
                return pd.DataFrame()
            columns = [description.name for description in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)

    def insert_frame(self, table: str, frame: pd.DataFrame) -> None:
        from psycopg.types.json import Jsonb

        if frame.empty:
            return
        if table == "prices":
            self.price_lake.persist(frame)
            return
        if table not in POSTGRES_CONFLICT_KEYS:
            raise ValueError(f"unsupported insert table: {table}")
        columns = list(frame.columns)
        conflict = POSTGRES_CONFLICT_KEYS[table]
        frame = frame.drop_duplicates(subset=conflict, keep="last")
        updates = [column for column in columns if column not in conflict]
        column_list = ",".join(columns)
        temp_table = "incoming_frame"
        merge_statement = (
            f"INSERT INTO {table} ({column_list}) "
            f"SELECT {column_list} FROM {temp_table} "
            f"ON CONFLICT ({','.join(conflict)}) DO UPDATE SET "
            + ",".join(f"{column}=EXCLUDED.{column}" for column in updates)
        )
        json_columns = {
            "feature_json",
            "metadata_json",
            "decision_json",
            "recommendation_json",
            "result_json",
            "response_json",
        }

        def converted_rows():
            for row in frame.itertuples(index=False, name=None):
                converted = []
                for column, value in zip(columns, row, strict=True):
                    if column in json_columns:
                        parsed = json.loads(value) if isinstance(value, str) else value
                        converted.append(Jsonb(parsed))
                    elif value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
                        converted.append(None)
                    elif isinstance(value, np.generic):
                        converted.append(value.item())
                    else:
                        converted.append(value)
                yield tuple(converted)

        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE TEMP TABLE {temp_table} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP"
            )
            with cursor.copy(f"COPY {temp_table} ({column_list}) FROM STDIN") as copy:
                for row in converted_rows():
                    copy.write_row(row)
            cursor.execute(merge_statement)

    def load_prices(
        self,
        start: date,
        end: date,
        *,
        company_ids: list[str] | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.price_lake.load(start, end, company_ids=company_ids, tickers=tickers)

    def price_coverage(self) -> tuple[date | None, date | None]:
        return self.price_lake.coverage()

    def price_sources(self) -> list[str]:
        return self.price_lake.sources()

    def source_by_hash(self, sha256: str) -> dict[str, Any] | None:
        frame = self.query_df("SELECT * FROM source_files WHERE sha256 = ?", [sha256])
        return None if frame.empty else frame.iloc[0].to_dict()

    def register_source_file(self, **values) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO source_files
            (source_file_id, dataset, original_name, archived_path, sha256, row_count, imported_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_file_id) DO UPDATE SET
              dataset=EXCLUDED.dataset, original_name=EXCLUDED.original_name,
              archived_path=EXCLUDED.archived_path, row_count=EXCLUDED.row_count,
              imported_at=EXCLUDED.imported_at, metadata_json=EXCLUDED.metadata_json
            """,
            [
                values["source_file_id"],
                values["dataset"],
                values["original_name"],
                values["archived_path"],
                values["sha256"],
                values["row_count"],
                datetime.utcnow(),
                Jsonb(values.get("metadata") or {}),
            ],
        )

    def record_issue(self, issue: DataQualityIssue) -> None:
        self.execute(
            """
            INSERT INTO data_quality_issues VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (issue_id) DO UPDATE SET
              severity=EXCLUDED.severity, message=EXCLUDED.message, created_at=EXCLUDED.created_at
            """,
            [
                issue.issue_id,
                issue.severity.value,
                issue.dataset,
                issue.code,
                issue.message,
                issue.source_file_id,
                issue.row_number,
                issue.created_at,
            ],
        )

    def list_issues(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.query_df(
            "SELECT * FROM data_quality_issues ORDER BY created_at DESC LIMIT ?", [limit]
        ).to_dict(orient="records")

    def source_status(self) -> list[dict[str, Any]]:
        return self.query_df(
            """
            SELECT dataset, COUNT(*) AS files, SUM(row_count) AS rows,
                   MAX(imported_at) AS latest_import
            FROM source_files GROUP BY dataset ORDER BY dataset
            """
        ).to_dict(orient="records")

    def upsert_job(self, job: JobRecord) -> None:
        job.updated_at = datetime.utcnow()
        self.execute(
            """
            INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (job_id) DO UPDATE SET
              status=EXCLUDED.status, progress=EXCLUDED.progress,
              message=EXCLUDED.message, result_ref=EXCLUDED.result_ref,
              error=EXCLUDED.error, updated_at=EXCLUDED.updated_at
            """,
            [
                job.job_id,
                job.kind,
                job.status.value,
                job.progress,
                job.message,
                job.result_ref,
                job.error,
                job.created_at,
                job.updated_at,
            ],
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        frame = self.query_df("SELECT * FROM jobs WHERE job_id = ?", [job_id])
        return None if frame.empty else JobRecord.model_validate(frame.iloc[0].to_dict())

    def save_backtest(self, result: BacktestResult) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO backtest_runs VALUES (?, ?, ?)
            ON CONFLICT (backtest_id) DO UPDATE SET result_json=EXCLUDED.result_json
            """,
            [result.backtest_id, Jsonb(result.model_dump(mode="json")), result.created_at],
        )

    def get_backtest(self, backtest_id: str) -> BacktestResult | None:
        frame = self.query_df(
            "SELECT result_json FROM backtest_runs WHERE backtest_id = ?", [backtest_id]
        )
        return None if frame.empty else BacktestResult.model_validate(frame.iloc[0]["result_json"])

    def list_backtests(self, limit: int = 20) -> list[BacktestResult]:
        frame = self.query_df(
            "SELECT result_json FROM backtest_runs ORDER BY created_at DESC LIMIT ?", [limit]
        )
        return [BacktestResult.model_validate(value) for value in frame["result_json"]]

    def save_model_benchmark(self, result) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO model_benchmarks VALUES (?, ?, ?)
            ON CONFLICT (benchmark_id) DO UPDATE SET result_json=EXCLUDED.result_json
            """,
            [result.benchmark_id, Jsonb(result.model_dump(mode="json")), result.created_at],
        )

    def list_model_benchmarks(self, limit: int = 20):
        from .schemas import ModelBenchmarkResult

        frame = self.query_df(
            "SELECT result_json FROM model_benchmarks ORDER BY created_at DESC LIMIT ?", [limit]
        )
        return [ModelBenchmarkResult.model_validate(value) for value in frame["result_json"]]

    def save_portfolio(self, recommendation: PortfolioRecommendation) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO portfolios VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (portfolio_id) DO UPDATE SET recommendation_json=EXCLUDED.recommendation_json
            """,
            [
                recommendation.portfolio_id,
                recommendation.as_of_date,
                recommendation.cadence,
                Jsonb(recommendation.model_dump(mode="json")),
                datetime.utcnow(),
            ],
        )

    def current_portfolio(self) -> PortfolioRecommendation | None:
        frame = self.query_df(
            "SELECT recommendation_json FROM portfolios ORDER BY as_of_date DESC, created_at DESC LIMIT 1"
        )
        return (
            None
            if frame.empty
            else PortfolioRecommendation.model_validate(frame.iloc[0]["recommendation_json"])
        )

    def save_consensus(self, decision: ConsensusDecision) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO consensus_decisions VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (company_id, as_of_date) DO UPDATE SET
              ticker=EXCLUDED.ticker, decision_json=EXCLUDED.decision_json,
              created_at=EXCLUDED.created_at
            """,
            [
                decision.company_id,
                decision.ticker,
                decision.as_of_date,
                Jsonb(decision.model_dump(mode="json")),
                datetime.utcnow(),
            ],
        )

    def save_research_run(self, run_id: str, as_of_date: date, result: dict[str, Any]) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO research_runs VALUES (?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET result_json=EXCLUDED.result_json
            """,
            [run_id, as_of_date, Jsonb(result), datetime.utcnow()],
        )

    def get_research_run(self, run_id: str) -> dict[str, Any] | None:
        frame = self.query_df("SELECT result_json FROM research_runs WHERE run_id = ?", [run_id])
        return None if frame.empty else frame.iloc[0]["result_json"]

    def save_agent_study_decision(self, variant: str, decision: ConsensusDecision) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO agent_study_decisions VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (company_id, as_of_date, variant) DO UPDATE SET
              ticker=EXCLUDED.ticker, decision_json=EXCLUDED.decision_json,
              created_at=EXCLUDED.created_at
            """,
            [
                decision.company_id,
                decision.ticker,
                decision.as_of_date,
                variant,
                Jsonb(decision.model_dump(mode="json")),
                datetime.utcnow(),
            ],
        )

    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        frame = self.query_df("SELECT * FROM agent_cache WHERE cache_key = ?", [cache_key])
        return None if frame.empty else frame.iloc[0].to_dict()

    def cache_put(self, values: dict[str, Any]) -> None:
        from psycopg.types.json import Jsonb

        self.execute(
            """
            INSERT INTO agent_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cache_key) DO NOTHING
            """,
            [
                values["cache_key"],
                values["role"],
                values["model_alias"],
                values.get("model_version"),
                values.get("system_fingerprint"),
                values["reasoning_effort"],
                values["prompt_version"],
                values["evidence_hash"],
                Jsonb(values["response_json"]),
                values.get("input_tokens", 0),
                values.get("output_tokens", 0),
                values.get("latency_ms", 0.0),
                values.get("created_at", datetime.utcnow()),
            ],
        )

    def latest_available_date(self, table: str) -> date | None:
        if table == "prices":
            return self.price_coverage()[1]
        safe = {
            "fundamentals": "as_of_date",
            "estimates": "as_of_date",
            "prices": "price_date",
            "index_membership": "as_of_date",
        }
        if table not in safe:
            raise ValueError(table)
        frame = self.query_df(f"SELECT MAX({safe[table]}) AS latest FROM {table}")
        return None if frame.empty or pd.isna(frame.iloc[0]["latest"]) else frame.iloc[0]["latest"]


def create_store(settings) -> Store:
    if settings.database_backend == "supabase":
        return SupabasePostgresStore(settings.require_supabase(), settings.price_lake_path)
    return DuckDBStore(settings.database_path)
