-- S&P 500 Institutional Quant Platform
-- Supabase/PostgreSQL schema migration v1

BEGIN;

CREATE SCHEMA IF NOT EXISTS institutional_quant;
SET search_path TO institutional_quant, public;

CREATE TABLE IF NOT EXISTS source_files (
    source_file_id VARCHAR PRIMARY KEY,
    dataset VARCHAR NOT NULL,
    original_name VARCHAR NOT NULL,
    archived_path VARCHAR NOT NULL,
    sha256 VARCHAR NOT NULL UNIQUE,
    row_count BIGINT NOT NULL,
    imported_at TIMESTAMP NOT NULL,
    metadata_json JSONB
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
    value DOUBLE PRECISION,
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
    value DOUBLE PRECISION,
    unit VARCHAR,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, fiscal_period, effective_at, metric)
);

CREATE TABLE IF NOT EXISTS prices (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    price_date DATE NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    adjusted_close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
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
    institutional_pct DOUBLE PRECISION,
    institutional_change DOUBLE PRECISION,
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
    shares DOUBLE PRECISION NOT NULL,
    value DOUBLE PRECISION,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, transaction_date, transaction_type, shares)
);

CREATE TABLE IF NOT EXISTS factor_observations (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    sector VARCHAR NOT NULL,
    feature_json JSONB NOT NULL,
    factor_score DOUBLE PRECISION NOT NULL,
    elastic_score DOUBLE PRECISION,
    tree_score DOUBLE PRECISION,
    ml_score DOUBLE PRECISION,
    ensemble_score DOUBLE PRECISION,
    next_month_excess_return DOUBLE PRECISION,
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
    response_json JSONB NOT NULL,
    input_tokens BIGINT,
    output_tokens BIGINT,
    latency_ms DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS consensus_decisions (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    decision_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id VARCHAR PRIMARY KEY,
    as_of_date DATE NOT NULL,
    result_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_study_decisions (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    variant VARCHAR NOT NULL,
    decision_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, as_of_date, variant)
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id VARCHAR PRIMARY KEY,
    as_of_date DATE NOT NULL,
    cadence VARCHAR NOT NULL,
    recommendation_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_id VARCHAR PRIMARY KEY,
    result_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS model_benchmarks (
    benchmark_id VARCHAR PRIMARY KEY,
    result_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    progress DOUBLE PRECISION NOT NULL,
    message VARCHAR NOT NULL,
    result_ref VARCHAR,
    error VARCHAR,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

COMMIT;
