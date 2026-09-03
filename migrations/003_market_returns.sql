-- Capital IQ short-horizon return snapshot (1D / 1W / 1M).
BEGIN;
SET search_path TO institutional_quant, public;
CREATE TABLE IF NOT EXISTS market_returns (
    company_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    effective_at TIMESTAMP NOT NULL,
    return_1d DOUBLE PRECISION,
    return_1w DOUBLE PRECISION,
    return_1m DOUBLE PRECISION,
    source_file_id VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, as_of_date)
);
COMMIT;
