-- Persistent local watchlist. Brokerage positions remain read-only telemetry.
BEGIN;
SET search_path TO institutional_quant, public;
CREATE TABLE IF NOT EXISTS watchlist_items (
    ticker VARCHAR PRIMARY KEY,
    note VARCHAR,
    created_at TIMESTAMP NOT NULL
);
COMMIT;
