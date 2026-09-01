-- Preserve distinct point-in-time snapshots of the same filed fundamental.

BEGIN;

SET search_path TO institutional_quant, public;

DO $$
DECLARE
    key_has_as_of BOOLEAN;
BEGIN
    SELECT pg_get_constraintdef(c.oid) ILIKE '%as_of_date%'
    INTO key_has_as_of
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'institutional_quant'
      AND t.relname = 'fundamentals'
      AND c.contype = 'p';

    IF NOT COALESCE(key_has_as_of, FALSE) THEN
        ALTER TABLE fundamentals DROP CONSTRAINT fundamentals_pkey;
        ALTER TABLE fundamentals ADD PRIMARY KEY (
            company_id,
            period_end,
            period_type,
            effective_at,
            as_of_date,
            metric
        );
    END IF;
END $$;

COMMIT;
