-- Configure a PostgreSQL group role for the isolated benchmark verifier.
-- Run as a database administrator and replace the schema/database names as
-- needed. Keep the LOGIN credential in the benchmark runner's secret store.

-- Run this file once per benchmark evaluation in a dedicated database.  Set
-- both values to evaluation-unique names; a shared role/database would expose
-- every frozen snapshot in that database to that benchmark worker.
\set hms_schema public
\set hms_benchmark_role hms_rsi_benchmark

BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'hms_benchmark_role') THEN
        EXECUTE format('CREATE ROLE %I NOLOGIN', :'hms_benchmark_role');
    END IF;
END
$$;

ALTER ROLE :hms_benchmark_role SET default_transaction_read_only = 'on';
DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'hms_benchmark_role');
END
$$;
GRANT USAGE ON SCHEMA :hms_schema TO :hms_benchmark_role;

-- The benchmark role sees frozen copies only. It cannot inspect live memory,
-- candidate staging, verifier receipts, documents, or write-capable tables.
GRANT SELECT ON TABLE
    :hms_schema.rsi_harness_snapshots,
    :hms_schema.rsi_snapshot_memory_units
TO :hms_benchmark_role;

-- Do not expose mutable/in-progress rows through the benchmark login.  RLS is
-- deliberately enabled on the frozen-copy tables because a plain SELECT grant
-- would otherwise allow the runner to read building snapshots while an
-- evolution round is still in flight.  The normal HMS table owner bypasses
-- RLS (as it must for the write API); the benchmark login is a separate role
-- and is governed by the policies below.
ALTER TABLE :hms_schema.rsi_harness_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE :hms_schema.rsi_snapshot_memory_units ENABLE ROW LEVEL SECURITY;
ALTER TABLE :hms_schema.rsi_snapshot_candidates ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rsi_benchmark_frozen_snapshots ON :hms_schema.rsi_harness_snapshots;
CREATE POLICY rsi_benchmark_frozen_snapshots
    ON :hms_schema.rsi_harness_snapshots
    FOR SELECT TO :hms_benchmark_role
    USING (status = 'frozen');

DROP POLICY IF EXISTS rsi_benchmark_frozen_memory ON :hms_schema.rsi_snapshot_memory_units;
CREATE POLICY rsi_benchmark_frozen_memory
    ON :hms_schema.rsi_snapshot_memory_units
    FOR SELECT TO :hms_benchmark_role
    USING (
        EXISTS (
            SELECT 1
            FROM :hms_schema.rsi_harness_snapshots s
            WHERE s.bank_id = rsi_snapshot_memory_units.bank_id
              AND s.snapshot_id = rsi_snapshot_memory_units.snapshot_id
              AND s.status = 'frozen'
        )
    );

DROP POLICY IF EXISTS rsi_benchmark_frozen_candidates ON :hms_schema.rsi_snapshot_candidates;
CREATE POLICY rsi_benchmark_frozen_candidates
    ON :hms_schema.rsi_snapshot_candidates
    FOR SELECT TO :hms_benchmark_role
    USING (
        EXISTS (
            SELECT 1
            FROM :hms_schema.rsi_harness_snapshots s
            WHERE s.bank_id = rsi_snapshot_candidates.bank_id
              AND s.snapshot_id = rsi_snapshot_candidates.snapshot_id
              AND s.status = 'frozen'
        )
    );

REVOKE ALL ON TABLE
    :hms_schema.memory_units,
    :hms_schema.documents,
    :hms_schema.chunks,
    :hms_schema.rsi_memory_candidates,
    :hms_schema.rsi_verifier_receipts
FROM :hms_benchmark_role;

-- Keep the guarantee independent of any permissive cluster-wide defaults.
-- PUBLIC normally has no table privileges, but explicitly revoking here makes
-- the deployment invariant auditable and prevents accidental future grants
-- from widening the benchmark surface through role inheritance.
REVOKE ALL ON TABLE
    :hms_schema.memory_units,
    :hms_schema.documents,
    :hms_schema.chunks,
    :hms_schema.rsi_memory_candidates,
    :hms_schema.rsi_verifier_receipts
FROM PUBLIC;

COMMIT;

-- Example deployment step (outside this file, using a secret manager):
--   CREATE ROLE hms_rsi_benchmark_runner LOGIN PASSWORD '...';
--   GRANT hms_rsi_benchmark TO hms_rsi_benchmark_runner;
--   ALTER ROLE hms_rsi_benchmark_runner SET default_transaction_read_only = 'on';
