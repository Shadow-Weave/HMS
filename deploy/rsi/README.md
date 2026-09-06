# RSI benchmark database isolation

Apply the HMS migration before creating the verifier role. The normal HMS
startup path calls Alembic with the process-wide migration lock; an explicit
deployment job can run the same path with:

```bash
PYTHONPATH=core/dataplane HMS_API_DATABASE_URL='postgresql://…' \
python -c 'import os; from hms_api.migrations import run_migrations; run_migrations(os.environ["HMS_API_DATABASE_URL"])'
```

For a non-public schema, pass `schema="<schema>"` to `run_migrations` and
use the same value for `hms_schema` in the SQL file. Verify that Alembic
reports a single head at `j5k6l7m8n9o0` before exposing a benchmark URL.

The benchmark verifier must use a dedicated PostgreSQL database (or a
dedicated schema with a separately managed database) for each evaluation, a
dedicated PostgreSQL login, and a role that is used only by that evaluation.
Run
[`benchmark_readonly_role.sql`](benchmark_readonly_role.sql) as a database
administrator, create the login through the deployment secret manager, and
grant it the evaluation's group role. Set `hms_schema` and
`hms_benchmark_role` at the top of the script before running it.

Then create a login only for that evaluation and grant the group role (keep
the password in the deployment secret manager):

```sql
CREATE ROLE hms_rsi_benchmark_runner_<eval> LOGIN PASSWORD '<secret>';
GRANT hms_rsi_benchmark_<eval> TO hms_rsi_benchmark_runner_<eval>;
ALTER ROLE hms_rsi_benchmark_runner_<eval>
  SET default_transaction_read_only = 'on';
```

The role can read frozen snapshot tables only. It cannot read live
`memory_units`, candidate staging, verifier receipts, documents, or write to
HMS tables. A shared role in the normal HMS database is not an isolation
boundary: the RLS policy intentionally permits every frozen row in its
database. The runner's URL must therefore point to the per-evaluation
database/schema created for this run. The runner also sets
`default_transaction_read_only=on` through `PGOPTIONS`; both database grants
and transaction mode are required. The database URL passed to the runner must
use a username containing `readonly`, `read_only`, or `benchmark`; this is a
fail-closed guard against accidentally passing the normal HMS write role.
