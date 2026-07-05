"""Lint-style enforcement for the dialect-dispatched migration pattern.

Every file in ``alembic/versions/`` must route ``upgrade``/``downgrade`` through
``alembic._dialect.run_for_dialect`` so PG and Oracle stay in lockstep. This
test fails the build if a new migration is added without filling at least one
dialect slot — without it we'd silently re-introduce drift on Oracle the first
time someone copies an old PG migration as a template.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

VERSIONS_DIR = Path(__file__).resolve().parent.parent / "hms_api" / "alembic" / "versions"


def _migration_files() -> list[Path]:
    return sorted(p for p in VERSIONS_DIR.glob("*.py") if not p.name.startswith("__"))


@pytest.mark.parametrize("path", _migration_files(), ids=lambda p: p.name)
def test_migration_uses_dialect_dispatcher(path: Path) -> None:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))

    imports_dispatcher = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "hms_api.alembic._dialect"
        and any(alias.name == "run_for_dialect" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_dispatcher, (
        f"{path.name}: missing 'from hms_api.alembic._dialect import run_for_dialect'. "
        "All migrations must dispatch through run_for_dialect — see CLAUDE.md."
    )

    top_level_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in ("upgrade", "downgrade"):
        assert required in top_level_fns, f"{path.name}: missing top-level def {required}()."
        assert _calls_run_for_dialect(top_level_fns[required]), (
            f"{path.name}: {required}() must call run_for_dialect(...)."
        )

    has_pg_slot = "_pg_upgrade" in top_level_fns
    has_oracle_slot = "_oracle_upgrade" in top_level_fns
    assert has_pg_slot or has_oracle_slot, (
        f"{path.name}: migration defines neither _pg_upgrade nor _oracle_upgrade — "
        "at least one dialect slot must be filled (set the other to None if intentional)."
    )


def _calls_run_for_dialect(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_for_dialect"
        ):
            return True
    return False
