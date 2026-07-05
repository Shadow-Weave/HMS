#!/usr/bin/env python3
"""
Check that the Python and TypeScript high-level client wrappers expose every
request-body parameter from the OpenAPI spec.

The generated (low-level) clients are auto-generated from the spec and are
always in sync.  The *wrapper* clients (``hms_client.py`` for Python,
``src/index.ts`` for TypeScript) are hand-written convenience layers that
re-export a curated surface.  This script catches fields that are present in
the OpenAPI spec but missing from the wrapper.

For each client, the script:
  1. Discovers which OpenAPI operations the wrapper covers by scanning for
     SDK call sites (e.g. ``sdk.recallMemories(`` or
     ``self._memory_api.recall_memories(``).
  2. For each covered operation, checks that every request-body property
     appears somewhere in the wrapper source around that call site (with
     automatic snake_case ↔ camelCase conversion).
  3. Allows explicit skips via per-client ``.openapi-coverage.toml`` files.

Usage:
    cd lab/evaluation
    uv run client-coverage-check
    uv run client-coverage-check --client python
    uv run client-coverage-check --client typescript
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def get_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "interface" / "sdk").is_dir() and (parent / "knowledge" / "site").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repo root from {here}")


# ── helpers ──────────────────────────────────────────────────────────────


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def load_openapi_operations(spec_path: Path) -> dict[str, list[str]]:
    """Return {operation_id: [property_names]} for operations with request bodies."""
    spec = json.loads(spec_path.read_text())
    schemas = spec.get("components", {}).get("schemas", {})
    result: dict[str, list[str]] = {}

    for path_item in spec.get("paths", {}).values():
        for method, op in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            if not body:
                continue
            if "$ref" in body:
                schema_name = body["$ref"].split("/")[-1]
                schema = schemas.get(schema_name, {})
            else:
                schema = body
            props = list((schema.get("properties") or {}).keys())
            if props:
                result[op_id] = props

    return result


def load_manifest(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Return (operation-level skips, per-operation field skips) from a TOML manifest."""
    if not path.exists():
        return {}, {}
    data = tomllib.loads(path.read_text())

    skip_raw = data.get("skip", {}) or {}
    op_skips: dict[str, str] = {}
    for op_id, reason in skip_raw.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: skip.{op_id} must be a non-empty string reason")
        op_skips[op_id] = reason.strip()

    fields_raw = data.get("fields", {}) or {}
    field_skips: dict[str, dict[str, str]] = {}
    for op_id, table in fields_raw.items():
        if not isinstance(table, dict):
            raise ValueError(f"{path}: [fields.{op_id}] must be a table")
        per_op: dict[str, str] = {}
        for field, reason in table.items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{path}: fields.{op_id}.{field} must be a non-empty string reason")
            per_op[field] = reason.strip()
        field_skips[op_id] = per_op

    return op_skips, field_skips


# ── per-client checking ─────────────────────────────────────────────────


def field_present_in_source(prop: str, source: str) -> bool:
    """Check if a field name (snake_case from OpenAPI) appears in wrapper source.

    Checks both snake_case and camelCase variants, looking for the field
    as an identifier (word boundary on at least one side).
    """
    variants = {prop, snake_to_camel(prop)}
    for v in variants:
        # Match as an identifier: preceded by word boundary or common delimiters
        # and followed by word boundary or common delimiters
        if re.search(r"(?:^|[\s{,.(:\[\"'])" + re.escape(v) + r"(?:$|[\s},:)?\[\"'=])", source):
            return True
    return False


def check_client(
    *,
    client_name: str,
    source_path: Path,
    manifest_path: Path,
    op_props: dict[str, list[str]],
    find_covered_ops: Callable[[str, set[str]], set[str]],
) -> list[str]:
    """Check a single client wrapper. Returns list of error strings."""
    source = source_path.read_text()
    op_skips, field_skips = load_manifest(manifest_path)
    errors: list[str] = []

    # Find which operations this wrapper covers
    covered_ops = find_covered_ops(source, set(op_props.keys()))

    # Check that skipped operations actually exist and aren't covered
    for op_id in sorted(op_skips.keys()):
        if op_id not in op_props:
            errors.append(f"STALE OP     {op_id}: listed in [skip] but has no request body in the spec")

    # Check fields for each covered operation
    total_props = 0
    covered_props = 0
    skipped_props = 0

    for op_id in sorted(covered_ops):
        props = op_props.get(op_id, [])
        if op_id in op_skips:
            continue
        per_op_skips = field_skips.get(op_id, {})
        for prop in props:
            total_props += 1
            if prop in per_op_skips:
                skipped_props += 1
                continue
            if field_present_in_source(prop, source):
                covered_props += 1
                continue
            errors.append(
                f"MISSING PARAM {op_id}.{prop}: request body field not found in "
                f"wrapper source (checked: {prop}, {snake_to_camel(prop)})"
            )

    # Stale field skips
    for op_id, per_op in field_skips.items():
        if op_id not in op_props:
            for field in per_op:
                errors.append(
                    f"STALE PARAM  {op_id}.{field}: [fields.{op_id}] references "
                    f"an operation with no request body in the spec"
                )
            continue
        known = set(op_props.get(op_id, []))
        for field in per_op:
            if field not in known:
                errors.append(
                    f"STALE PARAM  {op_id}.{field}: field not present in the "
                    f"{op_id} request schema. Remove from [fields.{op_id}]."
                )

    print(f"  [{client_name}]")
    print(f"    Source:          {source_path}")
    print(f"    Manifest:        {manifest_path}")
    print(f"    Operations:      {len(covered_ops)} covered")
    print(f"    Request params:  {total_props}")
    print(f"      covered:       {covered_props}")
    print(f"      skipped:       {skipped_props}")
    print(f"      missing:       {total_props - covered_props - skipped_props}")

    return errors


# ── TypeScript ───────────────────────────────────────────────────────────


def find_ts_covered_ops(source: str, op_ids: set[str]) -> set[str]:
    """Find operations called via ``sdk.<operationId>(`` in TS wrapper.

    The TS generated SDK uses camelCase method names (e.g. ``retainMemories``)
    while OpenAPI operation IDs are snake_case (``retain_memories``).  We check
    both variants.
    """
    covered: set[str] = set()
    for op_id in op_ids:
        camel = snake_to_camel(op_id)
        for variant in (op_id, camel):
            if re.search(r"sdk\.\s*" + re.escape(variant) + r"\s*\(", source):
                covered.add(op_id)
                break
    return covered


# ── Python ───────────────────────────────────────────────────────────────


def find_py_covered_ops(source: str, op_ids: set[str]) -> set[str]:
    """Find operations called via ``self._*_api.<operation_id>(`` in Python wrapper."""
    covered: set[str] = set()
    for op_id in op_ids:
        if re.search(r"self\._\w+_api\.\s*" + re.escape(op_id) + r"\s*\(", source):
            covered.add(op_id)
    return covered


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Check client SDK wrapper coverage against OpenAPI spec")
    parser.add_argument(
        "--client",
        choices=["python", "typescript", "all"],
        default="all",
        help="Which client to check (default: all)",
    )
    args = parser.parse_args()

    root = get_repo_root()
    spec_path = root / "knowledge" / "site" / "static" / "openapi.json"

    if not spec_path.exists():
        print(f"ERROR: OpenAPI spec not found at {spec_path}", file=sys.stderr)
        print("       Run ./scripts/generate-openapi.sh first.", file=sys.stderr)
        sys.exit(1)

    op_props = load_openapi_operations(spec_path)

    clients: list[dict] = []

    if args.client in ("python", "all"):
        clients.append(
            {
                "client_name": "python",
                "source_path": root / "interface" / "sdk" / "python" / "hms_client" / "hms_client.py",
                "manifest_path": root / "interface" / "sdk" / "python" / ".openapi-coverage.toml",
                "find_covered_ops": find_py_covered_ops,
            }
        )

    if args.client in ("typescript", "all"):
        clients.append(
            {
                "client_name": "typescript",
                "source_path": root / "interface" / "sdk" / "typescript" / "src" / "index.ts",
                "manifest_path": root / "interface" / "sdk" / "typescript" / ".openapi-coverage.toml",
                "find_covered_ops": find_ts_covered_ops,
            }
        )

    all_errors: list[str] = []

    print("Client SDK OpenAPI coverage check")
    print(f"  Spec: {spec_path.relative_to(root)}")
    print()

    for client_cfg in clients:
        if not client_cfg["source_path"].exists():
            print(f"  [{client_cfg['client_name']}] SKIPPED: source not found at {client_cfg['source_path']}")
            continue
        errs = check_client(op_props=op_props, **client_cfg)
        all_errors.extend(f"[{client_cfg['client_name']}] {e}" for e in errs)
        print()

    if all_errors:
        print(f"FAILED: {len(all_errors)} issue(s):")
        for e in all_errors:
            print(f"  {e}")
        print()
        print(
            "Fix by either exposing the field in the wrapper, or adding\n"
            "an entry to the client's .openapi-coverage.toml with a reason."
        )
        sys.exit(1)

    print("OK: all covered operations have their request params accounted for.")


if __name__ == "__main__":
    main()
