#!/usr/bin/env python3
"""
Generate OpenAPI specification from FastAPI app.

This script imports the FastAPI app and exports its OpenAPI schema to a JSON file.
"""

import argparse
import json
from pathlib import Path

from hms_api import MemoryEngine
from hms_api.api import create_app


def _add_binary_format_hints(value):
    """Keep multipart file fields usable across OpenAPI 3.1 generators.

    FastAPI emits JSON Schema's ``contentMediaType`` annotation for
    ``UploadFile`` under OpenAPI 3.1. Some current SDK generators only map the
    long-established ``format: binary`` annotation to Blob/file types. Keeping
    both annotations is valid in 3.1 and also survives the temporary 3.0
    compatibility projection.
    """

    if isinstance(value, list):
        for item in value:
            _add_binary_format_hints(item)
        return
    if not isinstance(value, dict):
        return

    if (
        value.get("type") == "string"
        and value.get("contentMediaType") == "application/octet-stream"
        and "contentEncoding" not in value
    ):
        value.setdefault("format", "binary")

    for item in value.values():
        _add_binary_format_hints(item)


def _convert_nullable_any_of_to_openapi_30(value):
    """Convert JSON Schema null unions to OpenAPI 3.0 ``nullable``.

    The canonical FastAPI document remains OpenAPI 3.1. OpenAPI Generator
    7.10's Python target cannot consume Pydantic's ``anyOf: [T, null]`` shape,
    while the Rust SDK already performs this same compatibility conversion at
    build time. This helper gives generated SDKs one reproducible 3.0 input
    without weakening the canonical schema.
    """

    if isinstance(value, list):
        for item in value:
            _convert_nullable_any_of_to_openapi_30(item)
        return
    if not isinstance(value, dict):
        return

    # ``contentMediaType`` is JSON Schema/OpenAPI 3.1-only. The companion
    # ``format: binary`` hint preserves UploadFile semantics for 3.0 clients.
    if value.get("format") == "binary":
        value.pop("contentMediaType", None)

    any_of = value.get("anyOf")
    if isinstance(any_of, list) and any(isinstance(item, dict) and item.get("type") == "null" for item in any_of):
        non_null = [item for item in any_of if not (isinstance(item, dict) and item.get("type") == "null")]
        value.pop("anyOf")
        if len(non_null) == 1:
            value.update(non_null[0])
        else:
            value["anyOf"] = non_null
        value["nullable"] = True

    for item in value.values():
        _convert_nullable_any_of_to_openapi_30(item)


def generate_openapi_spec(output_path: str | None = None, *, compatibility_openapi_30: bool = False):
    """Generate OpenAPI spec and save to file."""
    # Default to knowledge/site/static/openapi.json (single source of truth)
    if output_path is None:
        root_dir = Path(__file__).resolve().parents[3]
        output_path = str(root_dir / "knowledge" / "site" / "static" / "openapi.json")

    # Create a temporary memory instance for OpenAPI generation
    _memory = MemoryEngine(
        db_url="mock",
        memory_llm_provider="ollama",
        memory_llm_api_key="mock",
        memory_llm_model="mock",
    )
    app = create_app(_memory)

    # Get the OpenAPI schema from the app
    openapi_schema = app.openapi()
    _add_binary_format_hints(openapi_schema)
    if compatibility_openapi_30:
        openapi_schema["openapi"] = "3.0.3"
        _convert_nullable_any_of_to_openapi_30(openapi_schema)

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(openapi_schema, f, indent=2)
        f.write("\n")

    print(f"✓ OpenAPI specification generated: {output_file.absolute()}")
    print(f"  - Title: {openapi_schema['info']['title']}")
    print(f"  - Version: {openapi_schema['info']['version']}")
    print(f"  - Endpoints: {len(openapi_schema['paths'])}")

    # List endpoints
    print("\n  Endpoints:")
    for path, methods in openapi_schema["paths"].items():
        for method in methods.keys():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                endpoint_info = methods[method]
                summary = endpoint_info.get("summary", "No summary")
                tags = ", ".join(endpoint_info.get("tags", ["untagged"]))
                print(f"    {method.upper():6} {path:30} [{tags}] - {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", help="Output file (defaults to the canonical OpenAPI JSON path)")
    parser.add_argument(
        "--compatibility-openapi-30",
        action="store_true",
        help="Emit a temporary OpenAPI 3.0.3 SDK-generator input instead of the canonical 3.1 document",
    )
    args = parser.parse_args()
    generate_openapi_spec(args.output, compatibility_openapi_30=args.compatibility_openapi_30)
