from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .client import HMSVendorClient


def main() -> None:
    parser = argparse.ArgumentParser(prog="hms-vendor", description="Vendor-facing HMS retain/recall test client.")
    parser.add_argument("--base-url", default=os.getenv("HMS_BASE_URL"), help="HMS API base URL.")
    parser.add_argument("--api-key", default=os.getenv("HMS_API_KEY"), help="HMS API key.")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("HMS_TIMEOUT", "300")))

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Check HMS service health.")

    run_case = subparsers.add_parser("run-case", help="Run one JSON case through retain and recall.")
    run_case.add_argument("--case", required=True, help="Path to a vendor case JSON file.")
    run_case.add_argument("--bank-id", default=os.getenv("HMS_BANK_ID", "vendor-demo"))
    run_case.add_argument("--create-bank", action="store_true")
    run_case.add_argument("--reset-bank", action="store_true")
    run_case.add_argument("--retain-async", action="store_true")
    run_case.add_argument("--no-wait", action="store_true", help="Do not wait for async retain before recall.")
    run_case.add_argument("--no-organize", action="store_true", help="Skip evidence ledger organization.")
    run_case.add_argument("--output", help="Optional path to write the JSON result.")

    recall = subparsers.add_parser("recall", help="Run recall against an existing bank.")
    recall.add_argument("--bank-id", required=True)
    recall.add_argument("--question", required=True)
    recall.add_argument("--question-date")
    recall.add_argument("--output")

    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or HMS_BASE_URL is required")

    client = HMSVendorClient(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    if args.command == "health":
        _print_json(client.health())
        return

    if args.command == "run-case":
        case = _load_case(Path(args.case))
        result = client.pipeline(
            bank_id=args.bank_id,
            sessions=case["sessions"],
            question=case["question"],
            question_date=case.get("question_date"),
            create_bank=args.create_bank,
            reset_bank=args.reset_bank,
            bank_profile=case.get("bank_profile") or {},
            retain_async=args.retain_async,
            wait_for_retain=not args.no_wait,
            organize=not args.no_organize,
        ).to_dict()
        result["case_id"] = case.get("case_id")
        result["expected_answer"] = case.get("expected_answer")
        _write_or_print(result, args.output)
        return

    if args.command == "recall":
        result = client.recall(
            bank_id=args.bank_id,
            question=args.question,
            question_date=args.question_date,
        ).to_dict()
        _write_or_print(result, args.output)
        return


def _load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"question", "sessions"}
    missing = sorted(required - set(data))
    if missing:
        raise SystemExit(f"Case file is missing required field(s): {', '.join(missing)}")
    return data


def _write_or_print(payload: dict[str, Any], output_path: str | None) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        _print_json(payload)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
