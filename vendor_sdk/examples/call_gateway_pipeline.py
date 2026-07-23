from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import error, request


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the HMS vendor gateway pipeline endpoint.")
    parser.add_argument(
        "--case",
        required=True,
        help="Path to a case JSON file.",
    )
    parser.add_argument("--bank-id", default=os.getenv("HMS_BANK_ID", "vendor-demo"))
    parser.add_argument("--base-url", default=os.getenv("HMS_BASE_URL"), help="Vendor gateway base URL.")
    parser.add_argument("--api-key", default=os.getenv("HMS_API_KEY"), help="Vendor gateway API key.")
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    if not args.base_url:
        raise SystemExit("Set HMS_BASE_URL or pass --base-url, e.g. http://127.0.0.1:18081")
    if not args.api_key:
        raise SystemExit("Set HMS_API_KEY or pass --api-key.")

    with Path(args.case).open("r", encoding="utf-8") as handle:
        case = json.load(handle)

    payload = {
        "bank_id": args.bank_id,
        "sessions": case["sessions"],
        "question": case["question"],
        "question_date": case.get("question_date"),
        "create_bank": True,
        "reset_bank": True,
        "bank_profile": case.get("bank_profile") or {},
        "retain_async": False,
        "wait_for_retain": True,
        "recall_budget": "mid",
        "organize": True
    }

    req = request.Request(
        f"{args.base_url.rstrip('/')}/v1/vendor/pipeline",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Gateway request failed: HTTP {exc.code} {detail}") from exc

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
