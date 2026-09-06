"""Compare a baseline H0 result artifact with a frozen H1 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi_harness import compare_evaluation_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="H0 JSON result artifact")
    parser.add_argument("--evolved", type=Path, required=True, help="H1 frozen JSON result artifact")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    evolved = json.loads(args.evolved.read_text(encoding="utf-8"))
    comparison = compare_evaluation_reports(baseline, evolved)
    print(json.dumps(comparison.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
