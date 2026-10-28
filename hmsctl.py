#!/usr/bin/env python3
"""Select an HMS implementation and exec its original command, without importing it."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command_for(root, variants, variant, action, arguments, python, environ):
    """Resolve a registered command; only its cwd and source import paths change."""
    if variant not in variants:
        raise ValueError(f"Unknown variant {variant!r}; choose from: {', '.join(variants)}")
    spec = variants[variant]
    if action not in spec["commands"]:
        raise ValueError(f"Unknown action {action!r} for {variant}; choose from: {', '.join(spec['commands'])}")
    cwd = (root / spec["root"]).resolve()
    if not cwd.is_dir() or not cwd.is_relative_to(root.resolve()):
        raise ValueError(f"Variant root must be an existing directory inside the repository: {cwd}")
    command = [python if part == "{python}" else part for part in spec["commands"][action]]
    command.extend(arguments)
    env = dict(environ)
    paths = [str(cwd / path) for path in spec.get("python_paths", [])]
    if paths:
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return cwd, command, env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--variant", default=os.environ.get("HMS_VARIANT", "pro"), help="Implementation (default: HMS_VARIANT or pro)"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Interpreter for the selected variant's existing environment"
    )
    parser.add_argument("--list", action="store_true", help="List registered variants and commands")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without importing or starting a backend"
    )
    parser.add_argument("action", nargs="?", default="serve", help="Registered action (default: serve)")
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim; put launcher options before the action",
    )
    args = parser.parse_args(argv)
    try:
        variants = json.loads((ROOT / "variants" / "registry.json").read_text(encoding="utf-8"))
        if args.list:
            for name, spec in variants.items():
                print(f"{name}: {spec['name']} ({', '.join(spec['commands'])})")
            return 0
        python = shutil.which(args.python)
        if python is None:
            raise ValueError(f"Python interpreter is not executable: {args.python}")
        # Resolve a relative venv path before changing to the implementation's root.
        python = os.path.abspath(python)
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        cwd, command, env = command_for(ROOT, variants, args.variant, args.action, arguments, python, os.environ)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "variant": args.variant,
                        "cwd": str(cwd),
                        "argv": command,
                        "pythonpath": env.get("PYTHONPATH", ""),
                    },
                    indent=2,
                )
            )
            return 0
        os.chdir(cwd)
        # Process replacement preserves the backend's signals and exit status.
        os.execvpe(command[0], command, env)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
