"""Regression tests for dispatch, argument transparency, and source snapshot checks."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hmsctl


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_variants", ROOT / "scripts" / "verify_variants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VariantLauncherTests(unittest.TestCase):
    def cli(self, *args, env=None, cwd=None):
        environment = dict(os.environ)
        environment.pop("HMS_VARIANT", None)
        environment.update(env or {})
        return subprocess.run(
            [sys.executable, str(ROOT / "hmsctl.py"), *args],
            cwd=cwd or ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def dry_run(self, variant, action="serve", *args, env=None):
        result = self.cli("--variant", variant, "--dry-run", action, *args, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_default_remains_pro(self):
        result = self.cli("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads(result.stdout)
        self.assertEqual(command["variant"], "pro")
        self.assertEqual(command["argv"][1:], ["-c", "from hms_api.main import main; raise SystemExit(main())"])
        self.assertEqual(Path(command["cwd"]), ROOT)

    def test_native_arguments_are_untouched(self):
        arguments = ["--config", "a path/config.json", "--port=9000", "--variant", "native", "literal;$(value)"]
        command = self.dry_run("elite", "serve", *arguments)
        self.assertEqual(
            command["argv"][1:], ["-c", "from hms_elite.server.app import main; raise SystemExit(main())", *arguments]
        )
        self.assertEqual(Path(command["cwd"]), ROOT / "variants" / "elite")

    def test_environment_selector_and_explicit_override(self):
        selected = self.cli("--dry-run", env={"HMS_VARIANT": "elite"})
        self.assertEqual(json.loads(selected.stdout)["variant"], "elite")
        overridden = self.dry_run("pro", env={"HMS_VARIANT": "elite"})
        self.assertEqual(overridden["variant"], "pro")

    def test_import_paths_select_one_backend_and_keep_inherited_paths(self):
        for variant, source in (("pro", ROOT / "core/dataplane"), ("elite", ROOT / "variants/elite")):
            command = self.dry_run(variant, env={"PYTHONPATH": "inherited-path"})
            self.assertEqual(command["pythonpath"], str(source) + os.pathsep + "inherited-path")

    def test_native_benchmark_commands_and_flags_are_preserved(self):
        pro = self.dry_run("pro", "benchmark", "--resume", "--max-questions=2")
        self.assertEqual(pro["argv"], ["bash", ".aaaSCRIPT/run_benchmark.sh", "--resume", "--max-questions=2"])
        elite = self.dry_run("elite", "benchmark", "--arms", "elite_full", "--limit", "2")
        self.assertEqual(
            elite["argv"][1:], ["benchmarks/egolife/run_egolife.py", "--arms", "elite_full", "--limit", "2"]
        )

    def test_compose_uses_the_selected_native_project(self):
        for variant in ("pro", "elite"):
            command = self.dry_run(variant, "compose", "-f", "docker-compose.yml", "up", "-d")
            self.assertEqual(command["argv"], ["docker", "compose", "-f", "docker-compose.yml", "up", "-d"])

    def test_unknown_variant_and_action_fail_before_starting(self):
        for args in (("--variant", "missing"), ("--variant", "elite", "missing")):
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Unknown", result.stderr)

    def test_list_does_not_import_backends(self):
        with patch.dict(sys.modules, {"hms_api": None, "hms_elite": None}):
            self.assertEqual(hmsctl.main(["--list"]), 0)

    def test_registration_supports_a_future_backend_without_dispatch_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "variants/pmm").mkdir(parents=True)
            registry = {"pmm": {"root": "variants/pmm", "commands": {"serve": ["{python}", "-m", "pmm.server"]}}}
            cwd, command, env = hmsctl.command_for(
                root, registry, "pmm", "serve", ["--port", "7000"], "python-pmm", {"UNCHANGED": "yes"}
            )
            self.assertEqual(cwd, root / "variants/pmm")
            self.assertEqual(command, ["python-pmm", "-m", "pmm.server", "--port", "7000"])
            self.assertEqual(env, {"UNCHANGED": "yes"})

    def test_backend_root_cannot_escape_the_repository(self):
        registry = {"bad": {"root": "..", "commands": {"serve": ["anything"]}}}
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            hmsctl.command_for(ROOT, registry, "bad", "serve", [], sys.executable, {})

    def test_separator_is_not_forwarded_to_the_native_parser(self):
        command = self.dry_run("elite", "serve", "--", "--help")
        self.assertEqual(command["argv"][-1], "--help")
        self.assertNotIn("--", command["argv"])

    def test_execution_preserves_cwd_environment_and_exit_status(self):
        program = (
            "import json,os,sys; print(json.dumps([os.getcwd(),os.environ['UNCHANGED'],sys.argv[1:]])); sys.exit(23)"
        )
        result = self.cli("--variant", "elite", "run", "-c", program, "one argument", env={"UNCHANGED": "retained"})
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(json.loads(result.stdout), [str(ROOT / "variants/elite"), "retained", ["one argument"]])

    def test_launch_from_another_directory(self):
        result = self.cli("--variant", "elite", "--dry-run", cwd=ROOT.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(json.loads(result.stdout)["cwd"]), ROOT / "variants/elite")

    def test_explicit_interpreter_and_missing_interpreter(self):
        result = self.cli("--variant", "elite", "--python", sys.executable, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["argv"][0], sys.executable)
        result = self.cli("--python", "/missing/python", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not executable", result.stderr)


class VariantSnapshotTests(unittest.TestCase):
    def test_original_files_match_the_imported_snapshots(self):
        snapshots = json.loads((ROOT / "variants/sources.json").read_text(encoding="utf-8"))
        self.assertEqual(load_verifier().verify(ROOT, snapshots), [])

    def test_snapshot_detects_modified_and_missing_files(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.py").write_text("changed", encoding="utf-8")
            original = hashlib.sha256(b"original").hexdigest()
            snapshots = {"sample": {"root": ".", "files": {"source.py": original, "missing.py": original}}}
            self.assertEqual(
                load_verifier().verify(root, snapshots), ["sample: changed source.py", "sample: missing missing.py"]
            )


if __name__ == "__main__":
    unittest.main()
