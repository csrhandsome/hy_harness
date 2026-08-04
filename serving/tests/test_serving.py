from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SERVING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING_ROOT / "src"))

from vla_serving.config import load_registry
from vla_serving.runtimes import get_runtime


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(SERVING_ROOT / "config.yaml")

    def test_every_model_has_a_manifest_directory(self) -> None:
        self.assertEqual(
            set(self.registry.models),
            {"hy-embodied-0.5", "hy-embodied-vlm-1.0"},
        )
        for model in self.registry.models.values():
            self.assertEqual(model.id, model.manifest.parent.name)
            self.assertTrue(model.manifest.is_file())

    def test_cache_path_always_uses_model_id(self) -> None:
        for model in self.registry.models.values():
            self.assertEqual(
                self.registry.cache_dir(model),
                SERVING_ROOT / "cache" / model.id,
            )

    def test_runtime_and_plugin_references_exist(self) -> None:
        for model in self.registry.models.values():
            runtime = get_runtime(model.runtime)
            self.assertTrue(callable(runtime.build_command))
            plugin = model.runtime_options.get("plugin")
            if plugin:
                self.assertTrue((SERVING_ROOT / str(plugin) / "pyproject.toml").is_file())


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(SERVING_ROOT / "config.yaml")

    def test_transformers_command_uses_shared_server(self) -> None:
        model = self.registry.select("hy-embodied-0.5")
        with mock.patch.dict(os.environ, {"MODEL_PATH": "/models/hy05"}, clear=True):
            command = get_runtime(model.runtime).build_command(self.registry, model)
        self.assertIn("runtimes/transformers/server.py", command[1])
        self.assertIn("/models/hy05", command)
        self.assertNotIn("backends", " ".join(command))

    def test_vllm_command_contains_manifest_options(self) -> None:
        model = self.registry.select("hy-embodied-vlm-1.0")
        with mock.patch.dict(os.environ, {"MODEL_PATH": "/models/vlm10"}, clear=True):
            command = get_runtime(model.runtime).build_command(self.registry, model)
        self.assertEqual(command[1:3], ["serve", "/models/vlm10"])
        self.assertIn("--reasoning-parser", command)
        self.assertIn("hunyuan_v3", command)
        self.assertIn("--tool-call-parser", command)
        self.assertIn("hy_v3", command)
        self.assertIn("--enable-auto-tool-choice", command)

    def test_environment_overrides_vllm_defaults(self) -> None:
        model = self.registry.select("hy-embodied-vlm-1.0")
        overrides = {"TP": "2", "MAX_MODEL_LEN": "4096", "EXTRA_ARGS": "--load-format dummy"}
        with mock.patch.dict(os.environ, overrides, clear=True):
            command = get_runtime(model.runtime).build_command(self.registry, model)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(command[command.index("--max-model-len") + 1], "4096")
        self.assertEqual(command[-2:], ["--load-format", "dummy"])


if __name__ == "__main__":
    unittest.main()
