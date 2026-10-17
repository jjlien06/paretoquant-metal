"""Stdlib-only host checks; no array runtimes or Metal execution."""

import builtins
import importlib.util
import sys
import unittest
from pathlib import Path


class CPUAdmissionSmoke(unittest.TestCase):
    def setUp(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"mlx", "numpy", "mlx_lm"}:
                raise AssertionError(f"forbidden array runtime import: {name}")
            return original(name, *args, **kwargs)

        builtins.__import__ = guarded
        self.addCleanup(setattr, builtins, "__import__", original)
        path = Path(__file__).with_name("slot_qmv.py")
        spec = importlib.util.spec_from_file_location("stdlib_slot_qmv", path)
        assert spec is not None and spec.loader is not None
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(self.module)

    def test_fast_and_slow_output_contract(self):
        for k in (32, 64, 128, 256, 512, 896, 4864, 5120, 27648):
            for bits in (3, 4):
                with self.subTest(k=k, bits=bits):
                    meta = self.module.ArrayMeta
                    plan = self.module.shape_plan(
                        meta((1, 9, k), "float16"),
                        meta((8, k * bits // 32), "uint32"),
                        meta((8, k // 32), "float16"),
                        meta((8, k // 32), "float16"),
                        bits=bits,
                        group_size=32,
                    )
                    self.assertEqual(plan.output_shape, (1, 9, 8))
                    self.assertEqual(plan.fast, k % 512 == 0)
                    self.assertEqual(self.module._kernel.cache_info().currsize, 0)

    def test_boolean_configuration_rejected_without_runtime(self):
        meta = self.module.ArrayMeta
        args = (
            meta((2, 512), "float32"),
            meta((8, 64), "uint32"),
            meta((8, 8), "float32"),
            meta((8, 8), "float32"),
        )
        for kwargs in ({"bits": True}, {"bits": 4, "group_size": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.module.shape_plan(*args, **kwargs)
        self.assertEqual(self.module._kernel.cache_info().currsize, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
