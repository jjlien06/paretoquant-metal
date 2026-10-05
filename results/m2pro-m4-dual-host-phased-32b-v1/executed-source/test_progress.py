"""Stdlib validation and progress tests; never import a GPU runtime."""

import unittest

from phased_decode import validate_options


class OptionsTests(unittest.TestCase):
    def test_generation_is_bounded(self):
        for cap in (0, -1, 513, True):
            with self.assertRaises(ValueError):
                validate_options([1], cap, 256, 4096)
        with self.assertRaises(ValueError):
            validate_options([], 16, 256, 4096)
        with self.assertRaises(ValueError):
            validate_options([1] * 4090, 16, 256, 4096)
        validate_options([1, 2], 16, 256, 4096)

    def test_progress_is_flushed_with_memory_units(self):
        import io
        import json

        from probe import phase_reporter

        output = io.StringIO()
        report = phase_reporter(
            rank=1,
            output=output,
            memory=lambda: {"active_mlx_bytes": 7, "peak_mlx_bytes": 9, "cache_mlx_bytes": 2},
        )
        report({"phase": "loading_before"})
        event = json.loads(output.getvalue())
        self.assertEqual(event["rank"], 1)
        self.assertEqual(event["active_mlx_bytes"], 7)
        self.assertIn("process_maxrss_units", event)
        self.assertIn("elapsed_seconds", event)
        self.assertIn("pid", event)

    def test_loader_progress_is_optional(self):
        import inspect

        from stream_weights import load_from_tensors

        self.assertIsNone(inspect.signature(load_from_tensors).parameters["progress"].default)

    def test_two_rank_ownership_cannot_assign_all_layers_to_rank_one(self):
        from types import SimpleNamespace

        from phased_decode import ownership

        group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
        model = SimpleNamespace(
            model_type="qwen2",
            args=SimpleNamespace(num_hidden_layers=2),
            model=SimpleNamespace(
                pipeline_rank=1,
                pipeline_size=2,
                start_idx=0,
                end_idx=2,
                pipeline_layers=[object(), object()],
            ),
        )
        with self.assertRaises(ValueError):
            ownership(model, group)
