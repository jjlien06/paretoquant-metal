"""Source-level native arithmetic seam and per-request runtime integration."""

import ast
from pathlib import Path

ROOT = Path(__file__).parent


def test_native_receive_hook_precedes_local_graph_without_arithmetic_edits():
    tree = ast.parse((ROOT / "phased_decode.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "phased_forward")
    assert "receive_stage" in [a.arg for a in fn.args.kwonlyargs]
    text = ast.get_source_segment((ROOT / "phased_decode.py").read_text(), fn)
    assert (
        text.index("mx.eval(h)  # finish network wait")
        < text.index("receive_stage()")
        < text.index("fa_mask =")
    )


def test_native_arithmetic_ast_is_identical_to_frozen024_except_receive_hook_events():
    import hashlib

    # CPU-derived from the frozen 024 native forward, excluding only emits/hook.
    reference = "32d22ee1cd9d5b55a6bfb674a783ca300f68cf13cbeef5e1e80d22932055e1aa"

    def normalized(path):
        tree = ast.parse(path.read_text())
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "phased_forward"
        )
        if any(arg.arg == "receive_stage" for arg in fn.args.kwonlyargs):
            index = next(
                i for i, arg in enumerate(fn.args.kwonlyargs) if arg.arg == "receive_stage"
            )
            fn.args.kwonlyargs.pop(index)
            fn.args.kw_defaults.pop(index)

        class NormalizeEvidence(ast.NodeTransformer):
            def visit_Expr(self, node):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "emit"
                ):
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                if (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "receive_stage"
                ):
                    return None
                return self.generic_visit(node)

        return ast.dump(NormalizeEvidence().visit(fn), include_attributes=False)

    assert hashlib.sha256(normalized(ROOT / "phased_decode.py").encode()).hexdigest() == reference


def test_both_backend_constructors_explicit_per_request_mode_and_compound_dispatch():
    tree = ast.parse((ROOT / "mlx_backend.py").read_text())
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assert "prefill_mode" in [a.arg for a in init.args.kwonlyargs]
    text = (ROOT / "mlx_backend.py").read_text()
    assert "run_rank0_pipeline(" in text and "run_rank1_pipeline(" in text
    runtime = (ROOT / "workload_runtime.py").read_text()
    assert runtime.count("prefill_mode=") >= 2
    assert "source_request_id" in runtime
