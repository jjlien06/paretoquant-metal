"""CPU structural tests; no numerical or deployment admission claims."""

import unittest
from types import SimpleNamespace


class ShapeBomb:
    @property
    def shape(self):
        raise AssertionError("inactive input shape must not be inspected")


class EqualityBomb:
    def __eq__(self, other):
        raise AssertionError("metadata must not invoke tensor comparisons")

    def __bool__(self):
        raise AssertionError("metadata must not invoke tensor truth evaluation")


class ObservedNamespace(SimpleNamespace):
    def __setattr__(self, name, value):
        if name != "writes":
            self.writes.append(name)
        super().__setattr__(name, value)


class VerificationBackendTests(unittest.TestCase):
    def make(self, original, *, qmm=None, attention=None):
        from verification_backend import make_verification_forward

        self.native_qmm = lambda *args, **kwargs: "native qmm"
        self.native_attention = lambda *args, **kwargs: "native attention"
        self.mx = ObservedNamespace(writes=[], quantized_matmul=self.native_qmm)
        self.qwen2 = ObservedNamespace(
            writes=[], scaled_dot_product_attention=self.native_attention
        )
        self.qmm = qmm if qmm is not None else lambda *args, **kwargs: "injected qmm"
        self.attention = (
            attention if attention is not None else lambda *args, **kwargs: "injected attention"
        )
        return make_verification_forward(
            original,
            mx=self.mx,
            qwen2=self.qwen2,
            qmm=self.qmm,
            attention=self.attention,
        )

    def test_default_and_prefill_forward_unchanged_without_shape_access(self):
        model, inputs, cache, result, marker = (
            object(),
            ShapeBomb(),
            object(),
            object(),
            object(),
        )
        calls = []

        def original(*args, **kwargs):
            self.assertIs(self.mx.quantized_matmul, self.native_qmm)
            self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)
            calls.append((args, kwargs))
            return result

        forward = self.make(original)
        for kwargs in ({"marker": marker}, {"all_logits": False, "marker": marker}):
            self.assertIs(forward(model, inputs, cache, **kwargs), result)
            args, received = calls[-1]
            self.assertEqual(args, (model, inputs, cache))
            self.assertEqual(received, kwargs)

    def test_non_boolean_all_logits_rejected_before_shape_or_forward(self):
        calls = []
        forward = self.make(lambda *args, **kwargs: calls.append((args, kwargs)))
        for value in (None, 0, 1, "True", [], object(), EqualityBomb()):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "all_logits"):
                    forward(object(), ShapeBomb(), object(), all_logits=value)
                self.assertEqual(calls, [])
                self.assertEqual(self.mx.writes, [])
                self.assertEqual(self.qwen2.writes, [])
                self.assertIs(self.mx.quantized_matmul, self.native_qmm)
                self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_invalid_target_shapes_rejected_without_calls_or_patch_writes(self):
        calls = []
        forward = self.make(lambda *args, **kwargs: calls.append((args, kwargs)))
        bad_shapes = (
            None,
            (),
            (1,),
            (1, 2, 3),
            (2, 2),
            (0, 2),
            (1, 0),
            (1, -1),
            (1, 10),
            (True, 2),
            (1, True),
            (1.0, 2),
            (1, 2.0),
            (1, EqualityBomb()),
            EqualityBomb(),
            "(1, 2)",
        )
        for shape in bad_shapes:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    forward(object(), SimpleNamespace(shape=shape), object(), all_logits=True)
                self.assertEqual(calls, [])
                self.assertEqual(self.mx.writes, [])
                self.assertEqual(self.qwen2.writes, [])
        with self.assertRaisesRegex(ValueError, "shape"):
            forward(object(), object(), object(), all_logits=True)

    def test_verification_widths_patch_both_components_and_preserve_identities(self):
        model, cache, result, marker = object(), object(), object(), object()
        x, weights, scales, biases = (object() for _ in range(4))
        qmm_calls = []
        qmm_result = object()

        def injected_qmm(*args, **kwargs):
            qmm_calls.append((args, kwargs))
            return qmm_result

        def original(received_model, received_inputs, received_cache, **kwargs):
            self.assertIs(received_model, model)
            self.assertIs(received_inputs, inputs)
            self.assertIs(received_cache, cache)
            self.assertEqual(kwargs, {"all_logits": True, "marker": marker})
            self.assertIsNot(self.mx.quantized_matmul, self.native_qmm)
            self.assertIs(self.qwen2.scaled_dot_product_attention, self.attention)
            self.assertIs(
                self.mx.quantized_matmul(
                    x,
                    weights,
                    scales,
                    biases,
                    transpose=True,
                    mode="affine",
                    stream=None,
                    bits=8,
                    group_size=128,
                ),
                qmm_result,
            )
            self.assertEqual(
                qmm_calls[-1], ((x, weights, scales, biases), {"bits": 8, "group_size": 128})
            )
            return result

        forward = self.make(original, qmm=injected_qmm)
        for width in range(2, 10):
            for container in (tuple, list):
                with self.subTest(width=width, container=container):
                    inputs = SimpleNamespace(shape=container((1, width)))
                    self.assertIs(
                        forward(model, inputs, cache, all_logits=True, marker=marker), result
                    )
                    self.assertIs(self.mx.quantized_matmul, self.native_qmm)
                    self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_width_one_target_uses_unpatched_original(self):
        result = object()
        inputs, model, cache = SimpleNamespace(shape=(1, 1)), object(), object()

        def original(*args, **kwargs):
            self.assertEqual(args, (model, inputs, cache))
            self.assertEqual(kwargs, {"all_logits": True})
            self.assertIs(self.mx.quantized_matmul, self.native_qmm)
            self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)
            return result

        forward = self.make(original)
        self.assertIs(forward(model, inputs, cache, all_logits=True), result)
        self.assertEqual(self.mx.writes, [])
        self.assertEqual(self.qwen2.writes, [])

    def assert_qmm_rejected(self, field, value):
        calls = []
        operands = (object(), object(), object(), object())
        parameters = {"transpose": True, "mode": "affine", "stream": None}
        parameters[field] = value

        def original(*args, **kwargs):
            return self.mx.quantized_matmul(*operands, **parameters)

        forward = self.make(original, qmm=lambda *args, **kwargs: calls.append((args, kwargs)))
        with self.assertRaisesRegex(ValueError, field):
            forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True)
        self.assertEqual(calls, [])
        self.assertIs(self.mx.quantized_matmul, self.native_qmm)
        self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_qmm_rejects_transpose_not_exactly_true(self):
        for value in (False, None, 1, "True", EqualityBomb()):
            with self.subTest(value=value):
                self.assert_qmm_rejected("transpose", value)

    def test_qmm_rejects_mode_not_exact_string_affine(self):
        class StringSubclass(str):
            pass

        for value in (None, "mxfp4", "AFFINE", "", EqualityBomb(), StringSubclass("affine")):
            with self.subTest(value=value):
                self.assert_qmm_rejected("mode", value)

    def test_qmm_rejects_non_none_stream(self):
        for value in (False, 0, "gpu", object(), EqualityBomb()):
            with self.subTest(value=value):
                self.assert_qmm_rejected("stream", value)

    def test_qmm_rejects_missing_or_none_biases(self):
        for supplied in (False, True):
            with self.subTest(supplied=supplied):
                calls = []
                operands = (object(), object(), object())
                if supplied:
                    operands += (None,)

                def original(*args, **kwargs):
                    return self.mx.quantized_matmul(*operands, transpose=True)

                forward = self.make(
                    original, qmm=lambda *args, **kwargs: calls.append((args, kwargs))
                )
                with self.assertRaisesRegex(ValueError, "biases"):
                    forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True)
                self.assertEqual(calls, [])
                self.assertIs(self.mx.quantized_matmul, self.native_qmm)
                self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_forward_base_exception_restores_functions_and_cache_identity(self):
        class Abort(BaseException):
            pass

        failure = Abort("original failure")
        cache = []

        def original(model, inputs, received_cache, **kwargs):
            self.assertIs(received_cache, cache)
            received_cache.append("original mutation")
            self.assertIs(self.qwen2.scaled_dot_product_attention, self.attention)
            raise failure

        forward = self.make(original)
        with self.assertRaises(Abort) as caught:
            forward(object(), SimpleNamespace(shape=(1, 2)), cache, all_logits=True)
        self.assertIs(caught.exception, failure)
        self.assertEqual(cache, ["original mutation"])
        self.assertIs(self.mx.quantized_matmul, self.native_qmm)
        self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_injected_component_exceptions_restore_both_functions(self):
        for component in ("qmm", "attention"):
            with self.subTest(component=component):
                failure = RuntimeError(component)

                def raising(*args, **kwargs):
                    raise failure

                def original(*args, **kwargs):
                    if component == "qmm":
                        return self.mx.quantized_matmul(
                            object(), object(), object(), object(), transpose=True
                        )
                    return self.qwen2.scaled_dot_product_attention(object(), mask="causal")

                forward = self.make(original, **{component: raising})
                with self.assertRaises(RuntimeError) as caught:
                    forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True)
                self.assertIs(caught.exception, failure)
                self.assertIs(self.mx.quantized_matmul, self.native_qmm)
                self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_nested_scopes_restore_outer_bindings_then_native_bindings(self):
        from verification_backend import make_verification_forward

        for inner_raises in (False, True):
            with self.subTest(inner_raises=inner_raises):
                failure = RuntimeError("inner")
                inner_result, outer_result = object(), object()

                def inner_attention(*args, **kwargs):
                    return "inner attention"

                inner_qmm_calls = []

                def inner_original(*args, **kwargs):
                    self.assertIs(self.qwen2.scaled_dot_product_attention, inner_attention)
                    self.mx.quantized_matmul(object(), object(), object(), object(), transpose=True)
                    if inner_raises:
                        raise failure
                    return inner_result

                def outer_original(*args, **kwargs):
                    outer_qmm = self.mx.quantized_matmul
                    outer_attention = self.qwen2.scaled_dot_product_attention
                    if inner_raises:
                        with self.assertRaises(RuntimeError) as caught:
                            inner(*args, **kwargs)
                        self.assertIs(caught.exception, failure)
                    else:
                        self.assertIs(inner(*args, **kwargs), inner_result)
                    self.assertIs(self.mx.quantized_matmul, outer_qmm)
                    self.assertIs(self.qwen2.scaled_dot_product_attention, outer_attention)
                    return outer_result

                outer = self.make(outer_original)
                inner = make_verification_forward(
                    inner_original,
                    mx=self.mx,
                    qwen2=self.qwen2,
                    qmm=lambda *args, **kwargs: inner_qmm_calls.append((args, kwargs)),
                    attention=inner_attention,
                )
                self.assertIs(
                    outer(object(), SimpleNamespace(shape=(1, 9)), object(), all_logits=True),
                    outer_result,
                )
                self.assertEqual(len(inner_qmm_calls), 1)
                self.assertIs(self.mx.quantized_matmul, self.native_qmm)
                self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_attention_is_direct_injection_with_native_sdpa_owned_by_component(self):
        query, key, value, mask, stream, result = (object() for _ in range(6))
        calls = []

        def native_sdpa(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        def injected_attention(*args, **kwargs):
            return native_sdpa(*args, **kwargs)

        def original(*args, **kwargs):
            self.assertIs(self.qwen2.scaled_dot_product_attention, injected_attention)
            return self.qwen2.scaled_dot_product_attention(
                query, key, value, scale=0.5, mask=mask, stream=stream
            )

        forward = self.make(original, attention=injected_attention)
        self.assertIs(
            forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True), result
        )
        self.assertEqual(
            calls, [((query, key, value), {"scale": 0.5, "mask": mask, "stream": stream})]
        )

    def test_operands_and_quantization_metadata_are_forwarded_without_evaluation(self):
        class ArrayDouble(EqualityBomb):
            @property
            def shape(self):
                raise AssertionError("operand shape is owned by the injected component")

            def eval(self):
                raise AssertionError("no array evaluation")

            def __array__(self):
                raise AssertionError("no array materialization")

        operands = tuple(ArrayDouble() for _ in range(4))
        bits, group_size, result = object(), object(), object()
        calls = []

        def injected_qmm(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        def original(*args, **kwargs):
            return self.mx.quantized_matmul(
                *operands, transpose=True, bits=bits, group_size=group_size
            )

        forward = self.make(original, qmm=injected_qmm)
        self.assertIs(
            forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True), result
        )
        received_args, received_kwargs = calls[0]
        for received, expected in zip(received_args, operands):
            self.assertIs(received, expected)
        self.assertIs(received_kwargs["bits"], bits)
        self.assertIs(received_kwargs["group_size"], group_size)

    def test_second_patch_setup_failure_unwinds_first_patch(self):
        calls = []
        forward = self.make(lambda *args, **kwargs: calls.append((args, kwargs)))
        del self.qwen2.scaled_dot_product_attention
        with self.assertRaises(AttributeError):
            forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True)
        self.assertEqual(calls, [])
        self.assertIs(self.mx.quantized_matmul, self.native_qmm)
        self.assertFalse(hasattr(self.qwen2, "scaled_dot_product_attention"))

    def test_factory_and_inactive_paths_do_not_touch_missing_patch_targets(self):
        from verification_backend import make_verification_forward

        result, mx, qwen2 = object(), object(), object()
        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        forward = make_verification_forward(original, mx=mx, qwen2=qwen2, qmm=None, attention=None)
        for kwargs in ({}, {"all_logits": False}):
            self.assertIs(forward(object(), ShapeBomb(), object(), **kwargs), result)
        self.assertIs(
            forward(object(), SimpleNamespace(shape=(1, 1)), object(), all_logits=True), result
        )
        self.assertEqual(len(calls), 3)

    def test_input_shape_is_metadata_only_without_input_evaluation(self):
        class InputDouble(EqualityBomb):
            shape = (1, 2)

            def eval(self):
                raise AssertionError("input evaluation forbidden")

            def __array__(self):
                raise AssertionError("input materialization forbidden")

        result = object()
        forward = self.make(lambda *args, **kwargs: result)
        self.assertIs(forward(object(), InputDouble(), object(), all_logits=True), result)

    def test_container_subclasses_cannot_evaluate_during_shape_rejection(self):
        class HostileTuple(tuple):
            def __len__(self):
                raise AssertionError("no subclass shape length evaluation")

            def __iter__(self):
                raise AssertionError("no subclass shape iteration")

        calls = []
        forward = self.make(lambda *args, **kwargs: calls.append((args, kwargs)))
        with self.assertRaisesRegex(ValueError, "shape"):
            forward(
                object(), SimpleNamespace(shape=HostileTuple((1, 2))), object(), all_logits=True
            )
        self.assertEqual(calls, [])
        self.assertEqual(self.mx.writes, [])
        self.assertEqual(self.qwen2.writes, [])

    def test_unsupported_qmm_keyword_rejected_without_injected_call(self):
        calls = []

        def original(*args, **kwargs):
            return self.mx.quantized_matmul(
                object(), object(), object(), object(), transpose=True, unknown_option=True
            )

        forward = self.make(original, qmm=lambda *args, **kwargs: calls.append((args, kwargs)))
        with self.assertRaises(TypeError):
            forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True)
        self.assertEqual(calls, [])
        self.assertIs(self.mx.quantized_matmul, self.native_qmm)
        self.assertIs(self.qwen2.scaled_dot_product_attention, self.native_attention)

    def test_qmm_default_quantization_values_pass_through(self):
        calls = []

        def original(*args, **kwargs):
            return self.mx.quantized_matmul(object(), object(), object(), object(), transpose=True)

        forward = self.make(original, qmm=lambda *args, **kwargs: calls.append((args, kwargs)))
        forward(object(), SimpleNamespace(shape=(1, 2)), object(), all_logits=True)
        self.assertEqual(calls[0][1], {"bits": 4, "group_size": 64})


if __name__ == "__main__":
    unittest.main()
