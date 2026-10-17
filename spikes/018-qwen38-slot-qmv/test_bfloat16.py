from slot_qmv import ArrayMeta, shape_plan


def test_bfloat16_metadata_preserves_native_compute_dtype():
    plan = shape_plan(
        ArrayMeta((1, 2, 5120), 'bfloat16'),
        ArrayMeta((17408, 640), 'uint32'),
        ArrayMeta((17408, 80), 'bfloat16'),
        ArrayMeta((17408, 80), 'bfloat16'),
        bits=4, group_size=64,
    )
    assert plan.slots == 2
    assert plan.output_dtype == 'bfloat16'
    assert plan.output_shape == (1, 2, 17408)
