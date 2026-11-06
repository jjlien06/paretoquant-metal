// Experimental independent array kernel: no parity/speed claim, no decoder hook.
slot_qmv_impl<T, GROUP_SIZE, BITS, SLOTS, K, N, FAST, ROWS, SIMDS, MODE>(
    packed_weights, scales, affine_bias, x, y,
    threadgroup_position_in_grid,
    thread_position_in_threadgroup.y,
    thread_position_in_threadgroup.x);
