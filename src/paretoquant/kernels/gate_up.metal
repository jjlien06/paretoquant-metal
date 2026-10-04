// Original ParetoQuant kernel: one SIMD group computes one gate/up output row.
// Inputs use MLX's little-endian affine packed weight format.
const uint row = thread_position_in_grid.y;
const uint lane = thread_position_in_grid.x;
if (row >= N) return;
float gate_acc = 0.0f;
float up_acc = 0.0f;
for (uint col = lane; col < K; col += 32) {
    const uint bit_offset = col * BITS;
    const uint word = bit_offset / 32;
    const uint shift = bit_offset % 32;
    const uint base = row * (K * BITS / 32);
    uint gate_code = gate_w[base + word] >> shift;
    uint up_code = up_w[base + word] >> shift;
    if (shift + BITS > 32) {
        gate_code |= gate_w[base + word + 1] << (32 - shift);
        up_code |= up_w[base + word + 1] << (32 - shift);
    }
    const uint mask = (1u << BITS) - 1u;
    const uint group = row * (K / GROUP) + col / GROUP;
    const float gate = float(gate_s[group]) * float(gate_code & mask)
        + float(gate_b[group]);
    const float up = float(up_s[group]) * float(up_code & mask)
        + float(up_b[group]);
    const float value = float(x[col]);
    gate_acc = fma(value, gate, gate_acc);
    up_acc = fma(value, up, up_acc);
}
gate_acc = simd_sum(gate_acc);
up_acc = simd_sum(up_acc);
if (lane == 0) {
    // Match the stock projection output dtype before the fused nonlinearity.
    const float gate = float(T(gate_acc));
    const float up = float(T(up_acc));
    out[row] = T((gate / (1.0f + exp(-gate))) * up);
}
