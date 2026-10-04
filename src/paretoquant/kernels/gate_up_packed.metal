// Packed variant: each lane consumes a complete 3-byte or 4-byte weight pack.
// Scale/bias are applied once per pack, rather than once per decoded weight.
const uint row = thread_position_in_grid.y;
const uint lane = thread_position_in_grid.x;
if (row >= N) return;
constexpr uint VALUES = BITS == 6 ? 4 : 8;
constexpr uint PACK_BYTES = VALUES * BITS / 8;
const device uchar* gate_bytes = reinterpret_cast<const device uchar*>(gate_w);
const device uchar* up_bytes = reinterpret_cast<const device uchar*>(up_w);
float gate_acc = 0.0f;
float up_acc = 0.0f;
for (uint col = lane * VALUES; col < K; col += 32 * VALUES) {
    uint gate_pack;
    uint up_pack;
    if (BITS == 4) {
        const uint index = row * (K / 8) + col / 8;
        gate_pack = gate_w[index];
        up_pack = up_w[index];
    } else {
        const uint index = row * (K * BITS / 8) + (col / VALUES) * PACK_BYTES;
        gate_pack = uint(gate_bytes[index]) | (uint(gate_bytes[index + 1]) << 8)
            | (uint(gate_bytes[index + 2]) << 16);
        up_pack = uint(up_bytes[index]) | (uint(up_bytes[index + 1]) << 8)
            | (uint(up_bytes[index + 2]) << 16);
    }
    float gate_dot = 0.0f;
    float up_dot = 0.0f;
    float x_sum = 0.0f;
    #pragma unroll
    for (uint j = 0; j < VALUES; ++j) {
        const float value = float(x[col + j]);
        const uint shift = j * BITS;
        const uint mask = (1u << BITS) - 1u;
        gate_dot = fma(value, float((gate_pack >> shift) & mask), gate_dot);
        up_dot = fma(value, float((up_pack >> shift) & mask), up_dot);
        x_sum += value;
    }
    const uint group = row * (K / GROUP) + col / GROUP;
    gate_acc = fma(float(gate_s[group]), gate_dot, gate_acc);
    gate_acc = fma(float(gate_b[group]), x_sum, gate_acc);
    up_acc = fma(float(up_s[group]), up_dot, up_acc);
    up_acc = fma(float(up_b[group]), x_sum, up_acc);
}
gate_acc = simd_sum(gate_acc);
up_acc = simd_sum(up_acc);
if (lane == 0) {
    const float gate = float(T(gate_acc));
    const float up = float(T(up_acc));
    out[row] = T((gate / (1.0f + exp(-gate))) * up);
}
