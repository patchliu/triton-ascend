"""Groupwise INT4 weight-only matmul with on-chip dequantization."""

import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def matmul(A, PACKED, SCALE, O, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, GROUP: tl.constexpr,
           CORES: tl.constexpr, COMPUTE_STAGE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
           GROUPS_PER_TILE: tl.constexpr, UNPACK_BYTES: tl.constexpr):
    VECTOR_COLS: tl.constexpr = 128
    WORDS: tl.constexpr = BN // 8
    WORDS_PER_VECTOR: tl.constexpr = VECTOR_COLS // 8
    BYTES: tl.constexpr = BN // 2
    BYTES_PER_VECTOR: tl.constexpr = VECTOR_COLS // 2
    HALF_K: tl.constexpr = GROUP // 2
    K_TILE: tl.constexpr = GROUP * GROUPS_PER_TILE
    PACK_PITCH: tl.constexpr = (HALF_K + 1) * 16
    N_TILES: tl.constexpr = N // BN
    tl.static_assert(M % BM == 0 and N % BN == 0 and K % K_TILE == 0 and GROUP % 32 == 0)

    for job in tl.range(tl.program_id(0), (M // BM) * N_TILES, CORES):
        row = job // N_TILES * BM
        col = job % N_TILES * BN
        sub = tla.sub_block_id()
        accumulator = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space='l0c')

        for block in tla.pipeline(K // K_TILE):
            activation = tla.alloc((K_TILE // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
            weight = tla.alloc((BN // 16, K_TILE // 16, 16, 16), tl.bfloat16, space='l1')
            if UNPACK_BYTES:
                bytes_ub = tla.alloc((HALF_K, BYTES), tl.uint8, space='ub')
            else:
                words = tla.alloc((HALF_K, WORDS), tl.int32, space='ub')
            scale = tla.alloc((BN,), tl.bfloat16, space='ub')
            nz = tla.alloc((BN // 16, PACK_PITCH), tl.bfloat16, space='ub')
            k_start = block * K_TILE

            with tla.scope.cube().task('load_activation', stage=0):
                tla.load_gm_to_l1(tla.view(A + row * K + k_start, (BM, K_TILE), strides=(K, 1)), activation)

            with tla.scope.vector().task('dequantize_weight', stage=0):
                for slot in tl.static_range(GROUPS_PER_TILE):
                    group = block * GROUPS_PER_TILE + slot
                    offset = (k_start + slot * GROUP + sub * HALF_K) * (N // 8) + col // 8
                    if UNPACK_BYTES:
                        tla.load_gm_to_ub(tla.view(PACKED + offset * 4, (HALF_K, BYTES), strides=(N // 2, 1)), bytes_ub)
                    else:
                        tla.load_gm_to_ub(tla.view(PACKED + offset, (HALF_K, WORDS), strides=(N // 8, 1)),
                                          words)
                        flat_words = tla.reshape(words, (HALF_K * WORDS,))
                    tla.load_gm_to_ub(tla.view(SCALE + group * N + col, (BN,)), scale)
                    with tla.scope.simd():
                        zero = tla.simd.vsplat(0.0, tl.bfloat16)
                        lane = tla.simd.vci(0)
                        nibble_index = tla.simd.vand(lane, tla.simd.vsplat(7, tl.int32))
                        high_shift = tla.simd.vshls(tla.simd.vsub(tla.simd.vsplat(7, tl.int32), nibble_index), 2)
                        sign_shift = tla.simd.vsplat(28, tl.int32)
                        word = tla.simd.vshr(lane, tla.simd.vsplat(3, tl.int32))
                        word_lo = tla.simd.vbitcast(word, 'uint32')
                        word_hi = tla.simd.vbitcast(tla.simd.vadds(word, WORDS_PER_VECTOR // 2), 'uint32')
                        for col_group in tl.range(BN // VECTOR_COLS):
                            scale_register = tla.simd.vlds(scale[col_group * VECTOR_COLS])
                            if UNPACK_BYTES:
                                scale_even = tla.simd.vextf(scale_register, tl.float32)
                                pairs = tla.simd.vbitcast(scale_register, 'uint32')
                                odd_scale_bits = tla.simd.vshr(pairs, tla.simd.vsplat(16, tl.uint32))
                                scale_odd = tla.simd.vextf(tla.simd.vbitcast(odd_scale_bits, 'bf16'), tl.float32)
                            else:
                                scale0, scale1 = tla.simd.vintlv(scale_register, zero)
                                scale_lo = tla.simd.vextf(scale0, tl.float32)
                                scale_hi = tla.simd.vextf(scale1, tl.float32)
                            for k_row in tl.range(HALF_K):
                                start = k_row * WORDS + col_group * WORDS_PER_VECTOR
                                if UNPACK_BYTES:
                                    # One distributed load puts each byte in
                                    # an i32 lane. Its two nibbles are the
                                    # adjacent even and odd weight columns.
                                    packed_bytes = tla.simd.vlds(bytes_ub[k_row, col_group * BYTES_PER_VECTOR],
                                                                 dist='UNPK4_B8')
                                    byte_lanes = tla.simd.vbitcast(packed_bytes, 'i32')
                                    even = tla.simd.vshr(tla.simd.vshls(byte_lanes, 28), sign_shift)
                                    odd = tla.simd.vshr(tla.simd.vshls(byte_lanes, 24), sign_shift)
                                    even = tla.simd.vmul(tla.simd.vcvt(even, tl.float32), scale_even)
                                    odd = tla.simd.vmul(tla.simd.vcvt(odd, tl.float32), scale_odd)
                                    even_bits = tla.simd.vbitcast(tla.simd.vtruncf(even, tl.bfloat16), 'i32')
                                    odd_bits = tla.simd.vbitcast(tla.simd.vtruncf(odd, tl.bfloat16), 'i32')
                                    pairs = tla.simd.vor(even_bits, tla.simd.vshls(odd_bits, 16))
                                    values = tla.simd.vbitcast(pairs, 'bf16')
                                else:
                                    lo = tla.simd.vgather(flat_words[start], word_lo)
                                    hi = tla.simd.vgather(flat_words[start], word_hi)
                                    # Move the nibble to the sign bit, then
                                    # arithmetic-shift it into a signed INT32.
                                    lo = tla.simd.vshr(tla.simd.vshl(lo, high_shift), sign_shift)
                                    hi = tla.simd.vshr(tla.simd.vshl(hi, high_shift), sign_shift)
                                    lo = tla.simd.vmul(tla.simd.vcvt(lo, tl.float32), scale_lo)
                                    hi = tla.simd.vmul(tla.simd.vcvt(hi, tl.float32), scale_hi)
                                    values, unused = tla.simd.vdintlv(tla.simd.vtruncf(lo, tl.bfloat16),
                                                                      tla.simd.vtruncf(hi, tl.bfloat16))
                                tla.simd.vsstb(values, nz[col_group * (VECTOR_COLS // 16), k_row * 16], PACK_PITCH)
                    tla.copy_ub_to_l1(
                        tla.view(nz, (BN // 16, HALF_K * 16)),
                        tla.view(tla.reshape(weight, (BN // 16, K_TILE * 16)), (BN // 16, HALF_K * 16),
                                 offsets=(0, (slot * GROUP + sub * HALF_K) * 16)))

            with tla.scope.cube().task('matmul', stage=COMPUTE_STAGE):
                if block == 0:
                    tla.cube.mmad(activation, weight, accumulator, clear_accum=True)
                else:
                    tla.cube.mmad(activation, weight, accumulator, clear_accum=False)

        with tla.scope.cube():
            tla.store_l0c_to_gm(accumulator, tla.view(O + row * N + col, (BM, BN), strides=(N, 1)),
                                tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                        pre_quant=tla.FixpipePreQuantMode.F322BF16))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(activation, packed, scale, *, prefetch=True, m_tile=None, n_tile=None, groups_per_tile=1,
                 unpack_bytes=True):
    """Use native ND packing, with signed INT4 nibbles ordered along N."""
    import torch

    if activation.ndim != 2 or packed.ndim != 2 or scale.ndim != 2:
        raise ValueError('Expected 2-D activation, packed weight and group scales')
    m, k = activation.shape
    n = packed.shape[1] * 8
    groups = scale.shape[0]
    if m_tile is None:
        m_tile = 64 if m % 128 else 128
    if n_tile is None:
        n_tile = 512 if n % 512 == 0 else (256 if n % 256 == 0 else 128)
    if (m_tile not in (64, 128) or n_tile not in (128, 256, 512) or not m or m % m_tile or not k or not n or n % n_tile
            or packed.shape[0] != k
            or scale.shape[1] != n):
        raise ValueError('Expected M aligned to its 64- or 128-row tile, N aligned to 128 and consistent weights')
    if groups < 2 or k % groups or k // groups not in (64, 128):
        raise ValueError('Expected groupwise INT4 with K-group size 64 or 128')
    if groups_per_tile not in (1, 2) or groups % groups_per_tile:
        raise ValueError('Expected a K tile containing one or two whole scale groups')
    if activation.dtype != torch.bfloat16 or packed.dtype != torch.int32 or scale.dtype != torch.bfloat16:
        raise ValueError('Expected BF16 activation/scales and packed INT32 weights')
    if activation.device.type != 'npu' or any(x.device != activation.device or not x.is_contiguous()
                                              for x in (activation, packed, scale)):
        raise ValueError('Expected contiguous tensors on the same Ascend NPU')
    cube, vector = device_cores(activation.device)
    if cube <= 0 or vector != 2 * cube:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {cube}:{vector}')
    cores = min(cube, (m // m_tile) * (n // n_tile))
    packed_input = packed.view(torch.uint8) if unpack_bytes else packed

    def call():
        output = torch.empty((m, n), dtype=activation.dtype, device=activation.device)
        matmul[(cores,)](activation, packed_input, scale, output, m, k, n, k // groups, cores, int(prefetch),
                        m_tile, n_tile, groups_per_tile, unpack_bytes,
                        sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
        return output

    call.metadata = dict(m_tile=m_tile, n_tile=n_tile, k_tile=k // groups * groups_per_tile,
                         grid=cores, compute_stage=int(prefetch), unpack_bytes=unpack_bytes)
    return call
