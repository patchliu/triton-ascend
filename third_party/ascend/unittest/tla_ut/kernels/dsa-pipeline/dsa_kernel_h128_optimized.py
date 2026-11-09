"""Sparse MLA prefill expressed as semantic pipeline tasks.

Gather, QK, online softmax, PV and output update own logical data products.
The compiler owns buffer versions, fill/drain and all synchronization.
"""
from dataclasses import dataclass
import triton
import triton.language as tl
import triton.language.extra.tla as tla


@dataclass(frozen=True)
class Config:
    """Static tile sizes. Dispatch depends only on the public tensor shape."""

    heads: int
    gather_rows: int


def default_config(heads):
    if heads != 128:
        raise ValueError('This shared-KV kernel supports H128')
    # A reuse tile spans all heads; Cube computes two logical head partitions.
    gather_rows = 8
    return Config(128, gather_rows)


def device_cores(device):
    """Query runtime core limits, rather than assuming a physical core count."""
    import torch
    from triton.runtime import driver
    with torch.npu.device(device):
        properties = driver.active.utils.get_device_properties(device)
    cube, vector = properties['num_aicore'], properties['num_vectorcore']
    if cube <= 0 or vector != 2 * cube:
        raise ValueError(f'This A5 kernel requires a 1:2 Cube/Vector ratio; got {cube}:{vector}')
    return cube, vector


@triton.jit
def _pack_kv(KV, KR, PACKED):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = tl.arange(0, 1024)
    latent = tl.load(KV + row[:, None] * 512 + col[None, :], col[None, :] < 512, 0)
    rope = tl.load(KR + row[:, None] * 64 + col[None, :] - 512, (col[None, :] >= 512) & (col[None, :] < 576), 0)
    tl.store(
        PACKED + row[:, None] * 576 + col[None, :],
        tl.where(col[None, :] < 512, latent, rope),
        col[None, :] < 576,
    )


@tla.jit
def _sparse_mla(Q, QR, IDS, IDS64, OUT, MAX, SUM, PACKED_KV, SQ: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
                TOPK: tl.constexpr, BM: tl.constexpr, GR: tl.constexpr, CORES: tl.constexpr):
    BN: tl.constexpr = 128
    CM: tl.constexpr = 64
    VM: tl.constexpr = CM // 2
    ALL_VM: tl.constexpr = BM // 2
    PARTS: tl.constexpr = BM // CM
    OUT_ROWS: tl.constexpr = 8
    GRP: tl.constexpr = GR + 1
    tl.static_assert(BM == 128 and H == 128 and TOPK == 2048 and L == 8192)
    for token in tl.range(tl.program_id(0), SQ, CORES):
        sub = tla.sub_block_id()
        # A query owns both head tiles; these are logical head partitions.
        queries = (tla.alloc((36, CM // 16, 16, 16), tl.bfloat16,
                             space='l1'), tla.alloc((36, CM // 16, 16, 16), tl.bfloat16, space='l1'))
        maximum = tla.alloc((64, ), tl.float32, space='ub')
        denominator = tla.alloc((64, ), tl.float32, space='ub')
        acc = tla.alloc((ALL_VM, 512), tl.float32, space='ub')
        with tla.scope.cube():
            for query_part in tl.static_range(PARTS):
                query = queries[query_part]
                tla.load_gm_to_l1(tla.view(Q + (token * H + query_part * CM) * 512, (CM, 512)),
                                  tla.view(query, (32, CM // 16, 16, 16)))
                tla.load_gm_to_l1(tla.view(QR + (token * H + query_part * CM) * 64, (CM, 64)),
                                  tla.view(query, (4, CM // 16, 16, 16), offsets=(32, 0, 0, 0)))
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                tla.simd.vsts(maximum[0], tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64))
                tla.simd.vsts(denominator[0], zero)
                for row in tl.range(ALL_VM):
                    for col in tl.static_range(8):
                        tla.simd.vsts(acc[row, col * 64], zero)

        # Reuse tile = 128 heads; Cube tile = 64 heads. KV is gathered once.
        # Stage 0 gathers, stage 1 computes both head partitions.
        for tile in tla.pipeline(TOPK // BN):
            kv = tla.alloc((36, BN // 16, 16, 16), tl.bfloat16, space='l1')
            probabilities = (tla.alloc((BN // 16, CM // 16, 16, 16), tl.bfloat16,
                                       space='l1'), tla.alloc((BN // 16, CM // 16, 16, 16), tl.bfloat16, space='l1'))
            qk_l0c = tla.alloc((BN // 16, CM // 16, 16, 16), tl.float32, space='l0c')
            pv_l0c = tla.alloc((32, CM // 16, 16, 16), tl.float32, space='l0c')
            scores = (tla.alloc((VM, BN), tl.float32, space='ub'), tla.alloc((VM, BN), tl.float32, space='ub'))
            partial = tla.alloc((VM, 512), tl.float32, space='ub')
            prob = tla.alloc((BN // 16, VM * 16 + 16), tl.bfloat16, space='ub')
            ids = tla.alloc((BN, ), tl.int32, space='ub')
            correction = tla.alloc((64, ), tl.float32, space='ub')
            packed = tla.alloc((36, GRP * 16), tl.bfloat16, space='ub')

            with tla.scope.vector().task('gather', stage=0):
                tla.load_gm_to_ub(tla.view(IDS + token * TOPK + tile * BN, (BN, )), ids)
                for group in tl.range(BN // (2 * GR)):
                    position = sub * (BN // 2) + group * GR
                    for pair in tl.static_range(GR // 2):
                        pair_ids = tl.load(IDS64 + (token * TOPK + tile * BN + position) // 2 + pair)
                        safe0 = (pair_ids & 0xFFFFFFFF).to(tl.int32) & (L - 1)
                        safe1 = (pair_ids >> 32).to(tl.int32) & (L - 1)
                        tla.load_gm_to_ub(tla.view(PACKED_KV + safe0 * 576, (36, 16)),
                                          tla.view(packed, (36, 16), offsets=(0, pair * 2 * 16)))
                        tla.load_gm_to_ub(tla.view(PACKED_KV + safe1 * 576, (36, 16)),
                                          tla.view(packed, (36, 16), offsets=(0, (pair * 2 + 1) * 16)))
                    tla.copy_ub_to_l1(
                        tla.view(packed, (36, GR * 16)),
                        tla.view(tla.reshape(kv, (36, BN * 16)), (36, GR * 16), offsets=(0, position * 16)))

            for head_part in tl.static_range(PARTS):
                with tla.scope.cube().task(f'qk_{head_part}', stage=1):
                    query, score_part = queries[head_part], scores[head_part]
                    tla.cube.mmad(query, kv, qk_l0c, transpose_b=True, clear_accum=True)
                    tla.copy_l0c_to_ub(
                        qk_l0c, score_part,
                        tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

                with tla.scope.vector().task(f'softmax_{head_part}', stage=1):
                    score_part = scores[head_part]
                    with tla.scope.simd():
                        neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        scale = tla.simd.vsplat(1.0 / 24.0, tl.float32)
                        limit = tla.simd.vsplat(L, tl.int32)
                        valid0 = tla.simd.vcmp(tla.simd.vlds(ids, (0, )), limit, op=tla.simd.CMP.ULT)
                        valid1 = tla.simd.vcmp(tla.simd.vlds(ids, (64, )), limit, op=tla.simd.CMP.ULT)
                        for row in tl.range(VM):
                            state = head_part * VM + row
                            x0 = tla.simd.vsel(valid0, tla.simd.vmul(tla.simd.vlds(score_part[row, 0]), scale), neg)
                            x1 = tla.simd.vsel(valid1, tla.simd.vmul(tla.simd.vlds(score_part[row, 64]), scale), neg)
                            old_m = tla.simd.vlds(maximum[state], dist='BRC_B32')
                            m = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_m)
                            alpha = tla.simd.vexp(tla.simd.vsub(old_m, m))
                            e0 = tla.simd.vexp(tla.simd.vsub(x0, m))
                            e1 = tla.simd.vexp(tla.simd.vsub(x1, m))
                            total = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                            total = tla.simd.vadd(
                                total, tla.simd.vmul(tla.simd.vlds(denominator[state], dist='BRC_B32'), alpha))
                            tla.simd.vsts(maximum[state], m, dist='ONEPT_B32')
                            tla.simd.vsts(denominator[state], total, dist='ONEPT_B32')
                            tla.simd.vsts(correction[state], alpha, dist='ONEPT_B32')
                            values, unused = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                              tla.simd.vtruncf(e1, tl.bfloat16))
                            tla.simd.vsstb(values, prob[0, row * 16], (VM + 1) * 16)

                with tla.scope.vector().task(f'publish_{head_part}', stage=1):
                    probability = tla.reshape(probabilities[head_part], (BN // 16, CM * 16))
                    tla.copy_ub_to_l1(tla.view(prob, (BN // 16, VM * 16)),
                                      tla.view(probability, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)))

            for head_part in tl.static_range(PARTS):
                with tla.scope.cube().task(f'pv_{head_part}', stage=1):
                    probability = probabilities[head_part]
                    tla.cube.mmad(probability, tla.view(kv, (32, BN // 16, 16, 16)), pv_l0c, clear_accum=True)
                    tla.copy_l0c_to_ub(
                        pv_l0c, partial,
                        tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

                with tla.scope.vector().task(f'accumulate_{head_part}', stage=1):
                    with tla.scope.simd():
                        for row in tl.range(VM):
                            state = head_part * VM + row
                            alpha = tla.simd.vlds(correction[state], dist='BRC_B32')
                            for col in tl.static_range(8):
                                value = tla.simd.vfma(tla.simd.vlds(partial[row, col * 64]),
                                                      tla.simd.vlds(acc[state, col * 64]), alpha)
                                tla.simd.vsts(acc[state, col * 64], value)

        output = tla.alloc((OUT_ROWS, 512), tl.bfloat16, space='ub')
        with tla.scope.vector():
            for head_part in tl.static_range(PARTS):
                for row_block in tl.range(VM // OUT_ROWS):
                    with tla.scope.simd():
                        for local_row in tl.range(OUT_ROWS):
                            state = head_part * VM + row_block * OUT_ROWS + local_row
                            den = tla.simd.vlds(denominator[state], dist='BRC_B32')
                            for col in tl.static_range(4):
                                b0 = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(acc[state, col * 128]), den),
                                                      tl.bfloat16)
                                b1 = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(acc[state, col * 128 + 64]), den),
                                                      tl.bfloat16)
                                values, unused = tla.simd.vdintlv(b0, b1)
                                tla.simd.vsts(output[local_row, col * 128], values)
                    tla.store_ub_to_gm(
                        output,
                        tla.view(OUT + (token * H + head_part * CM + sub * VM + row_block * OUT_ROWS) * 512,
                                 (OUT_ROWS, 512), strides=(512, 1)))
                tla.store_ub_to_gm(tla.view(maximum, (VM, ), offsets=(head_part * VM, )),
                                   tla.view(MAX + token * H + head_part * CM + sub * VM, (VM, )))
                tla.store_ub_to_gm(tla.view(denominator, (VM, ), offsets=(head_part * VM, )),
                                   tla.view(SUM + token * H + head_part * CM + sub * VM, (VM, )))


def forward(q, kv, qr, kr, indices, config=None, *, core_limit=None):
    """Return BF16 output and FP32 online-softmax maximum and sum.

    Outputs have shapes [1,8192,H,512], [1,1,8192,H], [1,1,8192,H].
    core_limit is a diagnostic grid cap, not a fabricated device property.
    Production launches use the runtime core count. Jobs cover every query and
    head tile cyclically, without assuming that heads divide the core count.
    """
    import torch
    import torch_npu
    if q.ndim != 4:
        raise ValueError('q must have shape [1,8192,H,512]')
    h = q.shape[2]
    tensors = q, kv, qr, kr, indices
    expected = [(1, 8192, h, 512), (1, 8192, 1, 512), (1, 8192, h, 64), (1, 8192, 1, 64), (1, 8192, 1, 2048)]
    if h != 128 or any(tuple(t.shape) != s for t, s in zip(tensors, expected)):
        raise ValueError('Expected B1/SQ8192/KV8192/H128/TopK2048')
    if q.device.type != 'npu' or any(t.device != q.device or not t.is_contiguous() for t in tensors):
        raise ValueError('Inputs must be contiguous tensors on the same NPU')
    if any(t.dtype != torch.bfloat16 for t in tensors[:4]) or indices.dtype != torch.int32:
        raise ValueError('Expected BF16 data and int32 indices')
    config = config or default_config(h)
    if config != default_config(h):
        raise ValueError('Only the validated shape-specific tile configuration is exposed')
    bm, gr = config.heads, config.gather_rows
    cores, _ = device_cores(q.device)
    if core_limit is not None:
        if not 1 <= core_limit <= cores:
            raise ValueError('core_limit must be between 1 and the runtime Cube count')
        cores = core_limit
    cores = min(cores, 8192 * (h // bm))
    packed_kv = torch.empty((8192, 576), dtype=q.dtype, device=q.device)
    output = torch.empty_like(q)
    maximum = torch.empty((1, 1, 8192, h), dtype=torch.float32, device=q.device)
    denominator = torch.empty_like(maximum)
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
    indices64 = indices.view(torch.int64)
    args = (q, qr, indices, indices64, output, maximum, denominator, packed_kv, 8192, h, 8192, 2048, bm, gr, cores)
    with torch.npu.device(q.device):
        _pack_kv[(1024, )](kv, kr, packed_kv, **options)
        _sparse_mla[(cores, )](*args, **options)
    return output, maximum, denominator
