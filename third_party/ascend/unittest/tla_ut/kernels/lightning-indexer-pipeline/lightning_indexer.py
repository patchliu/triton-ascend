"""BF16 Lightning Indexer with tiled scoring and separate device TopK."""

from functools import partial

import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction


@partial(TLAJITFunction, do_not_specialize=['tile_count'])
def index_scores(Q, K, W, O, tile_count, B: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr,
                 BQ: tl.constexpr, CORES: tl.constexpr, DOT_STAGE: tl.constexpr, REDUCE_STAGE: tl.constexpr,
                 HEAD_CHAINS: tl.constexpr, FUSED_RELU: tl.constexpr):
    D: tl.constexpr = 128
    BN: tl.constexpr = 128
    VN: tl.constexpr = BN // 2
    M: tl.constexpr = BQ * H
    QUERY_TILES: tl.constexpr = SQ // BQ
    tl.static_assert(SQ % BQ == 0 and SK % BN == 0 and M % 16 == 0)

    for job in tl.range(tl.program_id(0), B * QUERY_TILES, CORES):
        batch = job // QUERY_TILES
        row_start = job % QUERY_TILES * BQ
        q_token = batch * SQ + row_start
        q = tla.alloc((D // 16, M // 16, 16, 16), tl.bfloat16, space='l1')
        weight = tla.alloc((M,), tl.float32, space='ub')
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + q_token * H * D, (M, D)), q)
        with tla.scope.vector():
            # The same weights serve every key tile in this work item.
            weight_bf16 = tla.alloc((M,), tl.bfloat16, space='ub')
            tla.load_gm_to_ub(tla.view(W + q_token * H, (M,)), weight_bf16)
            with tla.scope.simd():
                for head in tl.range(M):
                    value = tla.simd.vextf(tla.simd.vlds(weight_bf16[head], dist='BRC_B16'), tl.float32)
                    tla.simd.vsts(weight[head], value, dist='ONEPT_B32')

        # Keep one-tile runs in the same pipeline domain. A literal trip count
        # of 1 can make backend canonicalization erase the domain's scf.for.
        for tile in tla.pipeline(tile_count):
            k = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
            qk = tla.alloc((BN // 16, M // 16, 16, 16), tl.float32, space='l0c')
            products = tla.alloc((M, VN), tl.float32, space='ub')
            score = tla.alloc((BQ, VN), tl.float32, space='ub')

            with tla.scope.cube().task('load_key', stage=0):
                tla.load_gm_to_l1(tla.view(K + (batch * SK + tile * BN) * D, (BN, D)), k)

            with tla.scope.cube().task('dot', stage=DOT_STAGE):
                tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(qk, products, tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    pre_relu=tla.FixpipePreReluMode.NORMAL_RELU if FUSED_RELU else tla.FixpipePreReluMode.NO_RELU,
                    dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('reduce_heads', stage=REDUCE_STAGE):
                key_start = tile * BN + tla.sub_block_id() * VN
                with tla.scope.simd():
                    keys = tla.simd.vci(key_start.to(tl.int32))
                    zero = tla.simd.vfull(0.0, tl.float32, VN)
                    masked = tla.simd.vfull(float('-inf'), tl.float32, VN)
                    for row in tl.range(BQ):
                        if HEAD_CHAINS == 4:
                            total0, total1, total2, total3 = zero, zero, zero, zero
                            for group in tl.range(H // 4):
                                base = row * H + group * 4
                                qk0 = tla.simd.vlds(products[base, 0])
                                qk1 = tla.simd.vlds(products[base + 1, 0])
                                qk2 = tla.simd.vlds(products[base + 2, 0])
                                qk3 = tla.simd.vlds(products[base + 3, 0])
                                if not FUSED_RELU:
                                    qk0 = tla.simd.vmax(qk0, zero)
                                    qk1 = tla.simd.vmax(qk1, zero)
                                    qk2 = tla.simd.vmax(qk2, zero)
                                    qk3 = tla.simd.vmax(qk3, zero)
                                w0 = tla.simd.vlds(weight[base], dist='BRC_B32')
                                w1 = tla.simd.vlds(weight[base + 1], dist='BRC_B32')
                                w2 = tla.simd.vlds(weight[base + 2], dist='BRC_B32')
                                w3 = tla.simd.vlds(weight[base + 3], dist='BRC_B32')
                                total0 = tla.simd.vfma(total0, qk0, w0)
                                total1 = tla.simd.vfma(total1, qk1, w1)
                                total2 = tla.simd.vfma(total2, qk2, w2)
                                total3 = tla.simd.vfma(total3, qk3, w3)
                            total = tla.simd.vadd(tla.simd.vadd(total0, total1), tla.simd.vadd(total2, total3))
                        else:
                            total = zero
                            for head in tl.range(H):
                                dot = tla.simd.vlds(products[row * H + head, 0])
                                if not FUSED_RELU:
                                    dot = tla.simd.vmax(dot, zero)
                                w = tla.simd.vlds(weight[row * H + head], dist='BRC_B32')
                                total = tla.simd.vfma(total, dot, w)
                        visible = (SK - SQ + row_start + row + 1).to(tl.int32)
                        valid = tla.simd.vcmps(keys, visible, op=tla.simd.CMP.LT)
                        tla.simd.vsts(score[row, 0], tla.simd.vsel(valid, total, masked))
                tla.store_ub_to_gm(score, tla.view(O + q_token * SK + key_start, (BQ, VN), (SK, 1)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_scores(query, key, weights, *, prefetch=True, reduce_with_dot=False, head_chains=1, fused_relu=False):
    """Return a BSND scoring call for right-aligned causal full-length inputs."""
    import torch

    b, sq, h, d = query.shape
    if (d != 128 or b <= 0 or sq <= 0 or h not in (16, 32, 64) or
            key.ndim != 4 or key.shape[0] != b or key.shape[2:] != (1, d) or
            weights.shape != (b, sq, h)):
        raise ValueError('Expected BSND Q [B,Sq,H,128], K [B,Sk,1,128] and W [B,Sq,H]')
    sk = key.shape[1]
    if sk <= 0 or sk % 128:
        raise ValueError('Expected a positive key length divisible by 128')
    if query.device.type != 'npu' or any(x.device != query.device or x.dtype != torch.bfloat16 or not x.is_contiguous()
                                         for x in (query, key, weights)):
        raise ValueError('Expected contiguous BF16 query, key and weights on one NPU')
    if head_chains not in (1, 4):
        raise ValueError('Expected 1 or 4 head accumulation chains')
    aic, aiv = device_cores(query.device)
    if aic <= 0 or aiv != 2 * aic:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {aic}:{aiv}')
    bq = 4 if sq % 4 == 0 and h <= 32 else 2 if sq % 2 == 0 else 1
    cores = min(aic, b * (sq // bq))
    dot_stage = int(prefetch)
    reduce_stage = dot_stage if reduce_with_dot else 2 * dot_stage

    def call():
        scores = torch.empty((b, sq, sk), dtype=torch.float32, device=query.device)
        index_scores[(cores,)](query, key, weights, scores, sk // 128, b, sq, sk, h, bq, cores, dot_stage, reduce_stage,
                                head_chains, fused_relu,
                                sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
        return scores

    call.metadata = dict(grid=cores, query_tile=bq, key_tile=128, dot_stage=dot_stage, reduce_stage=reduce_stage,
                         head_chains=head_chains, fused_relu=fused_relu)
    return call


def prepare_call(query, key, weights, *, topk=2048, prefetch=True, reduce_with_dot=False, head_chains=1,
                 fused_relu=False):
    """Return an index-only call matching BSND, sparse_mode=3 and return_value=False."""
    import torch

    if topk < 1 or topk > 2048:
        raise ValueError('Expected TopK in [1, 2048]')
    score = prepare_scores(query, key, weights, prefetch=prefetch, reduce_with_dot=reduce_with_dot,
                           head_chains=head_chains, fused_relu=fused_relu)
    b, sq = query.shape[:2]
    sk = key.shape[1]
    device = query.device

    def call():
        if sk < topk:
            # The native index-only path returns all visible positions in
            # ascending order when the row has fewer candidates than TopK.
            valid = (sk - sq + 1 + torch.arange(sq, dtype=torch.int32, device=device))[:, None]
            positions = torch.arange(topk, dtype=torch.int32, device=device)[None, :]
            selected = torch.where(positions < valid, positions, -1).expand(b, -1, -1).contiguous()
        else:
            selected = torch.topk(score(), topk, dim=-1, sorted=False).indices.to(torch.int32)
            if sk - sq + 1 < topk:
                valid = (sk - sq + 1 + torch.arange(sq, dtype=torch.int32, device=device))[:, None]
                positions = torch.arange(topk, dtype=torch.int32, device=device)[None, :]
                short_rows = torch.where(positions < valid, positions, -1)
                selected = torch.where(valid < topk, short_rows, selected)
        return selected.unsqueeze(2)

    call.metadata = dict(score.metadata, topk=topk)
    return call
