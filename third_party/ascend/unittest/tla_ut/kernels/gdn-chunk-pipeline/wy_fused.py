"""Device WY preparation from TND inputs into the recurrence's coefficient layouts."""

import math

import triton
import triton.language as tl


@triton.jit
def _recompute_w_u(K, V, BETA, PREFIX, INVERSE, U, W, HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
                   BT: tl.constexpr, BD: tl.constexpr, PACKED_INVERSE: tl.constexpr):
    job = tl.program_id(0)
    chunk, head = job // HV, job % HV
    qk_head = head // (HV // HK)
    row = tl.arange(0, BT)
    col = tl.program_id(1) * BD + tl.arange(0, BD)
    token = chunk * BT + row
    value_row = token * HV + head
    key_row = token * HK + qk_head

    inverse_row = job * BT + row if PACKED_INVERSE else value_row
    inverse = tl.load(INVERSE + inverse_row[:, None] * BT + row[None, :])
    beta = tl.load(BETA + value_row).to(tl.float32)
    decay = tl.exp2(tl.load(PREFIX + value_row))
    v = tl.load(V + value_row[:, None] * D + col[None, :]).to(tl.float32)
    k = tl.load(K + key_row[:, None] * D + col[None, :]).to(tl.float32)
    u = tl.dot(inverse, beta[:, None] * v, input_precision='ieee')
    w = tl.dot(inverse, (beta * decay)[:, None] * k, input_precision='ieee')

    tl.store(U + (job * D + col[:, None]) * BT + row[None, :], tl.trans(u))
    tl.store(W + (job * BT + row[:, None]) * D + col[None, :], w.to(tl.bfloat16))


@triton.jit
def _prepare_q_k_a(Q, K, BETA, PREFIX, Q_DECAYED, K_DECAYED, A, END_DECAY, LOWER, HK: tl.constexpr, HV: tl.constexpr,
                   D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, SCALE: tl.constexpr, WRITE_LOWER: tl.constexpr):
    job = tl.program_id(0)
    chunk, head = job // HV, job % HV
    qk_head = head // (HV // HK)
    row = tl.arange(0, BT)
    token = chunk * BT + row
    prefix = tl.load(PREFIX + token * HV + head)
    end = tl.load(PREFIX + ((chunk + 1) * BT - 1) * HV + head)
    q_scale = (SCALE * tl.exp2(prefix))[:, None]
    k_scale = tl.exp2(end - prefix)[:, None]
    key_row = token * HK + qk_head
    out_row = job * BT + row

    qk = tl.full((BT, BT), 0, tl.float32)
    if WRITE_LOWER:
        kk = tl.full((BT, BT), 0, tl.float32)
    for start in range(D // BD):
        col = start * BD + tl.arange(0, BD)
        q = tl.load(Q + key_row[:, None] * D + col[None, :])
        k = tl.load(K + key_row[:, None] * D + col[None, :])
        kt = tl.trans(k)
        qk = tl.dot(q, kt, qk)
        if WRITE_LOWER:
            kk = tl.dot(k, kt, kk)
        dst = out_row[:, None] * D + col[None, :]
        tl.store(Q_DECAYED + dst, (q.to(tl.float32) * q_scale).to(tl.bfloat16))
        tl.store(K_DECAYED + dst, (k.to(tl.float32) * k_scale).to(tl.bfloat16))

    causal = row[:, None] >= row[None, :]
    delta = tl.where(causal, prefix[:, None] - prefix[None, :], 0.0)
    pair = tl.exp2(delta)
    a = tl.where(causal, qk * SCALE * pair, 0.0)
    tl.store(A + out_row[:, None] * BT + row[None, :], a.to(tl.bfloat16))
    tl.store(END_DECAY + job, tl.exp2(end))
    if WRITE_LOWER:
        beta = tl.load(BETA + token * HV + head).to(tl.float32)
        lower = tl.where(row[:, None] > row[None, :], (beta[:, None] * kk) * pair, 0.0)
        tl.store(LOWER + out_row[:, None] * BT + row[None, :], lower)


def prepare_wy(q, k, v, g, beta, chunk, scale, *, cumsum, solve_kkt=None, solve_tril=None):
    import torch

    tokens, hk, dim = q.shape
    hv = v.shape[1]
    chunks = tokens // chunk
    prefix = cumsum(g[None], chunk_size=chunk, scale=math.log2(math.e), output_dtype=torch.float32)
    u = torch.empty((chunks, hv, dim, chunk), dtype=torch.float32, device=q.device)
    w = torch.empty((chunks, hv, chunk, dim), dtype=torch.bfloat16, device=q.device)
    q_decay, k_decay = torch.empty_like(w), torch.empty_like(w)
    a = torch.empty((chunks, hv, chunk, chunk), dtype=torch.bfloat16, device=q.device)
    end_decay = torch.empty((chunks, hv), dtype=torch.float32, device=q.device)
    width = 64
    split = solve_tril is not None
    lower = torch.empty((1, tokens * hv, 1, chunk), dtype=torch.float32, device=q.device) if split else None
    _prepare_q_k_a[(chunks * hv,)](q, k, beta, prefix, q_decay, k_decay, a, end_decay, lower, hk, hv, dim, chunk, width,
                                 scale, split)
    if split:
        inverse = solve_tril(lower, output_dtype=torch.float32)
        del lower
    else:
        # The pinned fused kernel writes the lower blocks in interleaved TND layout.
        inverse = torch.zeros((1, tokens, hv, chunk), dtype=torch.float32, device=q.device)
        solve_kkt[(chunks, hv)](k=k[None], g=prefix, beta=beta[None], A=inverse, cu_seqlens=None, chunk_indices=None,
                                T=tokens, H=hk, HV=hv, K=dim, BT=chunk, BC=16)
    _recompute_w_u[(chunks * hv, dim // width)](k, v, beta, prefix, inverse, u, w, hk, hv, dim, chunk, width, split)
    return u, w, q_decay, k_decay, a, end_decay
