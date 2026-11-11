"""Independent chunk coefficients for the GDN pipeline."""


def prepare_wy(q, k, v, g, beta, chunk, scale, *, solve_tril=None):
    import torch

    tokens, hk, dk = q.shape
    _, hv, dv = v.shape
    count = tokens // chunk
    group = hv // hk
    q = q.reshape(count, chunk, hk, dk).transpose(1, 2).repeat_interleave(group, dim=1).float()
    k = k.reshape(count, chunk, hk, dk).transpose(1, 2).repeat_interleave(group, dim=1).float()
    v = v.reshape(count, chunk, hv, dv).transpose(1, 2).float()
    beta = beta.reshape(count, chunk, hv).transpose(1, 2).float().unsqueeze(-1)
    log_decay = g.reshape(count, chunk, hv).transpose(1, 2).cumsum(-1)
    decay = log_decay.exp().unsqueeze(-1)
    pair_decay = (log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)).tril().exp().tril()

    lower = (beta * (k @ k.mT) * pair_decay).tril(-1)
    rhs = torch.cat((beta * v, beta * decay * k), dim=-1)
    if solve_tril is None:
        # L is nilpotent. Apply (I - L)(I + L²)(I + L⁴)... to both RHS blocks.
        power = -lower
        del lower
        for span in range(chunk.bit_length() - 1):
            rhs = rhs + power @ rhs
            if 2**(span + 1) < chunk:
                power = power @ power
    else:
        # FLA solves (I + L)^-1 for each contiguous BT-row block of [B,T,H,BT].
        # Packing chunk/head into T gives H=1 without transposing the coefficient blocks.
        inverse = solve_tril(lower.reshape(1, -1, 1, chunk), output_dtype=torch.float32).reshape_as(lower)
        del lower
        rhs = inverse @ rhs
        del inverse
    u, w = rhs.split((dv, dk), dim=-1)

    q = q * scale
    q_decayed = q * decay
    k_decayed = k * (log_decay[..., -1:] - log_decay).exp().unsqueeze(-1)
    a = (q @ k.mT) * pair_decay
    end_decay = decay[..., -1, 0]
    return (u.mT.contiguous(), w.bfloat16().contiguous(), q_decayed.bfloat16().contiguous(),
            k_decayed.bfloat16().contiguous(), a.bfloat16().contiguous(), end_decay.contiguous())
