"""Independent FP32 attention reference and native baseline."""


def make_inputs(s, d, family, seed):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = (torch.randn(4, 8, s, d, generator=generator).to(torch.bfloat16).npu().contiguous() for _ in range(3))
    if family == "uniform":
        q.zero_()
    elif family == "peaked":
        q.mul_(4)
        k.mul_(4)
    return q, k, v


def native_call(q, k, v, causal):
    import torch
    import torch_npu
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=q.device), 1) if causal else None

    def call():
        result = torch_npu.npu_fusion_attention(q, k, v, head_num=q.shape[1], input_layout="BNSD", pse=None,
                                                padding_mask=None, atten_mask=mask, scale=q.shape[-1]**-0.5,
                                                keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647,
                                                inner_precise=0, prefix=None, actual_seq_qlen=None,
                                                actual_seq_kvlen=None, sparse_mode=2 if causal else 0)
        return result[:3]

    return call


def reference(q, k, v, causal):
    """Blocked FP32 reference, with HF32 disabled by the caller."""
    import torch
    b, h, s, d = q.shape
    qf, kf, vf = (x.reshape(b * h, s, d).float() for x in (q, k, v))
    out = torch.empty_like(qf)
    maxes = torch.empty((b * h, s), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maxes)
    for lo in range(0, s, 256):
        hi = min(lo + 256, s)
        end = hi if causal else s
        scores = (qf[:, lo:hi] @ kf[:, :end].transpose(1, 2)) * d**-0.5
        if causal:
            mask = torch.arange(end, device=q.device)[None, :] > torch.arange(lo, hi, device=q.device)[:, None]
            scores.masked_fill_(mask, float("-inf"))
        m = scores.amax(-1)
        maxes[:, lo:hi] = m
        sums[:, lo:hi] = (scores - m[..., None]).exp().sum(-1)
        out[:, lo:hi] = scores.softmax(-1) @ vf[:, :end]
    return out.reshape(b, h, s, d), maxes.reshape(b, h, s, 1), sums.reshape(b, h, s, 1)


def check_outputs(actuals, refs):
    import torch
    checks = {}
    expected_shapes = (refs[0].shape, (*refs[1].shape[:3], 8), (*refs[2].shape[:3], 8))
    for name, actual, ref, shape, dtype, tol in zip(("O", "max", "sum"), actuals, refs, expected_shapes,
                                                    (torch.bfloat16, torch.float32, torch.float32),
                                                    (.005, .0001, .0001)):
        assert actual.shape == shape and actual.dtype == dtype, (name, actual.shape, actual.dtype)
        assert bool(torch.isfinite(actual).all()), name
        error = float((actual.float() - ref).abs().max()) / max(float(ref.abs().max()), 1e-30)
        checks[name] = error
        assert error <= tol, (name, error, tol)
    return checks
