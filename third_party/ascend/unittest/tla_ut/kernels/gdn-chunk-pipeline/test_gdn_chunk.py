"""GDN prefill checks from raw inputs through output and final state."""

import hashlib
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('gdn_chunk.py')
BLOCKS = Path(__file__).with_name('wy_blocks.py')
HEAD_COUNTS = (16, 32)
TOKENS, CHUNK = 32768, 64
VALUE_TILE = int(os.environ.get('TLA_GDN_CHUNK_VALUE_TILE', '32'))
WY = os.environ.get('TLA_GDN_CHUNK_WY', 'doubling')
DIRECT_TND = os.environ.get('TLA_GDN_CHUNK_DIRECT_TND') == '1'
INPUTS = (('normal', 1103), ('normal', 1104), ('zero_key', 1103), ('slow_decay', 1103))
ROUNDS, SAMPLES = 3, 40
OUTPUT = Path(os.environ.get('TLA_GDN_RESULTS', 'gdn-results'))


@pytest.mark.parametrize('hv', HEAD_COUNTS)
@pytest.mark.parametrize('prefetch', (False, True))
def test_ir(hv, prefetch, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    signature = dict(U='*fp32', W='*bf16', Q_DECAYED='*bf16', K_DECAYED='*bf16', A='*bf16', END_DECAY='*fp32',
                     INITIAL='*bf16', O='*bf16', FINAL='*bf16')
    constants = dict(CHUNKS=TOKENS // CHUNK, HV=hv, BT=CHUNK, DV=128, DK=128, BV=VALUE_TILE, CORES=28,
                     COMPUTE_STAGE=int(prefetch), DIRECT_TND=DIRECT_TND)
    if output_dir is not None:
        output_dir = output_dir / f'hv{hv}-bv{VALUE_TILE}-prefetch{int(prefetch)}'
    module, context = compile_offline(kernel.chunk_recurrence, signature, constants, output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_coefficients', 'load_values', 'predict', 'form_residual', 'apply_chunk', 'advance_state'):
        assert f'hivm.task_name = "{task}"' in lowered


def inputs(torch, tokens, hv, family, seed, device):
    hk, dim = 16, 128
    rng = torch.Generator(device='cpu').manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn((tokens, hk, dim), generator=rng), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn((tokens, hk, dim), generator=rng), dim=-1).bfloat16()
    v = torch.randn((tokens, hv, dim), generator=rng).bfloat16()
    g = torch.empty((tokens, hv)).uniform_(-0.5, -0.03, generator=rng)
    beta = torch.empty((tokens, hv)).uniform_(0.1, 0.9, generator=rng).bfloat16()
    state = (torch.randn((1, hv, dim, dim), generator=rng) * 0.2).bfloat16()
    if family == 'zero_key':
        k.zero_()
    elif family == 'slow_decay':
        g.uniform_(-0.006, -0.001, generator=rng)
        beta.fill_(0.9)
    return tuple(x.to(device) for x in (q, k, v, g, beta, state))


def recurrent_reference(torch, q, k, v, g, beta, initial):
    """Independent token recurrence for short sequences."""
    group = v.shape[1] // q.shape[1]
    q = q.float().repeat_interleave(group, dim=1) * q.shape[-1]**-0.5
    k = k.float().repeat_interleave(group, dim=1)
    state = initial[0].float()
    output = []
    for query, key, value, log_decay, step in zip(q, k, v.float(), g, beta.float()):
        state = state * log_decay.exp()[:, None, None]
        delta = step[:, None] * (value - (state @ key.unsqueeze(-1)).squeeze(-1))
        state = state + delta.unsqueeze(-1) * key.unsqueeze(-2)
        output.append((state @ query.unsqueeze(-1)).squeeze(-1))
    return torch.stack(output), state.unsqueeze(0)


def chunk_reference(torch, q, k, v, g, beta, initial):
    """FP32 golden for the long shape, with CPU triangular solve and no BF16 intermediates."""
    group = v.shape[1] // q.shape[1]
    dk = q.shape[-1]
    state = initial[0].float()
    output = []
    for start in range(0, q.shape[0], CHUNK):
        end = start + CHUNK
        keys = k[start:end].transpose(0, 1).float().repeat_interleave(group, dim=0)
        queries = q[start:end].transpose(0, 1).float().repeat_interleave(group, dim=0) * dk**-0.5
        values = v[start:end].transpose(0, 1).float()
        steps = beta[start:end].T.float().unsqueeze(-1)
        prefix = g[start:end].T.cumsum(-1)
        decay = prefix.exp().unsqueeze(-1)
        pair = (prefix.unsqueeze(-1) - prefix.unsqueeze(-2)).tril().exp().tril()
        lower = (steps * (keys @ keys.mT) * pair).tril(-1)
        predicted = (keys @ state.mT) * decay
        residual = torch.linalg.solve_triangular(lower, steps * (values - predicted), upper=False,
                                                  unitriangular=True)
        out = (queries @ state.mT) * decay + ((queries @ keys.mT) * pair) @ residual
        output.append(out.transpose(0, 1))
        suffix = (prefix[..., -1:] - prefix).exp().unsqueeze(-1)
        state = state * decay[..., -1:, :] + residual.mT @ (keys * suffix)
    return torch.cat(output), state.unsqueeze(0)


def assert_accuracy(torch, actual, expected):
    errors = {}
    for name, x, y in zip(('output', 'state'), actual, expected):
        assert x.shape == y.shape and x.dtype == torch.bfloat16
        x = x.float()
        assert torch.isfinite(x).all()
        delta = x - y
        maximum = (delta.abs().amax() / y.abs().amax().clamp_min(1e-20)).item()
        rms = (delta.square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-20)).item()
        assert maximum <= 0.018 and rms <= 0.012, (name, maximum, rms)
        errors[name] = dict(normalized_max_error=maximum, relative_rms=rms)
    return errors


def cpu_sweep_inverse(torch, lower, diagonal_size):
    width = lower.shape[-1]
    if width == diagonal_size:
        inverse = torch.eye(width, dtype=lower.dtype).expand_as(lower)
        for pivot in range(width - 1):
            inverse = torch.addcmul(inverse, -lower[..., :, pivot:pivot + 1], inverse[..., pivot:pivot + 1, :])
        return inverse
    half = width // 2
    left = cpu_sweep_inverse(torch, lower[..., :half, :half], diagonal_size)
    right = cpu_sweep_inverse(torch, lower[..., half:, half:], diagonal_size)
    result = torch.zeros_like(lower)
    result[..., :half, :half] = left
    result[..., half:, half:] = right
    result[..., half:, :half] = -(right @ lower[..., half:, :half]) @ left
    return result


def cpu_fused_coefficients(torch, q, k, v, g, beta, wy):
    tokens, hk, dim = q.shape
    hv = v.shape[1]
    heads = torch.arange(hv) // (hv // hk)
    q = q.reshape(-1, CHUNK, hk, dim)[:, :, heads].transpose(1, 2).float()
    k = k.reshape(-1, CHUNK, hk, dim)[:, :, heads].transpose(1, 2).float()
    v = v.reshape(-1, CHUNK, hv, dim).transpose(1, 2).float()
    beta = beta.reshape(-1, CHUNK, hv).transpose(1, 2).unsqueeze(-1).float()
    prefix = g.reshape(-1, CHUNK, hv).cumsum(1).transpose(1, 2) * math.log2(math.e)
    pair = (prefix[..., :, None] - prefix[..., None, :]).tril().exp2().tril()
    k0, k1 = k.split(64, dim=-1)
    split = wy != 'fused_layout'
    kk = k0 @ k0.mT + k1 @ k1.mT if split else k @ k.mT
    lower = (beta * kk * pair).tril(-1)
    if split:
        # The split kernel writes each [BT,BT] system as a contiguous FLA time block.
        packed = lower.reshape(1, tokens * hv, 1, CHUNK)
        systems = packed.view(-1, CHUNK, CHUNK)
        assert packed.is_contiguous() and packed.data_ptr() == lower.data_ptr()
        assert torch.count_nonzero(systems.triu()) == 0
    else:
        systems = lower
    eye = torch.eye(CHUNK, dtype=torch.float32).expand_as(systems)
    expected_inverse = torch.linalg.solve_triangular(systems, eye, upper=False, unitriangular=True)
    if wy in ('split16_layout', 'split32_layout'):
        diagonal_size = 16 if wy == 'split16_layout' else 32
        inverse = cpu_sweep_inverse(torch, systems, diagonal_size)
        torch.testing.assert_close(inverse, expected_inverse, rtol=2e-5, atol=2e-6)
    else:
        inverse = expected_inverse
    inverse = inverse.view_as(lower)
    decay = prefix.exp2().unsqueeze(-1)
    u = (inverse @ (beta * v)).mT.contiguous()
    w = (inverse @ ((beta * decay) * k)).bfloat16().contiguous()
    scale = dim**-0.5
    q_decay = (q * (scale * decay)).bfloat16().contiguous()
    k_decay = (k * (prefix[..., -1:] - prefix).exp2().unsqueeze(-1)).bfloat16().contiguous()
    q0, q1 = q.split(64, dim=-1)
    a = ((q0 @ k0.mT + q1 @ k1.mT) * scale * pair).bfloat16().contiguous()
    return u, w, q_decay, k_decay, a, decay[..., -1, 0].contiguous()


@pytest.mark.parametrize('hv', HEAD_COUNTS)
@pytest.mark.parametrize('wy', ('doubling', 'fla_layout', 'fused_layout', 'split_layout', 'split16_layout',
                                'split32_layout'))
def test_cpu_reference(hv, wy):
    torch = pytest.importorskip('torch')
    data = inputs(torch, 2 * CHUNK, hv, 'slow_decay', 1103, 'cpu')
    expected = recurrent_reference(torch, *data)
    for actual, exact in zip(chunk_reference(torch, *data), expected):
        torch.testing.assert_close(actual, exact, rtol=5e-5, atol=1e-6)

    q, k, v, g, beta, initial = data

    def cpu_solve_tril(packed, *, output_dtype):
        # FLA's [B,T,H,BT] contract with packed independent 64-row systems.
        assert packed.shape == (1, q.shape[0] * hv, 1, CHUNK) and packed.is_contiguous()
        assert packed.dtype == output_dtype == torch.float32
        lower = packed.reshape(-1, CHUNK, CHUNK)
        assert torch.count_nonzero(lower.triu()) == 0
        eye = torch.eye(CHUNK, dtype=output_dtype).expand_as(lower)
        return torch.linalg.solve_triangular(lower, eye, upper=False, unitriangular=True).reshape_as(packed)

    if wy in ('fused_layout', 'split_layout', 'split16_layout', 'split32_layout'):
        coefficients = cpu_fused_coefficients(torch, q, k, v, g, beta, wy)
    else:
        solver = cpu_solve_tril if wy == 'fla_layout' else None
        coefficients = load_kernel(BLOCKS).prepare_wy(q, k, v, g, beta, CHUNK, q.shape[-1]**-0.5, solve_tril=solver)
    u, w, q_decay, k_decay, a, alpha = coefficients
    assert u.dtype == torch.float32 and w.dtype == torch.bfloat16
    state = initial[0].float()
    output = []
    rows_per_vector = VALUE_TILE // 2
    for chunk in range(u.shape[0]):
        # Reproduce each Vector sub-block's slice and BF16 operands with CPU matmuls.
        out = torch.empty_like(u[chunk])
        for start in range(0, state.shape[1], rows_per_vector):
            rows = slice(start, start + rows_per_vector)
            current = state[:, rows]
            s_operand = current.bfloat16().float()
            residual = (u[chunk, :, rows] - s_operand @ w[chunk].float().mT).bfloat16().float()
            out[:, rows] = s_operand @ q_decay[chunk].float().mT + residual @ a[chunk].float().mT
            state[:, rows] = current * alpha[chunk, :, None, None] + residual @ k_decay[chunk].float()
        output.append(out.permute(2, 0, 1).bfloat16())
    output_tnd = torch.cat(output)
    if DIRECT_TND:
        direct = torch.full_like(output_tnd, float('nan'))
        dv = v.shape[-1]
        for chunk, tile in enumerate(output):
            for head in range(hv):
                for start in range(0, dv, VALUE_TILE):
                    offset = (chunk * CHUNK * hv + head) * dv + start
                    dst = direct.as_strided((CHUNK, VALUE_TILE), (hv * dv, 1), offset)
                    dst.copy_(tile[:, head, start:start + VALUE_TILE])
        assert torch.equal(direct, output_tnd)
        output_tnd = direct
    assert_accuracy(torch, (output_tnd, state[None].bfloat16()), expected)


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('GDN chunk device tests require the run stage')
    torch = pytest.importorskip('torch')
    torch_npu = pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    previous = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    try:
        yield torch, torch_npu, load_kernel(OPERATOR)
    finally:
        torch.npu.matmul.allow_hf32 = previous


def prepared(runtime, data, implementation):
    torch, torch_npu, kernel = runtime
    if implementation != 'ascendc':
        return kernel.prepare_call(*data, prefetch=implementation == 'pipeline', value_tile=VALUE_TILE, wy=WY,
                                   direct_tnd=DIRECT_TND)
    backend = os.environ.get('TLA_GDN_NATIVE', 'cann')
    if backend == 'sgl':
        sgl = pytest.importorskip('sgl_kernel_npu')
        native = getattr(torch.ops.npu, 'chunk_gated_delta_rule', None)
        library = Path(sgl.__file__).parent / 'lib' / 'libsgl_kernel_npu.so'
    elif backend == 'cann':
        native = getattr(torch_npu, 'npu_chunk_gated_delta_rule', None)
        library = None
    else:
        raise ValueError('TLA_GDN_NATIVE must be cann or sgl')
    if native is None:
        pytest.skip(f'{backend} does not provide ChunkGatedDeltaRule')
    q, k, v, g, beta, initial = data
    lengths = torch.tensor([q.shape[0]], dtype=torch.int32, device=q.device)

    def call():
        try:
            return native(q, k, v, beta=beta, initial_state=initial, actual_seq_lengths=lengths,
                          scale=q.shape[-1]**-0.5, g=g)
        except RuntimeError as error:
            message = str(error)
            missing_config = ('JSON configuration file of operator aclnnChunkGatedDeltaRule' in message
                              and 'cannot be found' in message)
            missing_binary = 'Op ChunkGatedDeltaRule does not has any binary' in message
            if backend == 'cann' and 'error code is 561103' in message and (missing_config or missing_binary):
                pytest.skip('CANN ChunkGatedDeltaRule device kernel is unavailable (561103)')
            raise

    call.metadata = dict(backend=backend)
    if library is not None and library.is_file():
        call.metadata.update(library=str(library), sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    return call


@pytest.mark.parametrize('hv', HEAD_COUNTS)
@pytest.mark.parametrize('implementation', ('pipeline', 'stage0', 'ascendc'))
def test_run(runtime, hv, implementation):
    torch, _, _ = runtime
    device = torch.device('npu', torch.npu.current_device())
    for family, seed in INPUTS:
        host = inputs(torch, 2 * CHUNK, hv, family, seed, 'cpu')
        expected = tuple(x.to(device) for x in recurrent_reference(torch, *host))
        data = tuple(x.to(device) for x in host)
        call = prepared(runtime, data, implementation)
        for _ in range(3):
            assert_accuracy(torch, call(), expected)
            assert torch.equal(data[-1], host[-1].to(device))


@pytest.mark.parametrize('hv', HEAD_COUNTS)
@pytest.mark.parametrize('implementation', ('pipeline', 'stage0', 'ascendc'))
def test_run_long(runtime, hv, implementation):
    torch, _, _ = runtime
    device = torch.device('npu', torch.npu.current_device())
    host = inputs(torch, TOKENS, hv, *INPUTS[0], 'cpu')
    data = tuple(x.to(device) for x in host)
    call = prepared(runtime, data, implementation)
    expected = tuple(x.to(device) for x in chunk_reference(torch, *host))
    assert_accuracy(torch, call(), expected)
    assert torch.equal(data[-1], host[-1].to(device))


@pytest.mark.parametrize('hv', HEAD_COUNTS)
def test_perf(request, runtime, hv):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 to profile on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Run performance tests without pytest-xdist')
    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'

    import triton
    from triton.backends.ascend.utils import _get_npucompiler_path

    torch, _, kernel = runtime
    device = torch.device('npu', torch.npu.current_device())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'chunk-hv{hv}-bv{VALUE_TILE}-', dir=OUTPUT))
    spec = dict(bs=1, q_len=TOKENS, num_q_heads=16, num_key_heads=16, num_value_heads=hv, dk=128, dv=128,
                dtype='bf16', layout='TND', chunk=CHUNK, state_layout='B,Hv,Dv,Dk', state_update='out_of_place')
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator='gdn_chunk', specification=spec, compiler=dict(path=str(compiler), sha256=compiler_sha),
                  triton_path=triton.__file__, coefficient_source_sha256=hashlib.sha256(BLOCKS.read_bytes()).hexdigest(),
                  passed=False, profiles={})
    try:
        host = inputs(torch, TOKENS, hv, *INPUTS[0], 'cpu')
        expected = tuple(x.to(device) for x in chunk_reference(torch, *host))
        data = tuple(x.to(device) for x in host)
        names = ('pipeline', 'stage0', 'ascendc')
        calls = {name: prepared(runtime, data, name) for name in names}
        report['performance_precheck'] = {name: assert_accuracy(torch, call(), expected) for name, call in calls.items()}
        report['launch_options'] = {name: getattr(call, 'metadata', {}) for name, call in calls.items()}
        for rnd in range(ROUNDS):
            for name in names[rnd:] + names[:rnd]:
                measured = profile(calls[name], folder / f'{name}-{rnd}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measured)
                assert measured['occupancy_consistent'], f'NPU occupancy changed during {name} round {rnd}'
        medians = {name: statistics.median(t for run in runs for t in run['samples_us'])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        report['passed'] = True
    except pytest.skip.Exception as exc:
        report['baseline_unavailable'] = str(exc)
        raise
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
