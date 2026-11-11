"""GDN decode task contract, reference accuracy and same-work profiling."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('gdn_recurrent.py')
HEAD_COUNTS = (16, 32)
INPUTS = (('normal', 1103), ('normal', 1104), ('zero_key', 1103), ('slow_decay', 1103))
ROUNDS, SAMPLES = 3, 40
OUTPUT = Path(os.environ.get('TLA_GDN_RESULTS', 'gdn-results'))
VALUE_TILE = int(os.environ.get('TLA_GDN_VALUE_TILE', '64'))
STORE_STAGE = int(os.environ.get('TLA_GDN_STORE_STAGE', '2'))


@pytest.mark.parametrize('hv', HEAD_COUNTS)
@pytest.mark.parametrize('prefetch', (False, True))
def test_ir(hv, prefetch, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    signature = dict(Q='*bf16', K='*bf16', V='*bf16', G='*fp32', BETA='*bf16', STATE='*bf16', SLOTS='*i32', O='*bf16')
    constants = dict(B=64, HK=16, HV=hv, DK=128, DV=128, CORES=56, SCALE=128**-0.5,
                     COMPUTE_STAGE=int(prefetch), STORE_STAGE=STORE_STAGE if prefetch else 0, BV=VALUE_TILE)
    if output_dir is not None:
        output_dir = output_dir / f'hv{hv}-bv{VALUE_TILE}-stage{constants["STORE_STAGE"]}'
    module, context = compile_offline(kernel.gdn_recurrent, signature, constants, output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load', 'update', 'store'):
        assert f'hivm.task_name = "{task}"' in lowered


def inputs(torch, batch, hv, family, seed, device):
    hk, dim = 16, 128
    rng = torch.Generator(device='cpu').manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn((batch, hk, dim), generator=rng), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn((batch, hk, dim), generator=rng), dim=-1).bfloat16()
    v = torch.randn((batch, hv, dim), generator=rng).bfloat16()
    g = torch.empty((batch, hv)).uniform_(-0.7, -0.05, generator=rng)
    beta = torch.empty((batch, hv)).uniform_(0.1, 0.9, generator=rng).bfloat16()
    state = (torch.randn((batch + 3, hv, dim, dim), generator=rng) * 0.2).bfloat16()
    slots = torch.randperm(batch + 3, generator=rng, dtype=torch.int32)[:batch].contiguous()
    if family == 'zero_key':
        k.zero_()
    elif family == 'slow_decay':
        g.uniform_(-0.003, 0, generator=rng)
        beta.fill_(0.9)
    return tuple(x.to(device) for x in (q, k, v, g, beta, state, slots))


def reference(torch, q, k, v, g, beta, state, slots):
    """Independent per-token FP32 recurrence. Keep unused slots intact."""
    group = v.shape[1] // q.shape[1]
    q = q.float().repeat_interleave(group, dim=1) * q.shape[-1]**-0.5
    k = k.float().repeat_interleave(group, dim=1)
    indices = slots.long()
    decayed = state.index_select(0, indices).float() * g.exp()[..., None, None]
    error = beta.float()[..., None] * (v.float() - (decayed @ k.unsqueeze(-1)).squeeze(-1))
    updated = decayed + error.unsqueeze(-1) * k.unsqueeze(-2)
    result = (updated @ q.unsqueeze(-1)).squeeze(-1)
    final_state = state.clone().index_copy_(0, indices, updated.to(state.dtype))
    return result, final_state


def assert_accuracy(torch, actual, expected):
    errors = {}
    for name, x, y in zip(('output', 'state'), actual, expected):
        assert x.shape == y.shape and x.dtype == torch.bfloat16
        x, y = x.float(), y.float()
        assert torch.isfinite(x).all()
        delta = x - y
        maximum = (delta.abs().amax() / y.abs().amax().clamp_min(1e-20)).item()
        rms = (delta.square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-20)).item()
        assert maximum <= 0.006 and rms <= 0.006, (name, maximum, rms)
        errors[name] = dict(normalized_max_error=maximum, relative_rms=rms)
    return errors


def test_cpu_reference():
    torch = pytest.importorskip('torch')
    data = inputs(torch, 2, 32, 'zero_key', 1103, 'cpu')
    q, _, _, g, _, state, slots = data
    out, final = reference(torch, *data)
    initial = state[slots.long()].float()
    decayed = initial * g.exp()[..., None, None]
    torch.testing.assert_close(final[slots.long()], decayed.bfloat16())
    unused = torch.ones(state.shape[0], dtype=torch.bool)
    unused[slots.long()] = False
    assert torch.equal(final[unused], state[unused])
    q_grouped = q.float().repeat_interleave(2, dim=1)
    torch.testing.assert_close(out, (decayed * q_grouped.unsqueeze(-2)).sum(-1) * q.shape[-1]**-0.5,
                               rtol=1e-4, atol=2e-7)


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('GDN recurrent device tests require the run stage')
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
    q, k, v, g, beta, initial, slots = data
    state = initial.clone()
    if implementation == 'ascendc':
        native = getattr(torch_npu, 'npu_recurrent_gated_delta_rule', None)
        if native is None:
            pytest.skip('torch_npu does not provide npu_recurrent_gated_delta_rule')
        lengths = torch.ones(q.shape[0], dtype=torch.int32, device=q.device)

        def call():
            out = native(q, k, v, state, beta=beta, scale=q.shape[-1]**-0.5,
                         actual_seq_lengths=lengths, ssm_state_indices=slots, g=g)
            return out, state
    else:
        prefetch = implementation == 'pipeline'
        call = kernel.prepare_call(q, k, v, g, beta, state, slots, prefetch=prefetch, value_tile=VALUE_TILE,
                                   store_stage=STORE_STAGE if prefetch else 0)
    return call, state


@pytest.mark.parametrize('hv', HEAD_COUNTS)
@pytest.mark.parametrize('implementation', ('pipeline', 'stage0', 'ascendc'))
def test_run(runtime, hv, implementation):
    torch, _, _ = runtime
    device = torch.device('npu', torch.npu.current_device())
    for family, seed in INPUTS:
        data = inputs(torch, 64, hv, family, seed, device)
        expected = reference(torch, *data)
        call, state = prepared(runtime, data, implementation)
        unused = torch.ones(state.shape[0], dtype=torch.bool, device=device)
        unused[data[-1].long()] = False
        for _ in range(3):
            state.copy_(data[-2])
            actual = call()
            assert_accuracy(torch, actual, expected)
            assert torch.equal(state[unused], data[-2][unused])


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
    folder = Path(tempfile.mkdtemp(prefix=f'recurrent-hv{hv}-bv{VALUE_TILE}-store{STORE_STAGE}-', dir=OUTPUT))
    spec = dict(bs=64, q_len=1, num_q_heads=16, num_key_heads=16, num_value_heads=hv, dk=128, dv=128,
                dtype='bf16', layout='TND', state_layout='S,Hv,Dv,Dk', state_update='inplace')
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator='gdn_recurrent', specification=spec, compiler=dict(path=str(compiler), sha256=compiler_sha),
                  triton_path=triton.__file__, passed=False, profiles={})
    try:
        data = inputs(torch, 64, hv, *INPUTS[0], device)
        expected = reference(torch, *data)
        names = ('pipeline', 'stage0', 'ascendc')
        calls = {name: prepared(runtime, data, name) for name in names}
        report['performance_precheck'] = {name: assert_accuracy(torch, call(), expected)
                                           for name, (call, _) in calls.items()}
        report['launch_options'] = {name: getattr(call, 'metadata', {}) for name, (call, _) in calls.items()}
        for rnd in range(ROUNDS):
            for name in names[rnd:] + names[:rnd]:
                call, state = calls[name]
                state.copy_(data[-2])
                torch.npu.synchronize()
                measured = profile(call, folder / f'{name}-{rnd}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measured)
                assert measured['occupancy_consistent'], f'NPU occupancy changed during {name} round {rnd}'
        medians = {name: statistics.median(t for run in runs for t in run['samples_us'])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
