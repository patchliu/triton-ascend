"""IR, precision and full-operator timings for KDA chunk prefill."""

import hashlib
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('kda_chunk.py')
SIGNATURE = dict(W='*bf16', U='*bf16', QG='*bf16', KG='*bf16', AQK='*bf16', G='*fp32', H0='*fp32', O='*bf16',
                 HT='*fp32')
MAIN_TOKENS = 32768
HEADS, DIM, CHUNK = 12, 128, 64
CASES = ((192, 'normal', 1103), (192, 'no_update', 1104), (192, 'strong_gate', 1103),
         (MAIN_TOKENS, 'normal', 1104))


def constants(stage):
    return dict(T=MAIN_TOKENS, H=HEADS, K=DIM, V=DIM, BT=CHUNK, SCALE=DIM**-0.5, COMPUTE_STAGE=stage)


@pytest.mark.parametrize('stage', (0, 1))
def test_ir(stage, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    if output_dir is not None:
        output_dir = output_dir / f'kda-stage-{stage}'
    module, context = compile_offline(kernel.kda_scan, SIGNATURE, constants(stage), output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_cube', 'load_vector', 'publish_state', 'project_state', 'publish_correction', 'form_delta',
                 'advance_state', 'store_output'):
        assert f'hivm.task_name = "{task}"' in lowered
    assert lowered.count('hivm.hir.mmadL1') == 4


@pytest.mark.parametrize('stage', (0, 1))
def test_compile(stage):
    from utils.tla_test_utils import check_kernel_binary, get_stages

    if not get_stages()[1]:
        pytest.skip('Set TLA_TEST_STAGE=compile with the Ascend device runtime')
    import triton
    from triton.language.extra.tla.jit import TLASTSource
    from triton.runtime import driver

    kernel = load_kernel(OPERATOR)
    source = TLASTSource(kernel.kda_scan, SIGNATURE, constexprs=constants(stage))
    binary = triton.compile(source, target=driver.active.get_current_target(),
                            options=dict(enable_auto_bind_sub_block=False, enable_preload=False))
    check_kernel_binary(binary)


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('Set TLA_TEST_STAGE=run with Ascend hardware')
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    return torch, load_kernel(OPERATOR)


@pytest.fixture
def fla_prologue():
    pytest.importorskip('fla.ops.kda.chunk_intra')


@pytest.fixture
def native_operator(runtime):
    torch, _ = runtime
    if not hasattr(torch.ops.npu, 'chunk_kda_fwd') and importlib.util.find_spec('sgl_kernel_npu') is not None:
        import sgl_kernel_npu  # noqa: F401  # registers the Ascend C operator
    if not hasattr(torch.ops.npu, 'chunk_kda_fwd'):
        pytest.skip('Native comparison requires the Ascend C torch.ops.npu.chunk_kda_fwd registration')
    return torch.ops.npu.chunk_kda_fwd


def inputs(torch, tokens, family, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    shape = (1, tokens, HEADS, DIM)
    q = torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
    v = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    gate = -torch.rand(shape, generator=generator) * (0.35 if family == 'strong_gate' else 0.05)
    beta = torch.rand(shape[:-1], generator=generator, dtype=torch.bfloat16)
    initial = torch.randn((1, HEADS, DIM, DIM), generator=generator) * 0.05
    if family == 'no_update':
        beta.zero_()
        gate.zero_()
    return q, k, v, gate, beta, initial


def recurrence(torch, q, k, v, gate, beta, initial_state):
    """Independent FP32 token recurrence, evaluated on CPU outside profiling."""
    q, k, v, gate, beta = (x.float().transpose(0, 1).contiguous() for x in (q, k, v, gate, beta))
    output = torch.empty_like(v)
    state = initial_state.clone()
    scale = q.shape[-1]**-0.5
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for t in range(len(q)):
            key = k[t].unsqueeze(-1)
            state.mul_(gate[t].exp().unsqueeze(-1))
            prediction = key.transpose(-2, -1).matmul(state).squeeze(-2)
            update = beta[t].unsqueeze(-1) * (v[t] - prediction)
            state.addcmul_(key, update.unsqueeze(-2))
            output[t] = q[t].unsqueeze(-2).matmul(state).squeeze(-2) * scale
    finally:
        torch.set_num_threads(previous_threads)
    return output.transpose(0, 1).contiguous(), state


def to_device(torch, host):
    device = torch.device('npu', torch.npu.current_device())
    return tuple(x.to(device) for x in host)


def candidate_calls(kernel, device_inputs):
    return dict(pipeline=kernel.prepare_call(*device_inputs), stage0=kernel.prepare_call(*device_inputs, compute_stage=0))


def native_call(operator, device_inputs):
    q, k, v, gate, beta, initial = device_inputs

    def native():
        result = operator(
            q, k, v, gate, beta, scale=DIM**-0.5, initial_state=initial, output_final_state=True,
            chunk_size=CHUNK, layout='BSND', safe_gate=False, use_gate_in_kernel=False,
            state_v_first=False)
        return result[0], result[1]

    return native


def precision(torch, call, expected):
    output, final_state = call()
    assert output.dtype == torch.bfloat16 and final_state.dtype == torch.float32
    metrics = {}
    for name, actual, target in zip(('output', 'final_state'), (output, final_state), expected):
        actual = actual.cpu().float()
        assert actual.shape == target.shape and torch.isfinite(actual).all()
        error = actual - target
        rms = (error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-30)).item()
        peak = (error.abs().amax() / target.abs().amax().clamp_min(1e-30)).item()
        assert rms <= 0.01 and peak <= 0.03, (name, rms, peak)
        metrics[name] = dict(relative_rms=rms, normalized_max_error=peak)
    return metrics


@pytest.mark.parametrize('stage', (0, 1))
def test_run_scan_zero_update(runtime, stage):
    torch, kernel = runtime
    # One-hot queries expose any changed state row after the first chunk.
    tokens = 3 * CHUNK
    positions = torch.arange(tokens) % DIM
    query = torch.nn.functional.one_hot(positions, num_classes=DIM).view(1, tokens, 1, DIM)
    query = query.expand(1, tokens, HEADS, DIM).contiguous().bfloat16()
    row = torch.arange(DIM).view(1, 1, DIM, 1)
    column = torch.arange(DIM).view(1, 1, 1, DIM)
    head = torch.arange(HEADS).view(1, HEADS, 1, 1)
    initial = (((3 * row + 5 * column + 7 * head) % 251 - 125).float() / 128).contiguous()
    expected = (initial[:, :, positions, :].transpose(1, 2) * DIM**-0.5).bfloat16()
    qg, state = to_device(torch, (query, initial))
    zero = torch.zeros_like(qg)
    aqk = torch.zeros((1, tokens, HEADS, CHUNK), dtype=qg.dtype, device=qg.device)
    g = torch.zeros_like(qg, dtype=torch.float32)
    output = torch.empty_like(qg)
    final = torch.empty_like(state)
    kernel.kda_scan[(HEADS * 2,)](zero, zero, qg, zero, aqk, g, state, output, final, tokens, HEADS, DIM, DIM,
                                  CHUNK, DIM**-0.5, stage, enable_auto_bind_sub_block=False, enable_preload=False)
    torch.testing.assert_close(final.cpu(), initial, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize('tokens,family,seed', CASES)
def test_run(runtime, fla_prologue, tokens, family, seed):
    torch, kernel = runtime
    host = inputs(torch, tokens, family, seed)
    expected = recurrence(torch, *host)
    for call in candidate_calls(kernel, to_device(torch, host)).values():
        for _ in range(3):
            precision(torch, call, expected)


@pytest.mark.parametrize('tokens,family,seed', CASES)
def test_run_native(runtime, native_operator, tokens, family, seed):
    torch, _ = runtime
    host = inputs(torch, tokens, family, seed)
    expected = recurrence(torch, *host)
    call = native_call(native_operator, to_device(torch, host))
    for _ in range(3):
        precision(torch, call, expected)


def test_perf(request, runtime, fla_prologue, native_operator):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Use a single pytest process for performance')

    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'

    import triton
    from triton.backends.ascend.utils import _get_npucompiler_path

    torch, kernel = runtime
    result_dir = Path(os.environ.get('TLA_KDA_CHUNK_RESULTS', 'kda-chunk-results'))
    result_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='prefill-', dir=result_dir))
    spec = dict(bs=1, num_head=HEADS, kv_head=HEADS, head_dim=(DIM, DIM), q_len=MAIN_TOKENS, kv_len=MAIN_TOKENS,
                chunk=CHUNK, qkv='bf16', beta='bf16', gate='fp32', state='fp32', layout='BSND', packed=False)
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator='kda_chunk_prefill', specification=spec, passed=False, profiles={},
                  compiler=dict(path=str(compiler), sha256=compiler_sha256), triton_path=triton.__file__,
                  native_schema=str(native_operator.default._schema))
    try:
        host = inputs(torch, MAIN_TOKENS, 'normal', 1104)
        expected = recurrence(torch, *host)
        device_inputs = to_device(torch, host)
        calls = candidate_calls(kernel, device_inputs)
        calls['ascendc'] = native_call(native_operator, device_inputs)
        report['launch_options'] = {name: getattr(call, 'metadata', {}) for name, call in calls.items()}
        report['performance_precheck'] = {name: precision(torch, call, expected) for name, call in calls.items()}
        names = list(calls)
        for rnd in range(3):
            for name in names[rnd:] + names[:rnd]:
                measurement = profile(calls[name], folder / f'{name}-{rnd}.json', 40)
                report['profiles'].setdefault(name, []).append(measurement)
                assert measurement['occupancy_consistent'], f'NPU occupancy changed during {name} round {rnd}'
        medians = {name: statistics.median(t for run in runs for t in run['samples_us'])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        save_json(folder / 'results.json', report)
        print(f'KDA chunk results: {folder}')
