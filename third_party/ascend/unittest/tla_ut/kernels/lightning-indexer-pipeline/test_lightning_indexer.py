"""IR, score accuracy, index contract and end-to-end Lightning Indexer timing."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('lightning_indexer.py')
SIGNATURE = dict(Q='*bf16', K='*bf16', W='*bf16', O='*fp32', tile_count='i32')
HEADS, DIM, TOPK = 32, 128, 2048
SQ, SK = 1024, 32768
CASES = ((64, 128, 'short', 1103), (64, TOPK, 'boundary', 1104),
         (64, 4096, 'ties', 1103), (SQ, SK, 'normal', 1104))
TUNED = dict(shared_stage=dict(reduce_with_dot=True), four_chains=dict(head_chains=4),
             shared_stage_four_chains=dict(reduce_with_dot=True, head_chains=4),
             fixpipe_relu=dict(reduce_with_dot=True, head_chains=4, fused_relu=True))
SCHEDULES = dict(stage0=dict(prefetch=False), pipeline={}, **TUNED)


def constants(prefetch=True, reduce_with_dot=False, head_chains=1, fused_relu=False):
    dot_stage = int(prefetch)
    reduce_stage = dot_stage if reduce_with_dot else 2 * dot_stage
    return dict(B=1, SQ=SQ, SK=SK, H=HEADS, BQ=4, CORES=28, DOT_STAGE=dot_stage, REDUCE_STAGE=reduce_stage,
                HEAD_CHAINS=head_chains, FUSED_RELU=fused_relu)


@pytest.mark.parametrize('schedule', tuple(SCHEDULES))
def test_ir(schedule, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    if output_dir is not None:
        output_dir = output_dir / f'indexer-{schedule}'
    module, context = compile_offline(kernel.index_scores, SIGNATURE, constants(**SCHEDULES[schedule]), output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered and 'hivm.hir.mmadL1' in lowered
    for task in ('load_key', 'dot', 'reduce_heads'):
        assert f'hivm.task_name = "{task}"' in lowered


@pytest.mark.parametrize('schedule', tuple(SCHEDULES))
def test_compile(schedule):
    from utils.tla_test_utils import check_kernel_binary, get_stages

    if not get_stages()[1]:
        pytest.skip('Set TLA_TEST_STAGE=compile with the Ascend device runtime')
    import triton
    from triton.language.extra.tla.jit import TLASTSource
    from triton.runtime import driver

    kernel = load_kernel(OPERATOR)
    source = TLASTSource(kernel.index_scores, SIGNATURE, constexprs=constants(**SCHEDULES[schedule]))
    binary = triton.compile(source, target=driver.active.get_current_target(), options=dict(
        sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False))
    check_kernel_binary(binary)


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('Set TLA_TEST_STAGE=run with Ascend hardware')
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


@pytest.fixture
def native_operator(runtime):
    _, torch_npu, _ = runtime
    baseline = os.environ.get('TLA_INDEXER_NATIVE', 'ascendc')
    if baseline == 'ascendc_v2':
        adapter = load_kernel(OPERATOR.with_name('ascendc_v2.py'))
        return baseline, lambda inputs: adapter.prepare_call(*inputs, topk=TOPK)
    if baseline != 'ascendc':
        raise ValueError('TLA_INDEXER_NATIVE must be ascendc or ascendc_v2')
    if not hasattr(torch_npu, 'npu_lightning_indexer'):
        pytest.skip('Native comparison requires the Ascend C Lightning Indexer operator')
    return baseline, lambda inputs: native_call(torch_npu.npu_lightning_indexer, inputs)


def make_inputs(torch, sq, sk, family, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    device = torch.device('npu', torch.npu.current_device())
    q = torch.randn((1, sq, HEADS, DIM), dtype=torch.bfloat16, generator=generator)
    k = torch.randn((1, sk, 1, DIM), dtype=torch.bfloat16, generator=generator)
    w = torch.randn((1, sq, HEADS), dtype=torch.bfloat16, generator=generator)
    if family == 'ties':
        q.zero_()
    return q.to(device), k.to(device), w.to(device)


def reference(torch, q, k, w):
    b, sq, h, d = q.shape
    sk = k.shape[1]
    keys = k[:, :, 0].float().transpose(1, 2)
    key_indices = torch.arange(sk, device=q.device)
    scores = torch.empty((b, sq, sk), device=q.device, dtype=torch.float32)
    for start in range(0, sq, 8):
        stop = min(start + 8, sq)
        queries = q[:, start:stop].float().reshape(b, (stop - start) * h, d)
        dot = (queries @ keys).view(b, stop - start, h, sk).relu_()
        weighted = (dot * w[:, start:stop].float().unsqueeze(-1)).sum(dim=2)
        visible = sk - sq + 1 + torch.arange(start, stop, device=q.device)
        scores[:, start:stop] = weighted.masked_fill(key_indices[None, :] >= visible[:, None], -float('inf'))
    return scores


def native_call(operator, inputs):
    q, k, w = inputs

    def native():
        result = operator(query=q, key=k, weights=w, layout_query='BSND', layout_key='BSND',
                          sparse_mode=3, sparse_count=TOPK)
        return result[0] if isinstance(result, tuple) else result

    return native


def candidate_calls(kernel, inputs):
    return dict(pipeline=kernel.prepare_call(*inputs, topk=TOPK),
                stage0=kernel.prepare_call(*inputs, topk=TOPK, prefetch=False))


def reference_cutoff(scores):
    if scores.shape[-1] < TOPK:
        return None
    # Use the CPU selection path so a device TopK issue cannot affect both
    # the candidate and the reference threshold.
    return scores.cpu().topk(TOPK, dim=-1, sorted=False).values.amin(dim=-1).to(scores.device)


def check_selection(torch, scores, cutoff, result, implementation):
    b, sq, sk = scores.shape
    assert result.dtype == torch.int32 and result.shape == (b, sq, 1, TOPK), implementation
    indices = result.squeeze(2).long()
    visible = (sk - sq + 1 + torch.arange(sq, device=scores.device)).clamp_min(0)
    rank = torch.arange(TOPK, device=scores.device)
    valid = rank[None, :] < visible[:, None]
    assert torch.all(torch.where(valid, (indices >= 0) & (indices < visible[:, None]), indices == -1)), implementation
    short = visible < TOPK
    ordered_short = torch.where(valid, rank[None, :], -1)
    assert torch.all((indices == ordered_short) | ~short[:, None]), implementation
    ordered = indices.sort(dim=-1).values
    assert not torch.any((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)), implementation

    if cutoff is None:
        return dict(max_topk_cutoff_gap=0.0)
    # Arch35's long-row selection does not promise presentation order.
    # Any unique key within tolerance of the reference cutoff is accepted.
    chosen = scores.gather(-1, indices.clamp_min(0)).amin(dim=-1)
    chosen = torch.where(short, 0.0, chosen)
    cutoff = torch.where(short, 0.0, cutoff)
    gap = (cutoff - chosen).clamp_min(0.0)
    assert torch.all(gap <= 0.002 + cutoff.abs() * 0.001), (implementation, gap.amax().item())
    return dict(max_topk_cutoff_gap=gap.amax().item())


def check_scores(torch, actual, expected):
    invalid = torch.isneginf(expected)
    assert torch.equal(torch.isneginf(actual), invalid)
    torch.testing.assert_close(actual.masked_fill(invalid, 0.0), expected.masked_fill(invalid, 0.0),
                               atol=0.002, rtol=0.001)


@pytest.mark.parametrize('sq,sk,family,seed', CASES)
def test_run(runtime, sq, sk, family, seed):
    torch, _, kernel = runtime
    inputs = make_inputs(torch, sq, sk, family, seed)
    expected = reference(torch, *inputs)
    cutoff = reference_cutoff(expected)
    for prefetch in (False, True):
        actual = kernel.prepare_scores(*inputs, prefetch=prefetch)()
        check_scores(torch, actual, expected)
    for name, call in candidate_calls(kernel, inputs).items():
        for _ in range(3):
            check_selection(torch, expected, cutoff, call(), name)


@pytest.mark.parametrize('sq,sk,family,seed', CASES)
def test_run_native(runtime, native_operator, sq, sk, family, seed):
    torch, _, _ = runtime
    baseline, prepare = native_operator
    inputs = make_inputs(torch, sq, sk, family, seed)
    expected = reference(torch, *inputs)
    cutoff = reference_cutoff(expected)
    call = prepare(inputs)
    for _ in range(3):
        check_selection(torch, expected, cutoff, call(), baseline)


@pytest.mark.parametrize('schedule', tuple(TUNED))
def test_run_tuned(runtime, schedule):
    if os.environ.get('TLA_INDEXER_TUNING') != '1':
        pytest.skip('Set TLA_INDEXER_TUNING=1 to validate the tuning candidates')
    torch, _, kernel = runtime
    inputs = make_inputs(torch, SQ, SK, 'normal', 1104)
    expected = reference(torch, *inputs)
    cutoff = reference_cutoff(expected)
    options = TUNED[schedule]
    check_scores(torch, kernel.prepare_scores(*inputs, **options)(), expected)
    call = kernel.prepare_call(*inputs, topk=TOPK, **options)
    for _ in range(3):
        check_selection(torch, expected, cutoff, call(), schedule)


def test_perf(request, runtime, native_operator):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Use a single pytest process for performance')

    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'

    import triton
    from triton.backends.ascend.utils import _get_npucompiler_path

    torch, _, kernel = runtime
    baseline, prepare = native_operator
    output = Path(os.environ.get('TLA_LIGHTNING_INDEXER_RESULTS', 'lightning-indexer-results'))
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='prefill-', dir=output))
    spec = dict(bs=1, num_head=HEADS, kv_head=1, head_dim=DIM, q_len=SQ, kv_len=SK, sparse_count=TOPK,
                dtype='bf16', layout='BSND', sparse_mode=3, return_value=False, baseline=baseline)
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator='lightning_indexer', specification=spec, passed=False, profiles={},
                  compiler=dict(path=str(compiler), sha256=compiler_sha256), triton_path=triton.__file__)
    try:
        inputs = make_inputs(torch, SQ, SK, 'normal', 1104)
        expected = reference(torch, *inputs)
        cutoff = reference_cutoff(expected)
        calls = candidate_calls(kernel, inputs)
        calls[baseline] = prepare(inputs)
        if os.environ.get('TLA_INDEXER_TUNING') == '1':
            for name, options in TUNED.items():
                check_scores(torch, kernel.prepare_scores(*inputs, **options)(), expected)
                calls[name] = kernel.prepare_call(*inputs, topk=TOPK, **options)
        report['launch_options'] = {name: getattr(call, 'metadata', {}) for name, call in calls.items()}
        report['performance_precheck'] = {name: check_selection(torch, expected, cutoff, call(), name)
                                          for name, call in calls.items()}
        del expected
        names = list(calls)
        for rnd in range(3):
            for name in names[rnd:] + names[:rnd]:
                measurement = profile(calls[name], folder / f'{name}-{rnd}.json', 40)
                report['profiles'].setdefault(name, []).append(measurement)
                assert measurement['occupancy_consistent'], f'NPU occupancy changed during {name} round {rnd}'
        medians = {name: statistics.median(t for run in runs for t in run['samples_us'])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_ascendc'] = 100 * medians[baseline] / medians['pipeline']
        report['tuning_pct_of_ascendc'] = {name: 100 * medians[baseline] / medians[name]
                                          for name in TUNED if name in medians}
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        save_json(folder / 'results.json', report)
        print(f'Lightning Indexer results: {folder}')
