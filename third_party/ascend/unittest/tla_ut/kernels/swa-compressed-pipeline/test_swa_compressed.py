"""HCA visibility, precision and same-contract Ascend C comparisons."""

import hashlib
from importlib.util import find_spec
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('swa_compressed.py')
ACLNN_ADAPTER = Path(__file__).with_name('aclnn_hca.py')
MAIN = (4096, 32768)
EDGE = (129, 129)
RATIO, WINDOW = 128, 128
INPUTS = (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
ROUNDS, SAMPLES = 3, 40
OUTPUT = Path(os.environ.get('TLA_HCA_RESULTS', 'hca-results'))


@pytest.mark.parametrize('shape', (MAIN, EDGE))
@pytest.mark.parametrize('stage', (0, 1))
def test_ir(shape, stage, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    sq, history = shape
    signature = dict(Q='*bf16', ORIGINAL='*bf16', COMPRESSED='*bf16', SINK='*fp32', O='*bf16')
    constants = dict(SQ=sq, HISTORY=history, CORES=min(28, sq), COMPUTE_STAGE=stage,
                     EARLY_QK=bool(stage) and os.environ.get('TLA_HCA_EARLY_QK') == '1',
                     SKIP_FULL_MASK=os.environ.get('TLA_HCA_SKIP_FULL_MASK') == '1')
    if output_dir is not None:
        output_dir = output_dir / f'q{sq}-stage{stage}'
    module, context = compile_offline(kernel.hca_attention, signature, constants, output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_kv', 'qk', 'softmax', 'publish_probability', 'pv', 'accumulate'):
        assert f'hivm.task_name = "{task}"' in lowered


def inputs(torch, shape, family, seed, device):
    sq, history = shape
    rng = torch.Generator(device='cpu').manual_seed(seed)
    q = torch.randn((sq, 64, 512), generator=rng, dtype=torch.bfloat16)
    original = torch.randn((history, 1, 512), generator=rng, dtype=torch.bfloat16)
    compact = torch.randn((history // RATIO, 1, 512), generator=rng, dtype=torch.bfloat16)
    sink = torch.linspace(-4, 6, 64, dtype=torch.float32)
    if family == 'uniform':
        q.zero_()
    elif family == 'peaked':
        q.mul_(4)
    return tuple(x.to(device) for x in (q, original, compact, sink))


def reference(torch, q, original, compact, sink):
    """CPU FP32 softmax over the union, with a final zero-valued sink."""
    assert q.device.type == 'cpu'
    history = original.shape[0]
    keys = compact[:, 0].float()
    ori = original[:, 0].float()
    offsets = torch.arange(WINDOW, device=q.device)
    blocks = torch.arange(keys.shape[0], device=q.device)
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for start in range(0, q.shape[0], 16):
        qs = q[start:start + 16].float()
        positions = history - q.shape[0] + start + torch.arange(qs.shape[0], device=q.device)
        window = (positions - WINDOW + 1).clamp_min(0)[:, None] + offsets
        kv = torch.cat((ori[window], keys.expand(qs.shape[0], -1, -1)), dim=1)
        visible = torch.cat((window <= positions[:, None], blocks < (positions[:, None] + 1) // RATIO), dim=1)
        logits = (qs @ kv.transpose(1, 2)) * q.shape[-1]**-0.5
        logits.masked_fill_(~visible[:, None, :], -torch.inf)
        weighted = torch.cat((logits, sink[None, :, None].expand(qs.shape[0], -1, -1)), dim=-1).softmax(-1)
        output[start:start + qs.shape[0]] = weighted[..., :-1] @ kv
    return output


def assert_accuracy(torch, actual, expected):
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    assert expected.device.type == 'cpu' and expected.dtype == torch.float32
    actual = actual.detach().cpu()
    max_error = max_expected = squared_error = squared_expected = 0.0
    for start in range(0, expected.shape[0], 64):
        x = actual[start:start + 64].float()
        y = expected[start:start + 64]
        assert torch.isfinite(x).all(), f'Nonfinite attention output at query block {start}'
        assert torch.isfinite(y).all(), f'Nonfinite FP32 reference at query block {start}'
        error = x - y
        max_error = max(max_error, error.abs().amax().item())
        max_expected = max(max_expected, y.abs().amax().item())
        squared_error += error.square().sum(dtype=torch.float64).item()
        squared_expected += y.square().sum(dtype=torch.float64).item()
    maximum = max_error / max(max_expected, 1e-20)
    rms = math.sqrt(squared_error / expected.numel()) / max(math.sqrt(squared_expected / expected.numel()), 1e-20)
    assert maximum <= 0.006 and rms <= 0.006, (maximum, rms)
    return dict(normalized_max_error=maximum, relative_rms=rms)


def test_cpu_visibility():
    torch = pytest.importorskip('torch')
    q = torch.zeros((129, 64, 512), dtype=torch.bfloat16)
    ori = torch.ones((129, 1, 512), dtype=torch.bfloat16)
    compact = torch.full((1, 1, 512), 3.0, dtype=torch.bfloat16)
    sink = torch.cat((torch.zeros(32), torch.full((32,), math.log(2.0))))
    actual = reference(torch, q, ori, compact, sink)
    positions = torch.arange(129)
    local = (positions + 1).clamp(max=WINDOW)
    remote = (positions + 1) // RATIO
    expected = ((local + 3 * remote)[:, None] / ((local + remote)[:, None] + sink.exp()))
    torch.testing.assert_close(actual, expected[..., None].expand_as(actual))


@pytest.mark.parametrize('shape', (MAIN, EDGE))
def test_cpu_online_softmax(shape):
    torch = pytest.importorskip('torch')
    # Tail query subsets preserve the same absolute positions as chunked prefill.
    sq, history = shape
    data = inputs(torch, (min(sq, 7), history), 'normal', 1103, 'cpu')
    expected = reference(torch, *data)
    q, ori, compact, sink = data
    out = []
    for index, query in enumerate(q.float()):
        position = history - q.shape[0] + index
        raw = ori[max(position - WINDOW + 1, 0):position + 1, 0].float()
        cmp = compact[:(position + 1) // RATIO, 0].float()
        m = sink
        l = torch.ones_like(sink)
        accumulator = torch.zeros_like(query)
        for kv in (raw, *cmp.split(WINDOW)):
            if not kv.shape[0]:
                continue
            score = (query @ kv.T) * q.shape[-1]**-0.5
            next_m = torch.maximum(m, score.amax(-1))
            alpha = (m - next_m).exp()
            p = (score - next_m[:, None]).exp()
            l = l * alpha + p.sum(-1)
            accumulator = accumulator * alpha[:, None] + p.bfloat16().float() @ kv
            m = next_m
        out.append((accumulator / l[:, None]).bfloat16())
    assert_accuracy(torch, torch.stack(out), expected)


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('HCA device tests require the run stage')
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    previous = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    try:
        yield torch, load_kernel(OPERATOR)
    finally:
        torch.npu.matmul.allow_hf32 = previous


def prepare_aclnn(torch, q, original, compact, sink, *, host_cores=None):
    adapter = load_kernel(ACLNN_ADAPTER)
    try:
        return adapter.prepare_native(torch, q, original, compact, sink, host_cores=host_cores)
    except adapter.Unavailable as error:
        pytest.skip(str(error))


def prepare_native(torch, q, original, compact, sink):
    import torch_npu

    if os.environ.get('TLA_HCA_METADATA_DIR'):
        cores = load_kernel(OPERATOR).device_cores(q.device)
        return prepare_aclnn(torch, q, original, compact, sink, host_cores=cores)

    def registered(namespace):
        return (getattr(namespace, 'npu_sparse_flash_mla', None),
                getattr(namespace, 'npu_sparse_flash_mla_metadata', None))

    native, plan = registered(torch_npu)
    backend = 'torch_npu'
    if native is None or plan is None:
        native, plan = registered(torch.ops.npu)
    if native is None or plan is None:
        try:
            from cann_ops_transformer.ops.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
        except (ImportError, OSError):
            pass
        else:
            native, plan = sparse_flash_mla, sparse_flash_mla_metadata
            backend = 'cann_ops_transformer'
    if native is None or plan is None:
        backend = 'vllm_ascend'
        native, plan = registered(torch.ops._C_ascend)
    if native is None or plan is None:
        try:
            from vllm_ascend.utils import enable_custom_op
            enable_custom_op()
        except (ImportError, OSError):
            pass
        native, plan = registered(torch.ops._C_ascend)
    if native is None or plan is None:
        return prepare_aclnn(torch, q, original, compact, sink)
    sq, history, cmp_len = q.shape[0], original.shape[0], compact.shape[0]
    spans = [torch.tensor([0, n], dtype=torch.int32, device=q.device) for n in (sq, history, cmp_len)]
    residual = history % RATIO
    common = dict(cu_seqlens_q=spans[0], cu_seqlens_ori_kv=spans[1], cu_seqlens_cmp_kv=spans[2],
                  cmp_ratio=RATIO, ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=WINDOW - 1, ori_win_right=0,
                  layout_q='TND', layout_kv='TND')
    common['cmp_residual_kv'] = torch.tensor([residual], dtype=torch.int32, device=q.device)

    def metadata():
        return plan(64, 1, 512, **common, batch_size=1, max_seqlen_q=sq, max_seqlen_ori_kv=history,
                    max_seqlen_cmp_kv=cmp_len, has_ori_kv=True, has_cmp_kv=True)

    def apply(meta):
        return native(q, ori_kv=original, cmp_kv=compact, sinks=sink, metadata=meta,
                      softmax_scale=q.shape[-1]**-0.5, return_softmax_lse=False, **common)[0]

    prepared = metadata()

    def call():
        return apply(prepared)

    call.metadata = dict(backend=backend)
    return call, lambda: apply(metadata())


@pytest.mark.parametrize('shape', (MAIN, EDGE))
@pytest.mark.parametrize('implementation', ('pipeline', 'stage0', 'ascendc'))
def test_run(runtime, shape, implementation):
    torch, kernel = runtime
    device = torch.device('npu', torch.npu.current_device())
    families = INPUTS[:1] if shape == MAIN else INPUTS
    for family, seed in families:
        host = inputs(torch, shape, family, seed, 'cpu')
        data = tuple(tensor.to(device) for tensor in host)
        if implementation == 'ascendc':
            call, _ = prepare_native(torch, *data)
        else:
            call = prepare_candidate(kernel, data, prefetch=implementation == 'pipeline')
        expected = reference(torch, *host)
        for _ in range(3):
            assert_accuracy(torch, call(), expected)


def prepare_candidate(kernel, data, *, prefetch=True):
    return kernel.prepare_call(*data, prefetch=prefetch,
                               early_qk=prefetch and os.environ.get('TLA_HCA_EARLY_QK') == '1',
                               skip_full_mask=os.environ.get('TLA_HCA_SKIP_FULL_MASK') == '1')


def wall_latency(torch, call):
    for _ in range(5):
        call()
    samples = []
    for _ in range(10):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        call()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return dict(samples_us=samples, median_us=statistics.median(samples))


def test_perf(request, runtime):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 to profile on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Run performance tests without pytest-xdist')
    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'

    import triton
    from triton.backends.ascend.utils import _get_npucompiler_path

    torch, kernel = runtime
    device = torch.device('npu', torch.npu.current_device())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='hca-', dir=OUTPUT))
    sq, history = MAIN
    spec = dict(bs=1, num_head=64, kv_head=1, head_dim=512, q_len=sq, raw_history=history,
                compressed_kv_len=history // RATIO, window=WINDOW, compression_ratio=RATIO, layout='TND', dtype='bf16',
                sink='fp32', lse=False)
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator='swa_compressed_hca', specification=spec, compiler=dict(path=str(compiler), sha256=compiler_sha),
                  triton_path=triton.__file__, device_comparison='fixed_shape_reused_native_metadata',
                  passed=False, profiles={})
    try:
        host = inputs(torch, MAIN, *INPUTS[0], 'cpu')
        data = tuple(tensor.to(device) for tensor in host)
        native, native_total = prepare_native(torch, *data)
        native_library = native.metadata.get('library')
        if native.metadata['backend'] == 'vllm_ascend' and find_spec('vllm_ascend'):
            extension = find_spec('vllm_ascend.vllm_ascend_C')
            if extension is not None:
                native_library = extension.origin
        if native_library:
            with Path(native_library).open('rb') as stream:
                report['ascendc_library_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        expected = reference(torch, *host)
        calls = dict(pipeline=prepare_candidate(kernel, data), stage0=prepare_candidate(kernel, data, prefetch=False),
                     ascendc=native)
        report['performance_precheck'] = {name: assert_accuracy(torch, call(), expected) for name, call in calls.items()}
        report['launch_options'] = {name: getattr(call, 'metadata', {}) for name, call in calls.items()}
        names = list(calls)
        for rnd in range(ROUNDS):
            for name in names[rnd:] + names[:rnd]:
                measured = profile(calls[name], folder / f'{name}-{rnd}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measured)
                assert measured['occupancy_consistent'], f'NPU occupancy changed during {name} round {rnd}'
        medians = {name: statistics.median(t for run in runs for t in run['samples_us'])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        if native_total is not None:
            report['request_wall_including_native_metadata'] = dict(pipeline=wall_latency(torch, calls['pipeline']),
                                                                    ascendc=wall_latency(torch, native_total))
        report['passed'] = True
    except pytest.skip.Exception as exc:
        report['baseline_unavailable'] = str(exc)
        raise
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
