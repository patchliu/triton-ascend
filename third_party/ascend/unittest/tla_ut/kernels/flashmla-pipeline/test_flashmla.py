"""FlashMLA V4.1 decode from identical byte-packed inputs on both backends."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('flashmla.py')
BATCH, HEADS, DIM = 64, 64, 512
MAIN_TOP, EXTRA_TOP = 256, 2048
MAIN_ROWS, EXTRA_ROWS = 256, 4096
PAGE = 64
FAMILIES = (('normal', 1103), ('normal', 1104), ('peaked', 1103), ('invalid_prefix', 1103))
ROUNDS, SAMPLES = 3, 40


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('FlashMLA device checks require the run stage')
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


@pytest.mark.parametrize('stage', (0, 1))
@pytest.mark.parametrize('fused', (False, True))
def test_ir(stage, fused, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    signature = dict(Q='*bf16', KV='*bf16', VALID='*i32', MAIN='*u8', EXTRA='*u8', MAIN_IDS='*i32',
                     EXTRA_IDS='*i32', SINK='*fp32', O='*bf16', LSE='*fp32')
    constants = dict(B=BATCH, H=HEADS, TOP=MAIN_TOP + EXTRA_TOP, CORES=28, HAS_SINK=True, COMPUTE_STAGE=stage,
                     EARLY_QK=False, FUSED_GATHER=fused, MAIN_TOP=MAIN_TOP, MAIN_TOKENS=BATCH * MAIN_ROWS,
                     EXTRA_TOKENS=BATCH * EXTRA_ROWS)
    if output_dir is not None:
        output_dir = output_dir / f'fused-{int(fused)}-stage-{stage}'
    module, ctx = compile_offline(kernel._attention, signature, constants, output_dir)
    lowered = str(lower_offline(module, ctx, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('gather_kv' if fused else 'load_kv', 'qk', 'softmax', 'publish_probability', 'pv', 'accumulate'):
        assert f'hivm.task_name = "{task}"' in lowered


@pytest.mark.parametrize('tile,grouped', ((8, False), (16, True), (32, True)))
def test_compile_gather(tile, grouped):
    import triton
    from triton.compiler import ASTSource
    from utils.tla_test_utils import check_kernel_binary, get_stages

    if not get_stages()[1]:
        pytest.skip('FP8 gather requires a target binary compile stage')
    signature = dict(MAIN='*u8', EXTRA='*u8', MAIN_IDS='*i32', EXTRA_IDS='*i32', KV='*bf16', VALID='*i32')
    constants = dict(B=BATCH, MAIN_TOP=MAIN_TOP, EXTRA_TOP=EXTRA_TOP, MAIN_TOKENS=BATCH * MAIN_ROWS,
                     EXTRA_TOKENS=BATCH * EXTRA_ROWS, TILE=tile, GROUPED_GATHER=grouped)
    gather = load_kernel(OPERATOR)._gather_fp8
    check_kernel_binary(triton.compile(ASTSource(gather, signature, constexprs=constants)))


def packed_cache(torch, rows_per_request, generator):
    rows = BATCH * rows_per_request
    raw = torch.empty((rows, 528), dtype=torch.uint8)
    # Build actual byte records on CPU without relying on the NPU's FP8 casts.
    for chunk in raw.split(4096):
        values = torch.randn((chunk.shape[0], DIM), generator=generator) * 80
        chunk[:, :DIM] = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        chunk[:, DIM:] = torch.randint(119, 123, (chunk.shape[0], 16), generator=generator, dtype=torch.uint8)
    return raw.view(rows // PAGE, PAGE, 1, 528)


def prepared(runtime, family, seed, native=None):
    torch, kernel = runtime
    device = torch.device('npu', torch.npu.current_device())
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((BATCH, 1, HEADS, DIM), generator=generator, dtype=torch.bfloat16)
    sink = torch.randn((HEADS,), generator=generator, dtype=torch.float32)
    if family == 'peaked':
        q.mul_(4)
    main = packed_cache(torch, MAIN_ROWS, generator)
    extra = packed_cache(torch, EXTRA_ROWS, generator)
    # Each request owns distinct cache pages. Shuffled selections exercise
    # random token gathers without creating cross-request L2 reuse.
    main_ids = torch.stack([torch.randperm(MAIN_ROWS, generator=generator, dtype=torch.int32) + b * MAIN_ROWS
                            for b in range(BATCH)]).unsqueeze(1)
    extra_ids = torch.stack([torch.randperm(EXTRA_ROWS, generator=generator, dtype=torch.int32)[:EXTRA_TOP]
                             + b * EXTRA_ROWS for b in range(BATCH)]).unsqueeze(1)
    if family == 'invalid_prefix':
        main_ids[:, :, :128] = -1
        extra_ids[:, :, :128] = -1
        main_ids[0] = -1
        extra_ids[0] = -1
    caches = (main, extra)
    indices = (main_ids, extra_ids)
    q, sink = q.to(device), sink.to(device)
    args = (q, main.to(device), main_ids.to(device), extra.to(device), extra_ids.to(device), sink)
    fused_gather = os.environ.get('TLA_FLASHMLA_FUSED_GATHER')
    tiling = dict(gather_tile=int(os.environ.get('TLA_FLASHMLA_GATHER_TILE', '8')),
                  grouped_gather=os.environ.get('TLA_FLASHMLA_GROUPED_GATHER') == '1',
                  fused_gather=None if fused_gather is None else fused_gather == '1')
    calls = dict(pipeline=kernel.prepare_call(*args, **tiling, early_qk=os.environ.get('TLA_FLASHMLA_EARLY_QK') == '1'),
                 stage0=kernel.prepare_call(*args, **tiling, prefetch=False))
    if native is not None:
        schedule, _ = native.get_mla_metadata()

        def ascendc():
            q_npu, main_npu, ids_npu, extra_npu, extra_ids_npu, sink_npu = args
            return native.flash_mla_with_kvcache(
                q_npu, main_npu, None, None, DIM, schedule, softmax_scale=DIM**-0.5,
                indices=ids_npu, attn_sink=sink_npu, extra_k_cache=extra_npu,
                extra_indices_in_kvcache=extra_ids_npu,
            )

        calls['ascendc'] = ascendc
    return calls, reference(torch, q, sink, caches, indices)


def reference(torch, q, sink, caches, indices):
    def gather(cache, ids):
        selected = cache.view(-1, 528).index_select(0, ids.clamp_min(0).long())
        mantissa = selected[:, :DIM].contiguous().view(torch.float8_e4m3fn).float()
        scale = torch.exp2(selected[:, DIM:].float() - 127)
        return (mantissa.view(-1, 16, 32) * scale.unsqueeze(-1)).view(-1, DIM)

    outputs, lses = [], []
    for b in range(q.shape[0]):
        valid = torch.cat([ids[b, 0] >= 0 for ids in indices]).to(q.device)
        kv = torch.cat([gather(cache, ids[b, 0]) for cache, ids in zip(caches, indices)]).to(q.device)
        logits = (q[b, 0].float() @ kv.T) * DIM**-0.5
        logits = logits.masked_fill(~valid, -float('inf'))
        lse = logits.logsumexp(-1)
        empty = torch.isneginf(lse)
        safe_lse = lse.masked_fill(empty, 0.0)
        weights = (logits - safe_lse[:, None]).exp()
        out = (weights @ kv) * torch.sigmoid(safe_lse - sink)[:, None]
        outputs.append(out)
        lses.append(lse.masked_fill(empty, float('inf')))
    return torch.stack(outputs).unsqueeze(1), torch.stack(lses).unsqueeze(-1)


def precision(torch, call, expected):
    output, lse = call()
    torch.npu.synchronize()
    expected_output, expected_lse = expected
    assert output.shape == expected_output.shape and output.dtype == torch.bfloat16
    assert lse.shape == expected_lse.shape and lse.dtype == torch.float32
    assert torch.isfinite(output).all()
    empty = torch.isposinf(expected_lse)
    assert torch.equal(torch.isposinf(lse), empty)
    lse_error = (lse.masked_fill(empty, 0) - expected_lse.masked_fill(empty, 0)).abs().amax().item()
    error = output.float() - expected_output
    maximum = (error.abs().amax() / expected_output.abs().amax().clamp_min(1e-30)).item()
    rms = (error.square().mean().sqrt() / expected_output.square().mean().sqrt().clamp_min(1e-30)).item()
    assert maximum <= 0.005 and rms <= 0.005 and lse_error <= 0.005, (maximum, rms, lse_error)
    return dict(normalized_max_error=maximum, relative_rms=rms, lse_absolute_max_error=lse_error)


@pytest.mark.parametrize('family,seed', FAMILIES)
def test_run(runtime, family, seed):
    torch, _ = runtime
    calls, expected = prepared(runtime, family, seed)
    for call in calls.values():
        for _ in range(3):
            precision(torch, call, expected)


def load_ascendc():
    native = pytest.importorskip('flash_mla')
    build = pytest.importorskip('flash_mla._build_info')
    if build.BACKEND != 'npu':
        pytest.skip('Requires the Ascend C build of DeepSeek FlashMLA')
    return native, build


@pytest.mark.parametrize('family,seed', FAMILIES)
def test_run_native(runtime, family, seed):
    torch, _ = runtime
    native, _ = load_ascendc()
    calls, expected = prepared(runtime, family, seed, native)
    for _ in range(3):
        precision(torch, calls['ascendc'], expected)


def test_perf(request, runtime):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 to profile on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Run performance tests without pytest-xdist')
    native, build = load_ascendc()
    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'

    import triton
    from triton.backends.ascend.utils import _get_npucompiler_path

    torch, kernel = runtime
    output = Path(os.environ.get('TLA_FLASHMLA_RESULTS', 'flashmla-results'))
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='v41-decode-', dir=output))
    spec = dict(bs=BATCH, num_head=HEADS, kv_head=1, head_dim=DIM, q_len=1, main_topk=MAIN_TOP,
                extra_topk=EXTRA_TOP, main_cache_rows=MAIN_ROWS, extra_cache_rows=EXTRA_ROWS,
                page=PAGE, q_dtype='bf16', kv_format='v41_fp8_528B', sink=True, lse=True)
    extension = Path(native.__file__).parent / build.EXTENSION
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with extension.open('rb') as native_file, compiler.open('rb') as compiler_file:
        report = dict(specification=spec, triton_path=triton.__file__, compiler_path=str(compiler),
                      compiler_sha256=hashlib.file_digest(compiler_file, 'sha256').hexdigest(),
                      native_path=str(extension), native_sha256=hashlib.file_digest(native_file, 'sha256').hexdigest(),
                      profiles={}, passed=False)
    try:
        report['accuracy'] = []
        for family, seed in FAMILIES:
            calls, expected = prepared(runtime, family, seed, native)
            checks = {name: [precision(torch, call, expected) for _ in range(3)] for name, call in calls.items()}
            report['accuracy'].append(dict(family=family, seed=seed, checks=checks))
        calls, expected = prepared(runtime, *FAMILIES[0], native)
        report['tiling'] = {name: call.metadata for name, call in calls.items() if hasattr(call, 'metadata')}
        report['performance_precheck'] = {name: precision(torch, call, expected) for name, call in calls.items()}
        order = list(calls)
        for rnd in range(ROUNDS):
            for name in order[rnd:] + order[:rnd]:
                measured = profile(calls[name], folder / f'{name}-{rnd}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measured)
                assert measured['occupancy_consistent'], f'NPU occupancy changed in {name} round {rnd}'
        medians = {name: statistics.median([t for run in runs for t in run['samples_us']])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        # Defer optional device inventory until after the native launches.
        # A recovered memory-capability query can leave a thread-local ACL error.
        save_environment(folder, kernel, spec)
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
