"""MiniMax-M3 block-sparse attention tests with explicit sparse-pattern parity."""

import hashlib
from itertools import accumulate
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('minimax_sparse.py')
CASES = {'prefill': (4096, (32768,)), 'short-prefix': (96, (131, 521))}
HQ, HKV, D, PAGE, TOPK = 64, 4, 128, 128, 16
SCALE = D**-0.5
INPUTS = (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
ROUNDS, SAMPLES = 3, 40
RESULTS = Path(os.environ.get('TLA_MINIMAX_SPARSE_RESULTS', 'minimax-sparse-results'))
RESOLVE_METADATA = os.environ.get('TLA_MINIMAX_RESOLVE_METADATA') == '1'


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('MiniMax sparse attention requires the run stage')
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


@pytest.fixture
def native_backend():
    from importlib import import_module

    mode = os.environ.get('TLA_MINIMAX_NATIVE', 'gbsa')
    if mode == 'splitkv':
        ops = pytest.importorskip('custom_ops')
        if not all(callable(getattr(ops, name, None)) for name in
                   ('build_k2q_csr', 'npu_minimax_sparse_attention_split_kv')):
            pytest.skip('Installed custom_ops extension lacks MiniMax split-KV or the reference CSR builder')
        try:
            csr = import_module('minimax_k2q_csr_native').k2q_csr
        except ModuleNotFoundError as error:
            if error.name != 'minimax_k2q_csr_native':
                raise
            csr = None
        return mode, ops, csr
    if mode != 'gbsa':
        raise ValueError('TLA_MINIMAX_NATIVE must be gbsa or splitkv')
    names = ('generic_block_sparse_attention', 'generic_block_sparse_attention_metadata')
    for module in ('cann_ops_transformer.ops.attention.generic_block_sparse_attention', 'cann_ops_transformer.ops'):
        try:
            ops = import_module(module)
        except ImportError:
            continue
        if all(callable(getattr(ops, name, None)) for name in names):
            return mode, ops, None
    pytest.skip('Installed CANN extension lacks GBSA or its metadata operator')


@pytest.mark.parametrize('case', tuple(CASES))
@pytest.mark.parametrize('compute_stage', (0, 1))
def test_ir(case, compute_stage, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    sq, lengths = CASES[case]
    b = len(lengths)
    kernel = load_kernel(OPERATOR)
    signature = dict(Q='*bf16', K='*bf16', V='*bf16', TABLE='*i32', LENGTHS='*i32', INDICES='*i32',
                     COUNTS='*i32', OUT='*bf16', PHYSICAL='*i32', VISIBLE='*i32')
    constants = dict(B=b, SQ=sq, HQ=HQ, HKV=HKV, TABLE_STRIDE=(max(lengths) + PAGE - 1) // PAGE,
                     TOPK=TOPK, CORES=min(28, b * sq * HKV), COMPUTE_STAGE=compute_stage,
                     OVERLAP=bool(compute_stage) and os.environ.get('TLA_MINIMAX_OVERLAP') == '1',
                     SKIP_FULL_MASK=bool(compute_stage) and os.environ.get('TLA_MINIMAX_SKIP_FULL_MASK') == '1',
                     RESOLVED_METADATA=RESOLVE_METADATA)
    if output_dir is not None:
        output_dir = output_dir / f'{case}-stage-{compute_stage}'
    module, context = compile_offline(kernel.sparse_attention, signature, constants, output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_block', 'qk', 'softmax', 'publish_probability', 'pv', 'accumulate'):
        assert f'hivm.task_name = "{task}"' in lowered


def make_inputs(torch, sq, lengths, family, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    b = len(lengths)
    per_request = [(length + PAGE - 1) // PAGE for length in lengths]
    max_pages = max(per_request)
    physical = torch.randperm(sum(per_request), dtype=torch.int32, generator=generator)
    table = torch.full((b, max_pages), -1, dtype=torch.int32)
    for request, pages in enumerate(physical.split(per_request)):
        table[request, :pages.numel()] = pages

    length_cpu = torch.tensor(lengths, dtype=torch.int32)
    local = ((length_cpu[:, None] - sq + torch.arange(sq)[None, :]) // PAGE).reshape(-1)
    scores = torch.rand((HKV, b * sq, max_pages), generator=generator)
    scores.masked_fill_(torch.arange(max_pages)[None, None, :] > local[None, :, None], -float('inf'))
    # Match the M3 contract of forcing the query's local block into top-k.
    scores.scatter_(-1, local[None, :, None].expand(HKV, -1, -1), float('inf'))
    capacity = min(TOPK, max_pages)
    selected = scores.topk(capacity, dim=-1).indices
    valid = selected <= local[None, :, None]
    indices = torch.full((HKV, b * sq, TOPK), -1, dtype=torch.int32)
    sorted_indices = selected.masked_fill(~valid, max_pages).sort(dim=-1).values
    indices[..., :capacity] = sorted_indices.masked_fill(sorted_indices == max_pages, -1).to(torch.int32)
    counts = valid.sum(dim=-1).to(torch.int32)

    device = torch.device('npu', torch.npu.current_device())

    def randn(shape, *, query=False):
        if query and family == 'uniform':
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        tensor = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
        if query and family == 'peaked':
            tensor.mul_(8)
        return tensor.to(device)

    q = randn((b * sq, HQ, D), query=True)
    k = randn((physical.numel(), PAGE, HKV, D))
    v = randn(k.shape)
    return q, k, v, table.to(device), length_cpu.to(device), indices.to(device), counts.to(device)


def reference(torch, inputs, sq, lengths):
    q, k, v, table, _, indices, counts = inputs
    result = torch.empty_like(q, dtype=torch.float32)
    key_lane = torch.arange(PAGE, device=q.device)
    rank = torch.arange(TOPK, device=q.device)
    for batch, length in enumerate(lengths):
        for kv_head in range(HKV):
            group = slice(kv_head * (HQ // HKV), (kv_head + 1) * (HQ // HKV))
            for start in range(0, sq, 32):
                stop = min(start + 32, sq)
                query = slice(batch * sq + start, batch * sq + stop)
                logical = indices[kv_head, query].clamp_min(0).to(torch.int64)
                pages = table[batch].index_select(0, logical.flatten()).to(torch.int64)
                keys = k.index_select(0, pages)[:, :, kv_head].reshape(stop - start, TOPK * PAGE, D).float()
                values = v.index_select(0, pages)[:, :, kv_head].reshape(stop - start, TOPK * PAGE, D).float()
                causal_end = length - sq + torch.arange(start, stop, device=q.device)
                keep = logical[:, :, None] * PAGE + key_lane[None, None, :] <= causal_end[:, None, None]
                keep &= rank[None, :, None] < counts[kv_head, query, None, None]
                score = (q[query, group].float() @ keys.transpose(-1, -2)) * SCALE
                score.masked_fill_(~keep.flatten(1)[:, None, :], -float('inf'))
                result[query, group] = score.softmax(dim=-1) @ values
    return result


def native_call(torch, gbsa, inputs, sq, batches):
    q, k, v, table, lengths, indices, counts = inputs
    cu_q = torch.arange(batches + 1, dtype=torch.int64, device=q.device) * sq
    layout = dict(layout_q='TND', layout_kv='PA_BBND', mask_mode=1, quant_mode=0, softmax_precision=0)
    shape = [1, PAGE]

    def call():
        metadata = gbsa.generic_block_sparse_attention_metadata(
            indices, counts, HQ, HKV, D, shape, cu_seqlens_q=cu_q, seqused_kv=lengths,
            max_seqlen_q=sq, **layout)
        out, _ = gbsa.generic_block_sparse_attention(
            q, k, v, indices, counts, shape, metadata=metadata, cu_seqlens_q=cu_q,
            seqused_kv=lengths, block_table=table, softmax_scale=SCALE, return_softmax_lse=False, **layout)
        return out

    call.metadata = dict(api='cann_ops_transformer.ops.generic_block_sparse_attention',
                         includes_metadata=True, **layout)
    return call


def splitkv_call(torch, ops, csr_device, inputs, sq, kv_lengths):
    q, k, v, table, lengths, indices, counts = inputs
    batches = len(kv_lengths)
    q_lengths = torch.full((batches,), sq, dtype=torch.int32, device=q.device)
    topk = indices.shape[-1]
    if csr_device is None:
        # A precision-only diagnostic while the official device CSR is unavailable.
        csr = tuple(t.to(q.device) for t in ops.build_k2q_csr(
            indices.cpu(), (sq,) * batches, kv_lengths, PAGE, select_num_idx=counts.cpu(), input_layout='TND'))
    else:
        blocks = [(length + PAGE - 1) // PAGE for length in kv_lengths]
        cu_q = torch.arange(batches + 1, dtype=torch.int32, device=q.device) * sq
        cu_blocks = torch.tensor((0, *accumulate(blocks)), dtype=torch.int32, device=q.device)

    def call():
        if csr_device is None:
            row_ptr, q_idx, slots = csr
        else:
            # Test inputs already mark every unused slot -1. Use interleaved KV rows and global Q ids.
            row_ptr, q_idx, slots = csr_device(indices, cu_q, cu_blocks, 1, sum(blocks), max(blocks), 1, 1)
        out, _ = ops.npu_minimax_sparse_attention_split_kv(
            q, k, v, row_ptr, q_idx, slots, q_lengths, lengths, HKV, SCALE, PAGE, topk,
            block_table=table, inner_precise=0, softmax_lse_flag=False, input_layout='TND')
        return out

    call.metadata = dict(api='aclnnMinimaxSparseAttentionSplitKv', inner_precise=0,
                         csr='device_simt' if csr_device is not None else 'host_precached',
                         includes_csr=csr_device is not None)
    return call


def prepare(runtime, case, family, seed, native=None):
    torch, kernel = runtime
    sq, lengths = CASES[case]
    inputs = make_inputs(torch, sq, lengths, family, seed)
    calls = dict(pipeline=kernel.prepare_call(*inputs, overlap=os.environ.get('TLA_MINIMAX_OVERLAP') == '1',
                                             skip_full_mask=os.environ.get('TLA_MINIMAX_SKIP_FULL_MASK') == '1',
                                             resolve_metadata=RESOLVE_METADATA),
                 stage0=kernel.prepare_call(*inputs, prefetch=False, resolve_metadata=RESOLVE_METADATA))
    if native is not None:
        mode, ops, csr_device = native
        calls['ascendc'] = (splitkv_call(torch, ops, csr_device, inputs, sq, lengths) if mode == 'splitkv'
                            else native_call(torch, ops, inputs, sq, len(lengths)))
    return calls, reference(torch, inputs, sq, lengths)


def precision(torch, call, expected):
    actual = call()
    torch.npu.synchronize()
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    error = actual.float() - expected
    maximum = (error.abs().amax() / expected.abs().amax().clamp_min(1e-30)).item()
    rms = (error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)).item()
    assert maximum <= 0.008 and rms <= 0.005, (maximum, rms)
    return dict(normalized_max_error=maximum, relative_rms=rms)


def accuracy(runtime, case, native=None):
    torch, _ = runtime
    records = []
    for family, seed in INPUTS:
        calls, expected = prepare(runtime, case, family, seed, native)
        checks = {name: [precision(torch, call, expected) for _ in range(3)] for name, call in calls.items()}
        records.append(dict(family=family, seed=seed, checks=checks))
        del calls, expected
    return records


@pytest.mark.parametrize('case', tuple(CASES))
def test_run(runtime, case):
    accuracy(runtime, case)


@pytest.mark.parametrize('case', tuple(CASES))
def test_native(runtime, native_backend, case):
    accuracy(runtime, case, native_backend)


def test_perf(request, runtime, native_backend):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 to profile on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Run performance tests without pytest-xdist')
    if native_backend[0] == 'splitkv' and native_backend[2] is None:
        pytest.skip('Split-KV performance requires the official device CSR converter')
    import triton
    from triton.backends.ascend.utils import _get_npucompiler_path

    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'
    torch, kernel = runtime
    sq, lengths = CASES['prefill']
    spec = dict(bs=len(lengths), num_head=HQ, kv_head=HKV, head_dim=D, q_len=sq, kv_len=lengths[0],
                page=PAGE, topk=TOPK, dtype='bf16', layout='TND/PA_BBND', scale=SCALE,
                mask='right_aligned_causal')
    RESULTS.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='prefill-', dir=RESULTS))
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator='minimax_sparse_prefill', specification=spec, passed=False, profiles={},
                  compiler=dict(path=str(compiler), sha256=compiler_sha), triton_path=triton.__file__)
    try:
        report['accuracy'] = accuracy(runtime, 'prefill', native_backend)
        report['short_prefix_accuracy'] = accuracy(runtime, 'short-prefix', native_backend)
        calls, expected = prepare(runtime, 'prefill', *INPUTS[0], native_backend)
        report['launch_options'] = {name: call.metadata for name, call in calls.items()}
        report['performance_precheck'] = {name: precision(torch, call, expected) for name, call in calls.items()}
        names = list(calls)
        for round_id in range(ROUNDS):
            for name in names[round_id:] + names[:round_id]:
                measured = profile(calls[name], folder / f'{name}-{round_id}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measured)
                assert measured['occupancy_consistent'], f'NPU occupancy changed during {name} round {round_id}'
        medians = {name: statistics.median([t for run in runs for t in run['samples_us']])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        save_json(folder / 'results.json', report)
