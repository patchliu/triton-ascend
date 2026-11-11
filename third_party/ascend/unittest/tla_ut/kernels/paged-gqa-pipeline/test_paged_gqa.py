"""Paged GQA tests against FP32 and the CANN paged attention API."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('paged_gqa.py')
CASES = {'decode': (1, (4096,) * 16), 'prefill': (512, (8192,) * 4)}
TAIL_LENGTHS = (65, 128, 257, 511)
INPUTS = (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
ROUNDS, SAMPLES = 3, 40
OUTPUT = Path(os.environ.get('TLA_PAGED_GQA_RESULTS', 'paged-gqa-results'))


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('Paged GQA device tests require the run stage')
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


@pytest.mark.parametrize('case,schedule', [(case, mode) for case in CASES
                                          for mode in ('serial', 'prefetch', 'overlap', 'deep')]
                         + [('prefill', 'wide')])
def test_ir(case, schedule, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    signature = dict(Q='*bf16', K='*bf16', V='*bf16', PAGE_TABLE='*i32', LENGTHS='*i32', O='*bf16')
    sq, lengths = CASES[case]
    column = sq > 1 and schedule in ('overlap', 'deep', 'wide')
    q_tile = 16 if sq == 1 else {'overlap': 128, 'deep': 128, 'wide': 256}.get(schedule, 64)
    constants = dict(B=len(lengths), HQ=32, HKV=8, SQ=sq, Q_TILE=q_tile,
                     TABLE_STRIDE=(max(lengths) + 127) // 128, CORES=28, SCHEDULE=kernel._SCHEDULES[schedule],
                     COLUMN_SOFTMAX=column, PAGES_PER_TILE=1, COMPACT_DECODE=False,
                     LATE_VALUE=column and schedule == 'deep')
    if output_dir is not None:
        output_dir = output_dir / f'{case}-{schedule}'
    module, context = compile_offline(kernel.paged_gqa, signature, constants, output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_kv', 'qk', 'softmax', 'publish_probability', 'pv', 'accumulate'):
        assert f'hivm.task_name = "{task}"' in lowered


def make_inputs(torch, sq, lengths, family, seed):
    page = 128
    pages_per_request = [(length + page - 1) // page for length in lengths]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randperm(sum(pages_per_request), generator=generator, dtype=torch.int32)
    table = torch.full((len(lengths), max(pages_per_request)), -1, dtype=torch.int32)
    for row, group in enumerate(ids.split(pages_per_request)):
        table[row, :group.numel()] = group

    q = torch.randn((len(lengths), 32, sq, 128), generator=generator, dtype=torch.bfloat16)
    k = torch.randn((ids.numel(), 8, page, 128), generator=generator, dtype=torch.bfloat16)
    v = torch.randn(k.shape, generator=generator, dtype=torch.bfloat16)
    if family == 'uniform':
        q.zero_()
    elif family == 'peaked':
        q.mul_(4)
        k.mul_(4)
    device = torch.device('npu', torch.npu.current_device())
    return q.to(device), k.to(device), v.to(device), table.to(device), torch.tensor(lengths, dtype=torch.int32,
                                                                                  device=device), table


def reference(torch, q, k, v, table_cpu, lengths):
    rows = []
    group = q.shape[1] // k.shape[1]
    sq = q.shape[2]
    query_index = torch.arange(sq, device=q.device).repeat(group)
    for batch, length in enumerate(lengths):
        count = (length + 127) // 128
        ids = table_cpu[batch, :count].to(device=k.device, dtype=torch.int64)
        keys = k.index_select(0, ids).transpose(0, 1).flatten(1, 2)[:, :length].float()
        values = v.index_select(0, ids).transpose(0, 1).flatten(1, 2)[:, :length].float()
        queries = q[batch].reshape(k.shape[1], group * sq, q.shape[-1]).float()
        logits = (queries @ keys.transpose(-1, -2)) * q.shape[-1]**-0.5
        if sq > 1:
            keys_after_query = torch.arange(length, device=q.device)[None, :] > (length - sq + query_index[:, None])
            logits.masked_fill_(keys_after_query, -float('inf'))
        rows.append((logits.softmax(-1) @ values).reshape(q.shape[1:]))
    return torch.stack(rows)


def prepared(runtime, sq, lengths, family, seed):
    torch, torch_npu, kernel = runtime
    q, k, v, table, device_lengths, table_cpu = make_inputs(torch, sq, lengths, family, seed)
    causal = {} if sq == 1 else dict(sparse_mode=3, atten_mask=torch.ones((2048, 2048), dtype=torch.bool,
                                                                         device=q.device).triu_(1))
    seq_lengths = list(lengths)

    def ascendc():
        result, _ = torch_npu.npu_fused_infer_attention_score(
            q, k, v, num_heads=q.shape[1], num_key_value_heads=k.shape[1], input_layout='BNSD',
            scale=q.shape[-1]**-0.5, actual_seq_lengths_kv=seq_lengths, block_table=table, block_size=128, **causal)
        return result

    schedule = os.environ.get('TLA_PAGED_GQA_SCHEDULE')
    prefill_q_tile = int(value) if (value := os.environ.get('TLA_PAGED_GQA_Q_TILE')) else None
    column = (value == '1') if (value := os.environ.get('TLA_PAGED_GQA_COLUMN_SOFTMAX')) is not None else None
    pages_per_tile = int(os.environ.get('TLA_PAGED_GQA_PAGES_PER_TILE', '1'))
    compact_decode = sq == 1 and os.environ.get('TLA_PAGED_GQA_COMPACT_DECODE') == '1'
    value = os.environ.get('TLA_PAGED_GQA_LATE_VALUE')
    late_value = None if value is None else sq > 1 and value == '1'
    calls = dict(pipeline=kernel.prepare_call(q, k, v, table, device_lengths, schedule=schedule,
                                             prefill_q_tile=prefill_q_tile, column_softmax=column,
                                             pages_per_tile=pages_per_tile, compact_decode=compact_decode,
                                             late_value=late_value),
                 stage0=kernel.prepare_call(q, k, v, table, device_lengths, prefetch=False,
                                           prefill_q_tile=prefill_q_tile), ascendc=ascendc)
    return calls, reference(torch, q, k, v, table_cpu, lengths)


def precision(torch, call, expected):
    actual = call().float()
    torch.npu.synchronize()
    assert actual.shape == expected.shape and torch.isfinite(actual).all()
    error = actual - expected
    maximum = (error.abs().amax() / expected.abs().amax().clamp_min(1e-30)).item()
    rms = (error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)).item()
    assert maximum <= 0.005 and rms <= 0.005, (maximum, rms)
    return dict(normalized_max_error=maximum, relative_rms=rms)


def accuracy(runtime, sq, lengths):
    torch, _, _ = runtime
    records = []
    for family, seed in INPUTS:
        calls, expected = prepared(runtime, sq, lengths, family, seed)
        checks = {name: [precision(torch, call, expected) for _ in range(3)] for name, call in calls.items()}
        records.append(dict(family=family, seed=seed, checks=checks))
        del calls, expected
    return records


@pytest.mark.parametrize('case', ('decode', 'prefill', 'ragged'))
def test_run(runtime, case):
    sq, lengths = (1, TAIL_LENGTHS) if case == 'ragged' else CASES[case]
    accuracy(runtime, sq, lengths)


@pytest.mark.parametrize('case', tuple(CASES))
def test_perf(request, runtime, case):
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'{case}-', dir=OUTPUT))
    sq, lengths = CASES[case]
    spec = dict(bs=len(lengths), num_head=32, kv_head=8, head_dim=128, q_len=sq, kv_len=lengths[0], page=128,
                dtype='bf16', mask='right_aligned_causal' if sq > 1 else 'none')
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator=f'paged_gqa_{case}', specification=spec, passed=False, profiles={},
                  compiler=dict(path=str(compiler), sha256=compiler_sha256), triton_path=triton.__file__,
                  kernel_sha256=hashlib.sha256(OPERATOR.read_bytes()).hexdigest())
    try:
        report['accuracy'] = accuracy(runtime, sq, lengths)
        if case == 'decode':
            report['ragged_accuracy'] = accuracy(runtime, 1, TAIL_LENGTHS)
        calls, expected = prepared(runtime, sq, lengths, *INPUTS[0])
        report['launch_options'] = {name: getattr(call, 'metadata', {}) for name, call in calls.items()}
        report['performance_precheck'] = {name: precision(torch, call, expected) for name, call in calls.items()}
        names = list(calls)
        for rnd in range(ROUNDS):
            for name in names[rnd:] + names[:rnd]:
                measurement = profile(calls[name], folder / f'{name}-{rnd}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measurement)
                assert measurement['occupancy_consistent'], f'NPU occupancy changed during {name} round {rnd}'
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
        print(f'Paged GQA results: {folder}')
