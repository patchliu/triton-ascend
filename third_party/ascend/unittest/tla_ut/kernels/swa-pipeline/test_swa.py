"""Causal sliding-window GQA against FP32 and Ascend C band attention."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('swa.py')
BATCH, HQ, HKV, SEQ, DIM, WINDOW = 1, 32, 16, 8192, 256, 1024
FAMILIES = (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
ROUNDS, SAMPLES = 3, 40


def candidate_options():
    q_tile = int(os.environ.get('TLA_SWA_Q_TILE', '128'))
    column = os.environ.get('TLA_SWA_COLUMN_SOFTMAX', '1' if q_tile == 128 else '0') == '1'
    kv_tile = int(os.environ.get('TLA_SWA_KV_TILE', '128'))
    schedule = 'early_pv' if kv_tile == 256 else ('overlap' if column else 'prefetch')
    return dict(q_tile=q_tile, skip_interior_mask=os.environ.get('TLA_SWA_SKIP_INTERIOR_MASK', '1') == '1',
                column_softmax=column, schedule=os.environ.get('TLA_SWA_SCHEDULE', schedule), kv_tile=kv_tile,
                late_value=os.environ.get('TLA_SWA_LATE_VALUE', '1' if column else '0') == '1')


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('Sliding-window attention device checks require the run stage')
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


@pytest.mark.parametrize('stage', (0, 1))
def test_ir(stage, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    signature = dict(Q='*bf16', K='*bf16', V='*bf16', O='*bf16', kv_tiles='i32')
    tuning = candidate_options()
    kernel = load_kernel(OPERATOR)
    constants = dict(B=BATCH, HQ=HQ, HKV=HKV, S=SEQ, D=DIM, WINDOW=WINDOW, CORES=28,
                     STAGES=kernel.SCHEDULES[tuning['schedule'] if stage else 'serial'],
                     QT=tuning['q_tile'], SKIP_INTERIOR_MASK=tuning['skip_interior_mask'],
                     COLUMN_SOFTMAX=tuning['column_softmax'], KT=tuning['kv_tile'], LATE_VALUE=tuning['late_value'])
    if output_dir is not None:
        output_dir = output_dir / f"qt{tuning['q_tile']}-skipmask{int(tuning['skip_interior_mask'])}-stage{stage}"
    module, ctx = compile_offline(kernel.sliding_attention, signature, constants, output_dir)
    lowered = str(lower_offline(module, ctx, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_kv', 'qk', 'softmax', 'publish_probability', 'pv', 'accumulate'):
        assert f'hivm.task_name = "{task}"' in lowered


def prepared(runtime, family, seed, *, seq=SEQ, window=WINDOW):
    torch, torch_npu, kernel = runtime
    generator = torch.Generator().manual_seed(seed)
    device = torch.device('npu', torch.npu.current_device())
    q = torch.randn((BATCH, HQ, seq, DIM), dtype=torch.bfloat16, generator=generator)
    k = torch.randn((BATCH, HKV, seq, DIM), dtype=torch.bfloat16, generator=generator)
    v = torch.randn(k.shape, dtype=torch.bfloat16, generator=generator)
    if family == 'uniform':
        q.zero_()
    elif family == 'peaked':
        q.mul_(4)
        k.mul_(4)
    q, k, v = (x.to(device) for x in (q, k, v))
    mask = torch.ones((2048, 2048), dtype=torch.bool, device=device).triu_(1)

    def ascendc():
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v, input_layout='BNSD', num_heads=HQ, num_key_value_heads=HKV, scale=DIM**-0.5,
            sparse_mode=4, atten_mask=mask, pre_tokens=window - 1, next_tokens=0,
        )[0]

    tuning = candidate_options()
    calls = dict(pipeline=kernel.prepare_call(q, k, v, window, **tuning),
                 stage0=kernel.prepare_call(q, k, v, window, prefetch=False, **tuning), ascendc=ascendc)
    return calls, reference(torch, q, k, v, window)


def reference(torch, q, k, v, window):
    result = torch.empty_like(q, dtype=torch.float32)
    group = q.shape[1] // k.shape[1]
    seq = q.shape[2]
    # Reference blocks are independent of the candidate's tiling.
    for start in range(0, seq, 256):
        stop = min(start + 256, seq)
        left = max(start - window + 1, 0)
        query = torch.arange(start, stop, device=q.device)
        key = torch.arange(left, stop, device=q.device)
        distance = query[:, None] - key[None, :]
        mask = (distance < 0) | (distance >= window)
        keys = k[:, :, left:stop].repeat_interleave(group, dim=1).float()
        values = v[:, :, left:stop].repeat_interleave(group, dim=1).float()
        logits = (q[:, :, start:stop].float() @ keys.transpose(-1, -2)) * q.shape[-1]**-0.5
        result[:, :, start:stop] = logits.masked_fill(mask, -float('inf')).softmax(-1) @ values
    return result


def precision(torch, call, expected):
    output = call()
    torch.npu.synchronize()
    assert output.shape == expected.shape and output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    error = output.float() - expected
    maximum = (error.abs().amax() / expected.abs().amax().clamp_min(1e-30)).item()
    rms = (error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)).item()
    assert maximum <= 0.005 and rms <= 0.005, (maximum, rms)
    return dict(normalized_max_error=maximum, relative_rms=rms)


def accuracy(runtime, family, seed):
    torch, _, _ = runtime
    calls, expected = prepared(runtime, family, seed)
    return {name: [precision(torch, call, expected) for _ in range(3)] for name, call in calls.items()}


@pytest.mark.parametrize('family,seed', FAMILIES)
def test_run(runtime, family, seed):
    accuracy(runtime, family, seed)


def check_window(runtime, seq, window):
    torch, _, _ = runtime
    calls, expected = prepared(runtime, 'uniform', 1103, seq=seq, window=window)
    for call in calls.values():
        precision(torch, call, expected)


def test_run_diagonal(runtime):
    check_window(runtime, seq=128, window=1)


def test_run_unaligned_window(runtime):
    check_window(runtime, seq=512, window=193)


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

    torch, _, kernel = runtime
    output = Path(os.environ.get('TLA_SWA_RESULTS', 'swa-results'))
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='band-', dir=output))
    spec = dict(bs=BATCH, num_head=HQ, kv_head=HKV, head_dim=DIM, q_len=SEQ, kv_len=SEQ,
                window=WINDOW, dtype='bf16', mask='causal_band')
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as binary:
        report = dict(specification=spec, tuning=candidate_options(), triton_path=triton.__file__,
                      compiler_path=str(compiler),
                      compiler_sha256=hashlib.file_digest(binary, 'sha256').hexdigest(), profiles={}, passed=False)
    try:
        report['accuracy'] = [dict(family=family, seed=seed, checks=accuracy(runtime, family, seed))
                              for family, seed in FAMILIES]
        calls, expected = prepared(runtime, *FAMILIES[0])
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
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
