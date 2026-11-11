"""Packed FlexAttention forward against an independent mask and Mojo A5."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('flex_attention.py')
HQ, HKV, DIM, WINDOW, GLOBAL = 16, 8, 128, 1024, 4
MAIN_SEGMENTS = (2000, 22000, 2000)
SMALL_SEGMENTS = (65, 191, 33)
SINGLE_SLOT_SEGMENTS = (16, 32, 16)
FAMILIES = (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
ROUNDS, SAMPLES = 3, 40


def candidate_options():
    q_tile = int(os.environ.get('TLA_FLEX_Q_TILE', '128'))
    column = os.environ.get('TLA_FLEX_COLUMN_SOFTMAX', '1' if q_tile == 128 else '0') == '1'
    packed = os.environ.get('TLA_FLEX_PACKED_PREDICATE', '1' if column else '0') == '1'
    return dict(skip_full_mask=os.environ.get('TLA_FLEX_SKIP_FULL_MASK', '1') == '1',
                merge_buckets=os.environ.get('TLA_FLEX_MERGE_BUCKETS', '1') == '1',
                column_softmax=column, packed_predicate=packed,
                deep=os.environ.get('TLA_FLEX_DEEP', '1' if packed else '0') == '1',
                q_tile=q_tile)


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('FlexAttention device checks require the run stage')
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    mojo = pytest.importorskip('mojo_opset.functions')
    mask_module = pytest.importorskip('mojo_opset.utils.flex_attention_mask')
    previous = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    try:
        yield torch, load_kernel(OPERATOR), mojo, mask_module
    finally:
        torch.npu.matmul.allow_hf32 = previous


@pytest.mark.parametrize('stage', (0, 1))
def test_ir(stage, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    signature = dict(Q='*bf16', K='*bf16', V='*bf16', Q_BLOCKS='*i32', KV_BLOCKS='*i32', MASK_SLOTS='*i32',
                     PARTIAL_MASK='*i32', O='*bf16', SLOTS='i32', PLAN_ROWS='i32')
    padded = (2 * sum(MAIN_SEGMENTS) + 127) // 128 * 128
    tuning = candidate_options()
    if tuning['packed_predicate']:
        signature['PARTIAL_MASK'] = '*u8'
    kv_pad = padded + 128 * int(tuning['merge_buckets'])
    constants = dict(HQ=HQ, HKV=HKV, Q_PAD=padded, KV_PAD=kv_pad, D=DIM, CORES=28, COMPUTE_STAGE=stage,
                     SKIP_FULL_MASK=tuning['skip_full_mask'], QT=tuning['q_tile'],
                     COLUMN_SOFTMAX=tuning['column_softmax'], PACKED_PREDICATE=tuning['packed_predicate'],
                     DEEP=tuning['deep'])
    if output_dir is not None:
        output_dir = output_dir / (f"stage-{stage}-qt{tuning['q_tile']}-kvpad{kv_pad}"
                                   f"-skipmask{int(tuning['skip_full_mask'])}")
    module, ctx = compile_offline(kernel.flex_forward, signature, constants, output_dir)
    lowered = str(lower_offline(module, ctx, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_kv', 'qk', 'softmax', 'publish_probability', 'pv', 'accumulate'):
        assert f'hivm.task_name = "{task}"' in lowered


def prepared(runtime, family, seed, segments):
    torch, kernel, mojo, mask_module = runtime
    device = torch.device('npu', torch.npu.current_device())
    generator = torch.Generator().manual_seed(seed)
    doc = sum(segments)
    seq = 2 * doc
    q = torch.randn((1, HQ, seq, DIM), dtype=torch.bfloat16, generator=generator)
    k = torch.randn((1, HKV, seq, DIM), dtype=torch.bfloat16, generator=generator)
    v = torch.randn(k.shape, dtype=torch.bfloat16, generator=generator)
    if family == 'uniform':
        q.zero_()
    elif family == 'peaked':
        q.mul_(3)
        k.mul_(3)
    q, k, v = (tensor.to(device) for tensor in (q, k, v))
    start, image, _ = segments

    def mask_mod(batch, head, q_idx, k_idx):
        same_doc = q_idx // doc == k_idx // doc
        q_pos, k_pos = q_idx % doc, k_idx % doc
        distance = q_pos - k_pos
        local_or_global = (distance >= 0) & ((distance <= WINDOW) | (k_pos < GLOBAL))
        same_image = (q_pos >= start) & (q_pos < start + image) & (k_pos >= start) & (k_pos < start + image)
        return same_doc & (local_or_global | same_image)

    block_mask = mask_module.create_flex_block_mask(mask_mod, B=1, H=1, Q_LEN=seq, KV_LEN=seq,
                                                    device=device, BLOCK_SIZE=128)
    tuning = candidate_options()
    plan = kernel.make_plan(block_mask, merge_buckets=tuning.pop('merge_buckets'),
                            column_softmax=tuning.pop('column_softmax'),
                            packed_predicate=tuning.pop('packed_predicate'))
    calls = dict(pipeline=kernel.prepare_call(q, k, v, plan, **tuning),
                 stage0=kernel.prepare_call(q, k, v, plan, prefetch=False, **tuning),
                 mojo=lambda: mojo.flex_attention(q, k, v, block_mask=block_mask, implementation='triton'))
    if os.environ.get('TLA_FLEX_FIA_BASELINE') == '1':
        native = load_kernel(OPERATOR.with_name('fia_multimodal.py'))
        calls['ascendc'] = native.prepare_call(q, k, v, segments, WINDOW, GLOBAL)
    if seq <= 2 * sum(SMALL_SEGMENTS):
        rows = torch.arange(seq, device=device, dtype=torch.int64)
    else:
        edges = (0, GLOBAL, WINDOW, start, start + image, doc, doc + start, doc + start + image, seq - 1)
        selected = {p for edge in edges for p in range(max(edge - 2, 0), min(edge + 3, seq))}
        selected.update(range(0, seq, 1024))
        rows = torch.tensor(sorted(selected), dtype=torch.int64, device=device)
    expected = reference(torch, q, k, v, rows, segments)
    return calls, rows, expected, len(plan[-1])


def reference(torch, q, k, v, rows, segments):
    doc = sum(segments)
    image_begin, image_end = segments[0], segments[0] + segments[1]
    pos = torch.arange(q.shape[2], device=q.device, dtype=torch.int64)
    distance = rows[:, None] - pos[None, :]
    same_doc = rows[:, None] // doc == pos[None, :] // doc
    image_token = (pos % doc >= image_begin) & (pos % doc < image_end)
    bidirectional = image_token.index_select(0, rows)[:, None] & image_token[None, :]
    visible = same_doc & (bidirectional | ((distance >= 0) & ((distance <= WINDOW) | (pos % doc < GLOBAL))))
    queries = q.index_select(2, rows).float()
    group = q.shape[1] // k.shape[1]
    out = []
    for head in range(k.shape[1]):
        q_group = queries[:, head * group:(head + 1) * group]
        logits = (q_group @ k[:, head:head + 1].float().transpose(-1, -2)) * DIM**-0.5
        out.append(logits.masked_fill(~visible, -float('inf')).softmax(-1) @ v[:, head:head + 1].float())
    return torch.cat(out, dim=1)


def compare(torch, actual, expected):
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    expected = expected.float()
    error = actual.float() - expected
    maximum = (error.abs().amax() / expected.abs().amax().clamp_min(1e-30)).item()
    rms = (error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)).item()
    assert maximum <= 0.006 and rms <= 0.006, (maximum, rms)
    return dict(normalized_max_error=maximum, relative_rms=rms)


def accuracy(runtime, calls, rows, expected):
    torch, _, _, _ = runtime
    checks = []
    for _ in range(3):
        outputs = {name: call() for name, call in calls.items()}
        torch.npu.synchronize()
        fp32 = {name: compare(torch, output.index_select(2, rows), expected) for name, output in outputs.items()}
        full = {name: compare(torch, output, outputs['mojo']) for name, output in outputs.items() if name != 'mojo'}
        checks.append(dict(fp32=fp32, all_rows_against_mojo=full))
    return checks


@pytest.mark.parametrize('family,seed', FAMILIES)
def test_run(runtime, family, seed):
    calls, rows, expected, _ = prepared(runtime, family, seed, SMALL_SEGMENTS)
    accuracy(runtime, calls, rows, expected)


def test_run_single_slot(runtime):
    # Two 64-token documents share one 128-token mask tile. The JIT must
    # retain the pipeline loop for the resulting one-slot bucket.
    calls, rows, expected, buckets = prepared(runtime, *FAMILIES[0], SINGLE_SLOT_SEGMENTS)
    assert buckets == 1
    accuracy(runtime, calls, rows, expected)


def test_run_main(runtime):
    calls, rows, expected, buckets = prepared(runtime, *FAMILIES[0], MAIN_SEGMENTS)
    if candidate_options()['merge_buckets']:
        assert buckets == 3
    accuracy(runtime, calls, rows, expected)


def test_run_empty_rows(runtime):
    torch, kernel, _, mask_module = runtime
    seq = 384
    device = torch.device('npu', torch.npu.current_device())
    q = torch.zeros((1, HQ, seq, DIM), dtype=torch.bfloat16, device=device)
    k = torch.zeros((1, HKV, seq, DIM), dtype=torch.bfloat16, device=device)
    generator = torch.Generator().manual_seed(1103)
    v = torch.randn(k.shape, dtype=torch.bfloat16, generator=generator)
    # No query can read tile 0. Dummy slots must read the dedicated zero tile.
    v[:, :, :128].fill_(float('nan'))
    v = v.to(device)

    def mask_mod(batch, head, q_idx, k_idx):
        full = (q_idx >= 128) & (q_idx < 256) & (k_idx >= 128) & (k_idx < 256)
        partial = (q_idx >= 128) & (q_idx % 4 != 0) & (k_idx >= 256) & (k_idx <= q_idx + 128)
        return full | partial

    block_mask = mask_module.create_flex_block_mask(mask_mod, B=1, H=1, Q_LEN=seq, KV_LEN=seq,
                                                    device=device, BLOCK_SIZE=128)
    tuning = candidate_options()
    plan = kernel.make_plan(block_mask, merge_buckets=tuning.pop('merge_buckets'),
                            column_softmax=tuning.pop('column_softmax'),
                            packed_predicate=tuning.pop('packed_predicate'))
    tags = {tag for _, _, slots in plan[-1] for row in slots.cpu().tolist() for tag in row}
    assert {-2, -1} <= tags and any(tag >= 0 for tag in tags)

    rows = torch.arange(seq, dtype=torch.int64, device=device)
    allowed = mask_mod(0, 0, rows[:, None], rows[None, :])
    weights = allowed.float() / allowed.sum(dim=-1, keepdim=True).clamp_min(1)
    expected = weights[:, 128:] @ v[:, :, 128:].repeat_interleave(HQ // HKV, dim=1).float()
    empty_rows = torch.cat((rows[:128], rows[256::4]))
    for prefetch in (True, False):
        call = kernel.prepare_call(q, k, v, plan, prefetch=prefetch, **tuning)
        output = call()
        compare(torch, output, expected)
        assert (output.index_select(2, empty_rows) == 0).all()


def test_perf(request, runtime):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 to profile on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Run performance tests without pytest-xdist')
    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'

    import triton
    from mojo_opset.kernels.npu_a5_triton import flex_attention as mojo_kernel
    from triton.backends.ascend.utils import _get_npucompiler_path

    _, kernel, _, _ = runtime
    output = Path(os.environ.get('TLA_FLEX_RESULTS', 'flex-attention-results'))
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='multimodal-', dir=output))
    seq = 2 * sum(MAIN_SEGMENTS)
    baselines = ['mojo_a5_triton']
    if os.environ.get('TLA_FLEX_FIA_BASELINE') == '1':
        baselines.append('ascendc_fia_multimodal_composition')
    spec = dict(bs=1, num_head=HQ, kv_head=HKV, head_dim=DIM, q_len=seq, kv_len=seq, dtype='bf16',
                segments=MAIN_SEGMENTS, window=WINDOW, global_prefix=GLOBAL, baselines=baselines)
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as binary:
        report = dict(operator='flex_attention', specification=spec, tuning=candidate_options(), triton_path=triton.__file__,
                      compiler_path=str(compiler),
                      compiler_sha256=hashlib.file_digest(binary, 'sha256').hexdigest(),
                      mojo_sha256=hashlib.sha256(Path(mojo_kernel.__file__).read_bytes()).hexdigest(), profiles={},
                      passed=False)
    try:
        calls, rows, expected, bucket_count = prepared(runtime, *FAMILIES[0], MAIN_SEGMENTS)
        report['buckets'] = bucket_count
        report['accuracy'] = accuracy(runtime, calls, rows, expected)
        order = list(calls)
        for rnd in range(ROUNDS):
            for name in order[rnd:] + order[:rnd]:
                measured = profile(calls[name], folder / f'{name}-{rnd}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measured)
                assert measured['occupancy_consistent'], f'NPU occupancy changed in {name} round {rnd}'
        medians = {name: statistics.median([t for run in runs for t in run['samples_us']])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['pipeline_pct_of_mojo'] = 100 * medians['mojo'] / medians['pipeline']
        if 'ascendc' in medians:
            report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
