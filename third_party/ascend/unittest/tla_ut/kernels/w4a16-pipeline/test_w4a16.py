"""Groupwise W4A16 with the same ND-packed weights on both backends."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('w4a16.py')
M, K, N, GROUP = 128, 4096, 12288, 128
FAMILIES = (('normal', 1103), ('normal', 1104), ('all_codes', 1103))
ROUNDS, SAMPLES = 3, 40


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('W4A16 device checks require the run stage')
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
@pytest.mark.parametrize('m_tile,n_tile,unpack_bytes', ((64, 128, False), (128, 512, True)))
def test_ir(stage, m_tile, n_tile, unpack_bytes, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    signature = dict(A='*bf16', PACKED='*u8' if unpack_bytes else '*i32', SCALE='*bf16', O='*bf16')
    constants = dict(M=M, K=K, N=N, GROUP=GROUP, CORES=28, COMPUTE_STAGE=stage, BM=m_tile, BN=n_tile,
                     GROUPS_PER_TILE=1, UNPACK_BYTES=unpack_bytes)
    if output_dir is not None:
        output_dir = output_dir / f'm{m_tile}-n{n_tile}-stage-{stage}'
    module, ctx = compile_offline(load_kernel(OPERATOR).matmul, signature, constants, output_dir)
    lowered = str(lower_offline(module, ctx, output_dir))
    assert 'hivm.pipeline' in lowered
    for task in ('load_activation', 'dequantize_weight', 'matmul'):
        assert f'hivm.task_name = "{task}"' in lowered


def prepared(runtime, family, seed):
    torch, torch_npu, kernel = runtime
    device = torch.device('npu', torch.npu.current_device())
    generator = torch.Generator().manual_seed(seed)
    activation = torch.randn((M, K), dtype=torch.bfloat16, generator=generator).to(device)
    scale = (torch.rand((K // GROUP, N), generator=generator) / 8 + 0.025).to(dtype=torch.bfloat16, device=device)
    if family == 'all_codes':
        weights = ((torch.arange(K)[:, None] + torch.arange(N)[None, :]) % 16 - 8).to(torch.int8)
    else:
        weights = torch.randint(-8, 8, (K, N), dtype=torch.int8, generator=generator)
    weights = weights.to(device)
    packed = torch_npu.npu_convert_weight_to_int4pack(weights.to(torch.int32))
    assert packed.shape == (K, N // 8) and packed.dtype == torch.int32 and packed.is_contiguous()
    shifts = torch.arange(8, dtype=torch.int64) * 4
    first_words = packed[:4, :16].cpu().long().unsqueeze(-1)
    nibbles = (first_words >> shifts) & 15
    assert torch.equal(((nibbles ^ 8) - 8).view(4, 128), weights[:4, :128].cpu().long())

    def ascendc():
        return torch_npu.npu_weight_quant_batchmatmul(activation, packed, scale, antiquant_group_size=GROUP)

    tiling = dict(m_tile=int(os.environ.get('TLA_W4A16_M_TILE', '128')),
                  n_tile=int(os.environ.get('TLA_W4A16_N_TILE', '512')),
                  groups_per_tile=int(os.environ.get('TLA_W4A16_GROUPS_PER_TILE', '1')),
                  unpack_bytes=os.environ.get('TLA_W4A16_UNPACK_BYTES', '1') == '1')
    calls = dict(pipeline=kernel.prepare_call(activation, packed, scale, **tiling),
                 stage0=kernel.prepare_call(activation, packed, scale, prefetch=False, **tiling), ascendc=ascendc)
    # Dequantize the independent unpacked weights to BF16, then use FP32 MM.
    expected = torch.empty((M, N), dtype=torch.float32, device=device)
    for start in range(0, N, 512):
        stop = start + 512
        group_weights = weights[:, start:stop].reshape(K // GROUP, GROUP, 512).float()
        dequantized = (group_weights * scale[:, None, start:stop].float()).reshape(K, 512).to(torch.bfloat16)
        expected[:, start:stop] = activation.float() @ dequantized.float()
    return calls, expected


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
    output = Path(os.environ.get('TLA_W4A16_RESULTS', 'w4a16-results'))
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='nd-groupwise-', dir=output))
    spec = dict(m=M, k=K, n=N, group_size=GROUP, weight='int4_signed_nd', activation='bf16', scale='bf16')
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as binary:
        report = dict(specification=spec, triton_path=triton.__file__, compiler_path=str(compiler),
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
