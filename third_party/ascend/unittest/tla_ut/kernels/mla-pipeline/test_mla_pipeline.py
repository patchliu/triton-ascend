"""Output/LSE correctness and reuse of a prepared call with changed inputs."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def runtime():
    if os.environ.get('TLA_TEST_STAGE', 'run') != 'run':
        pytest.skip('Set TLA_TEST_STAGE=run for device validation')
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    import importlib.util

    path = Path(__file__).with_name('benchmark.py')
    spec = importlib.util.spec_from_file_location(
        '_mla_validation_' + path.parent.name.replace('-', '_'), path
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    module = benchmark.load(path.with_name('mla.py'))
    previous = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    yield torch, benchmark, module
    torch.npu.matmul.allow_hf32 = previous


@pytest.mark.parametrize('case', ('full', 'ragged'))
@pytest.mark.parametrize(
    'family,seed', (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
)
def test_accuracy(runtime, case, family, seed):
    _, benchmark, module = runtime
    inputs = benchmark.make_inputs(benchmark.CASES[case], family, seed)
    expected = benchmark.reference(inputs)
    call = module.prepare_call(*inputs, split_pages=2 if case == 'ragged' else None)
    for _ in range(3):
        benchmark.precision(call, expected)


def test_prepared_call_reads_current_inputs(runtime):
    torch, benchmark, module = runtime
    inputs = benchmark.make_inputs(benchmark.CASES['ragged'], 'normal', 8291)
    call = module.prepare_call(*inputs, split_pages=2)
    benchmark.precision(call, benchmark.reference(inputs))
    q, kv, table, lengths = inputs
    q.add_(0.25)
    kv.mul_(0.75)
    # Modify a live request length and a page mapping without rebuilding call.
    lengths[-1] = 901
    table[-1, [0, 1]] = table[-1, [1, 0]]
    benchmark.precision(call, benchmark.reference(inputs))
    output = module.prepare_call(*inputs, split_pages=2, return_lse=False)()
    torch.npu.synchronize()
    expected, _ = benchmark.reference(inputs)
    assert output.dtype == torch.bfloat16 and output.shape == expected.shape
    assert ((output.float() - expected).abs().amax() / expected.abs().amax()).item() <= 0.005
