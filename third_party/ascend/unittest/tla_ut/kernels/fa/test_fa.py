"""FlashAttention: eight shapes, full precision checks and opt-in profiling."""
from pathlib import Path
import pytest
from kernels.fa import testing

CASES = tuple(f'FA{i}' for i in range(1, 9))
ACCURACY_INPUTS = (('normal', 20261030), ('normal', 20261031), ('uniform', 20261101), ('peaked', 20261102))
ACCURACY_REPEATS = 3
RUN_PERF = False
PERF_ROUNDS, PERF_SAMPLES = 3, 40
PERF_COUNTERS = False
OUTPUT = Path('fa-results')


@pytest.fixture
def runtime(request):
    return testing.runtime(request, Path(__file__).with_name('fa_kernel.py'), RUN_PERF)


@pytest.mark.parametrize('case', CASES)
def test_run(runtime, case):
    testing.run_accuracy(runtime, case, ACCURACY_INPUTS, ACCURACY_REPEATS, OUTPUT)


@pytest.mark.parametrize('case', CASES)
def test_perf(runtime, case):
    testing.run_performance(runtime, case, PERF_ROUNDS, PERF_SAMPLES, PERF_COUNTERS, OUTPUT)
