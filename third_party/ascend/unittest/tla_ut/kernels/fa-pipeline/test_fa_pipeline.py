"""Full FA8 comparison: task/stage candidate, optimized FA, and native AscendC."""
from pathlib import Path
import pytest
from kernels.fa import testing

ACCURACY_INPUTS = (('normal', 20261030), ('normal', 20261031), ('uniform', 20261101), ('peaked', 20261102))
ACCURACY_REPEATS = 3
RUN_PERF = False
PERF_ROUNDS, PERF_SAMPLES = 3, 40
PERF_COUNTERS = False
OUTPUT = Path('fa-pipeline-results')


@pytest.fixture(params=("v1", "v2", "v3", "v4"))
def runtime(request):
    name = 'fa_pipeline_kernel.py' if request.param == 'v1' else f'fa_pipeline_kernel_{request.param}.py'
    return testing.runtime(request, Path(__file__).with_name(name), RUN_PERF)


def test_run(runtime):
    testing.run_accuracy(runtime, 'FA8', ACCURACY_INPUTS, ACCURACY_REPEATS, OUTPUT, compare_fa=True)


def test_perf(runtime):
    testing.run_performance(runtime, 'FA8', PERF_ROUNDS, PERF_SAMPLES, PERF_COUNTERS, OUTPUT, compare_fa=True)
