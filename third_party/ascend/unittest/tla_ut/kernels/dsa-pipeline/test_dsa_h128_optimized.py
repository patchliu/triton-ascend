"""H128 shared-KV pipeline: reuse the common DSA precision/performance driver."""
from pathlib import Path
import importlib.util
import sys

import pytest

ENTRY_PATH = Path(__file__).with_name('test_dsa_pipeline.py')
_spec = importlib.util.spec_from_file_location('_dsa_h128_common_driver', ENTRY_PATH)
_entry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _entry
_spec.loader.exec_module(_entry)
_entry.HEADS = (128, )
_entry.OPERATOR = Path(__file__).with_name('dsa_kernel_h128_optimized.py')
_entry.OUTPUT = Path('dsa-h128-optimized-results')
RUN_PERF = False
_entry.RUN_PERF = RUN_PERF
runtime = _entry.runtime


@pytest.mark.parametrize('heads', (128, ))
def test_run(runtime, heads):
    _entry.accuracy(runtime, heads, _entry.OUTPUT)


@pytest.mark.parametrize('heads', (128, ))
def test_perf(runtime, heads):
    _entry.performance(runtime, heads, _entry.OUTPUT)


if __name__ == '__main__':
    _entry.main()
