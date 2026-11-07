"""Shared FA precision/profiling driver; settings live in the pytest entrypoints."""
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import statistics
import tempfile

from kernels.utils import device_snapshot, load_kernel, pipe_counters, profile, save_environment, save_json
from kernels.fa.reference import make_inputs, native_call, reference, check_outputs
from kernels.fa.fa_kernel import CASES


def runtime(request, path, run_perf):
    import pytest
    from utils.tla_test_utils import get_stages
    if not get_stages()[2]:
        pytest.skip('FA tests require the run stage')
    if request.node.originalname == 'test_perf':
        if not run_perf:
            pytest.skip('Set RUN_PERF = True in this test file')
        if hasattr(request.config, 'workerinput'):
            pytest.skip('Run performance tests without pytest-xdist')
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    return load_kernel(path)


@contextmanager
def prepared(kernel, case, family, seed, compare_fa):
    import torch
    previous = torch.npu.matmul.allow_hf32
    calls = {}
    data = refs = None
    try:
        torch.npu.matmul.allow_hf32 = False
        s, d, causal = CASES[case]
        data = make_inputs(s, d, family, seed)
        calls['candidate'] = kernel.prepare_call(*data, causal=causal)
        if compare_fa:
            baseline = load_kernel(Path(__file__).with_name('fa_kernel.py'))
            calls['fa'] = baseline.prepare_call(*data, causal=causal)
        calls['ascendc'] = native_call(*data, causal)
        refs = reference(*data, causal)
        yield calls, refs
    finally:
        calls.clear()
        data = refs = None
        torch.npu.matmul.allow_hf32 = previous
        gc.collect()
        torch.npu.empty_cache()


def precision(call, refs):
    import torch
    for output in getattr(call, 'outputs', ()):
        output.fill_(float('nan'))
    actuals = call()
    torch.npu.synchronize()
    errors = check_outputs(actuals, refs)
    rms = {name: float((actual.float() - ref).square().mean().sqrt() /
                       ref.square().mean().sqrt().clamp_min(1e-30))
           for name, actual, ref in zip(('O', 'max', 'sum'), actuals, refs)}
    return dict(normalized_max_error=errors, relative_rms=rms, finite=True)


def new_report(kernel, case, output, kind, settings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'{case}-{kind}-', dir=output))
    save_environment(folder, kernel, dict(case=case, **settings))
    return folder


def run_accuracy(kernel, case, inputs, repeats, output, compare_fa=False):
    if repeats <= 0 or not inputs:
        raise ValueError('Accuracy requires positive repeats and input families')
    folder = new_report(kernel, case, output, 'accuracy', dict(inputs=inputs, repeats=repeats))
    report = dict(case=case, records=[], passed=False)
    try:
        for family, seed in inputs:
            with prepared(kernel, case, family, seed, compare_fa) as (calls, refs):
                for name, call in calls.items():
                    row = dict(operator=name, family=family, seed=seed,
                               metadata=getattr(call, 'metadata', {}), checks=[], passed=False)
                    report['records'].append(row)
                    for _ in range(repeats):
                        row['checks'].append(precision(call, refs))
                    row['passed'] = True
                    save_json(folder / 'results.json', report)
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
        print(f'FA accuracy: {folder}')


def run_performance(kernel, case, rounds, samples, counters, output, compare_fa=False):
    if rounds <= 0 or samples <= 0:
        raise ValueError('Performance rounds and samples must be positive')
    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'
    folder = new_report(kernel, case, output, 'performance', dict(rounds=rounds, samples=samples))
    report = dict(case=case, checks={}, profiles={}, passed=False)
    try:
        with prepared(kernel, case, 'normal', 20261030, compare_fa) as (calls, refs):
            report['metadata'] = {name: getattr(call, 'metadata', {}) for name, call in calls.items()}
            for name, call in calls.items():
                report['checks'][name] = precision(call, refs)
            order = list(calls)
            for rnd in range(rounds):
                shift = rnd % len(order)
                for name in order[shift:] + order[:shift]:
                    result = profile(calls[name], folder / f'{name}-{rnd}.json', samples)
                    result['round'] = rnd
                    report['profiles'].setdefault(name, []).append(result)
                    save_json(folder / 'results.json', report)
            medians = {name: statistics.median([t for r in runs for t in r['samples_us']])
                       for name, runs in report['profiles'].items()}
            report['device_median_us'] = medians
            report['baseline_over_candidate'] = {name: t / medians['candidate']
                                                  for name, t in medians.items() if name != 'candidate'}
            report['occupancy_consistent'] = all(r['occupancy_consistent']
                                                  for runs in report['profiles'].values() for r in runs)
            if counters:
                report['pipe_counters'] = {name: pipe_counters(call, folder / name) for name, call in calls.items()}
            report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
        print(f'FA performance: {folder}')
        if report.get('device_median_us'):
            print(json.dumps(dict(device_median_us=report['device_median_us'],
                                  baseline_over_candidate=report['baseline_over_candidate']), indent=2))
