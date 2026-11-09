"""Sparse MLA: pipeline/task vs original TLA and AscendC, including KV packing."""
from contextlib import contextmanager
from dataclasses import asdict
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.dsa import test_dsa as reference
from kernels.utils import device_snapshot, load_kernel, passed, pipe_counters, profile, save_environment, save_json

HEADS = (128, 64, 32, 16)
SEEDS = (1103, 1104)
FAMILIES = ('normal', 'uniform', 'peaked', 'duplicates', 'invalid', 'invalid_prefix')
RUN_PERF = False
PERF_ROUNDS, PERF_SAMPLES = 3, 40
PERF_COUNTERS = True
OUTPUT = Path('dsa-pipeline-results')
OPERATOR = Path(__file__).with_name('dsa_kernel.py')
ORIGINAL = Path(reference.__file__).with_name('dsa_kernel.py')


def runtime_kernel():
    import torch
    import torch_npu
    from utils import tla_test_utils  # Pins the canonical decorator before loading two TLA modules.
    if not torch.npu.is_available():
        raise RuntimeError('A real Ascend NPU is required')
    reference.torch, reference.torch_npu = torch, torch_npu
    return load_kernel(OPERATOR)


@pytest.fixture
def runtime(request):
    from utils.tla_test_utils import get_stages
    if not get_stages()[2]:
        pytest.skip('DSA device tests require the run stage')
    if request.node.originalname == 'test_perf':
        if not RUN_PERF:
            pytest.skip('Set RUN_PERF=True or use the performance CLI')
        if hasattr(request.config, 'workerinput'):
            pytest.skip('Run performance serially')
    return runtime_kernel()


def hashes():
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (('pipeline', OPERATOR), ('original', ORIGINAL), ('reference', Path(reference.__file__)))
    }


@contextmanager
def prepared(kernel, heads, seed, family):
    import torch
    prior = torch.npu.matmul.allow_hf32
    data = outputs = None
    calls = {}
    try:
        torch.npu.matmul.allow_hf32 = False
        data = reference.make_inputs(heads, seed, 'normal' if family == 'invalid_prefix' else family)
        if family == 'invalid_prefix':
            # Exercise several entirely masked early KV tiles, then both
            # negative and positive out-of-range indices with valid keys left.
            data[-1][..., :1024] = -1
            data[-1][..., 1024::3] = 8192
        baseline = load_kernel(ORIGINAL)
        calls = dict(pipeline=lambda: kernel.forward(*data), original=lambda: baseline.forward(*data))
        # Native -1 semantics differ on the installed CANN. Both TLA operators
        # still verify invalid indices against the independent masked reference.
        if family not in ('invalid', 'invalid_prefix'):
            calls['ascendc'] = lambda: reference.ascendc(data)
        outputs = {}
        for name, call in calls.items():
            outputs[name] = call()
            torch.npu.synchronize()
        checks = reference.check_outputs(outputs, data)
        assert passed(checks), json.dumps(dict(heads=heads, seed=seed, family=family, checks=checks), indent=2)
        outputs = None
        yield calls, checks
    finally:
        torch.npu.matmul.allow_hf32 = prior
        data = outputs = None
        calls.clear()
        gc.collect()
        torch.npu.empty_cache()


def report_dir(kernel, heads, output, kind, settings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'H{heads}-{kind}-', dir=output))
    save_environment(folder, kernel, dict(heads=heads, **settings))
    return folder


def accuracy(kernel, heads, output, smoke=False):
    folder = report_dir(kernel, heads, output, 'accuracy', dict(smoke=smoke))
    report = dict(heads=heads, hashes=hashes(), records=[], passed=False)
    try:
        for seed in SEEDS[:1] if smoke else SEEDS:
            for family in FAMILIES[:1] if smoke else FAMILIES:
                with prepared(kernel, heads, seed, family) as (_, checks):
                    report['records'].append(dict(seed=seed, family=family, checks=checks))
                    save_json(folder / 'results.json', report)
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
        print('DSA accuracy:', folder, flush=True)


def performance(kernel, heads, output, rounds=PERF_ROUNDS, samples=PERF_SAMPLES):
    if rounds <= 0 or samples <= 0:
        raise ValueError('rounds and samples must be positive')
    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0 and not (set(snapshot['pids']) - {str(os.getpid())}), snapshot
    folder = report_dir(kernel, heads, output, 'performance', dict(rounds=rounds, samples=samples))
    report = dict(heads=heads, hashes=hashes(), config=asdict(kernel.default_config(heads)), profiles={}, passed=False)
    try:
        with prepared(kernel, heads, SEEDS[0], 'normal') as (calls, checks):
            report['checks'] = checks
            names = list(calls)
            for rnd in range(rounds):
                order = names[rnd % len(names):] + names[:rnd % len(names)]
                for name in order:
                    row = profile(calls[name], folder / f'{name}-{rnd}.json', samples)
                    row['round'] = rnd
                    report['profiles'].setdefault(name, []).append(row)
                    save_json(folder / 'results.json', report)
            report['device_median_us'] = {
                name: statistics.median(t
                                        for row in rows
                                        for t in row['samples_us'])
                for name, rows in report['profiles'].items()
            }
            report['wall_median_us'] = {
                name: statistics.median(t
                                        for row in rows
                                        for t in row['wall_samples_us'])
                for name, rows in report['profiles'].items()
            }
            report['occupancy_consistent'] = all(row['occupancy_consistent']
                                                 for rows in report['profiles'].values()
                                                 for row in rows)
            if PERF_COUNTERS:
                report['pipe_counters'] = {name: pipe_counters(call, folder / name) for name, call in calls.items()}
            report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', report)
        print('DSA performance:', folder, report.get('device_median_us'), flush=True)


@pytest.mark.parametrize('heads', HEADS)
def test_run(runtime, heads):
    accuracy(runtime, heads, OUTPUT)


@pytest.mark.parametrize('heads', HEADS)
def test_perf(runtime, heads):
    performance(runtime, heads, OUTPUT)


def table(output, heads=HEADS):
    output = Path(output)
    expected = hashes()
    accuracy_records, performance_records = {}, {}
    for path in sorted(output.glob('H*-*/results.json'), key=lambda p: p.stat().st_mtime):
        row = json.loads(path.read_text())
        if row.get('hashes') != expected or not row.get('passed'):
            continue
        if '-accuracy-' in path.parent.name:
            coverage = {(x['seed'], x['family']) for x in row['records'] if passed(x['checks'])}
            if coverage >= {(s, f) for s in SEEDS for f in FAMILIES}:
                accuracy_records[row['heads']] = path
        elif (row.get('occupancy_consistent') and row.get('pipe_counters', {}).get('ascendc')
              and set(row.get('profiles', {})) == {'pipeline', 'original', 'ascendc'} and all(
                  len(runs) >= PERF_ROUNDS and all(len(x.get('samples_us', [])) >= PERF_SAMPLES
                                                   for x in runs)
                  for runs in row['profiles'].values())):
            performance_records[row['heads']] = (path, row)
    missing = [h for h in heads if h not in accuracy_records or h not in performance_records]
    if missing:
        raise RuntimeError('Missing current-source precision/uncontested performance: ' + str(missing))
    rows = []
    for h in heads:
        path, record = performance_records[h]
        times = record['device_median_us']
        wall = record['wall_median_us']
        rows.append(
            dict(heads=h, pipeline_us=times['pipeline'], original_us=times['original'], ascendc_us=times['ascendc'],
                 ascendc_over_pipeline=times['ascendc'] / times['pipeline'],
                 original_over_pipeline=times['original'] / times['pipeline'], pipeline_wall_us=wall['pipeline'],
                 original_wall_us=wall['original'], ascendc_wall_us=wall['ascendc'],
                 pipeline_sha256=expected['pipeline'], accuracy_record=str(accuracy_records[h]),
                 performance_record=str(path)))
    with (output / 'performance.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / 'performance.json', rows)
    lines = ['| H | Pipeline µs | Original TLA µs | AscendC µs | AscendC / Pipeline |', '|---:|---:|---:|---:|---:|']
    lines += [
        f"| {r['heads']} | {r['pipeline_us']:.3f} | {r['original_us']:.3f} | {r['ascendc_us']:.3f} | {r['ascendc_over_pipeline']:.3f}x |"
        for r in rows
    ]
    (output / 'performance.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('accuracy', 'performance', 'smoke', 'table', 'all'), default='accuracy')
    p.add_argument('--heads', type=int, nargs='+', choices=HEADS, default=HEADS)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--rounds', type=int, default=PERF_ROUNDS)
    p.add_argument('--samples', type=int, default=PERF_SAMPLES)
    args = p.parse_args()
    if args.mode == 'table':
        table(args.output, args.heads)
        return
    kernel = runtime_kernel()
    for h in args.heads:
        if args.mode in ('accuracy', 'all', 'smoke'):
            accuracy(kernel, h, args.output, smoke=args.mode == 'smoke')
        if args.mode in ('performance', 'all', 'smoke'):
            performance(kernel, h, args.output, args.rounds, args.samples)
    if args.mode == 'all': table(args.output, args.heads)


if __name__ == '__main__':
    main()
