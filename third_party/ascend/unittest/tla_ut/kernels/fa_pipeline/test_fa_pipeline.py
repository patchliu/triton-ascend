"""Eight FA shapes: pipeline/task, original TLA FA and native AscendC."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

# Direct-script mode uses the same import root as tla_ut/conftest.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.fa import testing
from kernels.fa.fa_kernel import CASES as FA_CASES
from kernels.utils import load_kernel

CASES = tuple(FA_CASES)
ACCURACY_INPUTS = (('normal', 20261030), ('normal', 20261031), ('uniform', 20261101), ('peaked', 20261102))
ACCURACY_REPEATS = 3
RUN_PERF = False
PERF_ROUNDS, PERF_SAMPLES = 3, 40
# Diagnostic counters also retain the actual AscendC launch dimensions. Their
# durations are separate from the Level1 samples used for the performance table.
PERF_COUNTERS = True
OUTPUT = Path('fa-pipeline-results')
OPERATOR = Path(__file__).with_name('fa_kernel.py')


@pytest.fixture
def runtime(request):
    return testing.runtime(request, OPERATOR, RUN_PERF)


@pytest.mark.parametrize('case', CASES)
def test_run(runtime, case):
    testing.run_accuracy(runtime, case, ACCURACY_INPUTS, ACCURACY_REPEATS, OUTPUT, compare_fa=True)


@pytest.mark.parametrize('case', CASES)
def test_perf(runtime, case):
    testing.run_performance(runtime, case, PERF_ROUNDS, PERF_SAMPLES, PERF_COUNTERS, OUTPUT, compare_fa=True)


def ascendc_launch(record):
    rows = record.get('pipe_counters', {}).get('ascendc', [])
    attention = [
        row for row in rows if 'flashattention' in ''.join(c.lower()
                                                           for c in row.get('Op Name', '') + row.get('OP Type', '')
                                                           if c.isalnum())
    ]
    if not attention:
        raise ValueError('Missing AscendC FlashAttention launch metadata')
    dimensions = {(int(row['Block Num']), int(row['Mix Block Num'])) for row in attention}
    if dimensions != {(28, 56)}:
        raise ValueError(f'AscendC launch differs from 28 AIC / 56 AIV: {dimensions}')
    return dict(aic=28, aiv=56, operations=sorted({row['Op Name'] for row in attention}))


def full_accuracy(record, source_hash):
    if not record.get('passed'):
        return False
    required = set(ACCURACY_INPUTS)
    for operator in ('candidate', 'fa', 'ascendc'):
        rows = [row for row in record.get('records', []) if row.get('operator') == operator]
        covered = {(row['family'], row['seed'])
                   for row in rows
                   if row.get('passed') and len(row.get('checks', [])) >= ACCURACY_REPEATS}
        if not required <= covered:
            return False
        if operator == 'candidate' and any(row.get('metadata', {}).get('source_sha256') != source_hash for row in rows):
            return False
    return True


def export_table(output, cases=CASES):
    """Only publish correct, idle-device results from this exact source file."""
    output = Path(output)
    source_hash = hashlib.sha256(OPERATOR.read_bytes()).hexdigest()
    records = {}
    for path in sorted(output.glob('FA*-performance-*/results.json'), key=lambda p: p.stat().st_mtime):
        record = json.loads(path.read_text())
        metadata = record.get('metadata', {}).get('candidate', {})
        profiles = record.get('profiles', {})
        full_samples = (set(profiles) == {'candidate', 'fa', 'ascendc'} and all(
            len(runs) >= PERF_ROUNDS and all(len(row.get('samples_us', [])) >= PERF_SAMPLES
                                             for row in runs)
            for runs in profiles.values()))
        if (record.get('case') in cases and record.get('passed') and record.get('occupancy_consistent') and full_samples
                and metadata.get('source_sha256') == source_hash):
            # Device-event timings alone do not expose the native core count.
            # Refuse to publish a comparison with unverified launch dimensions.
            ascendc_launch(record)
            records[record['case']] = (path, record)
    accuracy = {}
    for path in output.glob('FA*-accuracy-*/results.json'):
        record = json.loads(path.read_text())
        if full_accuracy(record, source_hash):
            accuracy[record['case']] = path
    missing = [case for case in cases if case not in records or case not in accuracy]
    if missing:
        raise RuntimeError('Missing current-source accuracy or uncontested performance: ' + ', '.join(missing))
    rows = []
    for case in cases:
        path, record = records[case]
        s, d, causal = FA_CASES[case]
        times = record['device_median_us']
        assert set(times) == {'candidate', 'fa', 'ascendc'}
        rows.append(
            dict(case=case, S=s, D=d, causal=causal, pipeline_us=times['candidate'], original_fa_us=times['fa'],
                 ascendc_us=times['ascendc'], ascendc_over_pipeline=times['ascendc'] / times['candidate'],
                 original_fa_over_pipeline=times['fa'] / times['candidate'], ascendc_aic=28, ascendc_aiv=56,
                 performance_record=str(path), accuracy_record=str(accuracy.get(case, '')), source_sha256=source_hash))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'performance.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / 'performance.json').write_text(json.dumps(rows, indent=2))
    lines = [
        '| Case | S | D | Causal | Pipeline µs | Original FA µs | AscendC µs | AscendC / Pipeline |',
        '|---|---:|---:|:---:|---:|---:|---:|---:|'
    ]
    for row in rows:
        lines.append(f"| {row['case']} | {row['S']} | {row['D']} | {row['causal']} | "
                     f"{row['pipeline_us']:.3f} | {row['original_fa_us']:.3f} | {row['ascendc_us']:.3f} | "
                     f"{row['ascendc_over_pipeline']:.3f}x |")
    (output / 'performance.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('accuracy', 'performance', 'smoke', 'table', 'all'), default='accuracy')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--rounds', type=int, default=PERF_ROUNDS)
    parser.add_argument('--samples', type=int, default=PERF_SAMPLES)
    args = parser.parse_args()
    if args.mode == 'table':
        export_table(args.output, args.cases)
        return
    kernel = load_kernel(OPERATOR)
    for case in args.cases:
        if args.mode in ('accuracy', 'smoke', 'all'):
            inputs = ACCURACY_INPUTS if args.mode != 'smoke' else ACCURACY_INPUTS[:1]
            repeats = ACCURACY_REPEATS if args.mode != 'smoke' else 1
            testing.run_accuracy(kernel, case, inputs, repeats, args.output, compare_fa=True)
        if args.mode in ('performance', 'smoke', 'all'):
            testing.run_performance(kernel, case, args.rounds, args.samples, PERF_COUNTERS, args.output,
                                    compare_fa=True)
    if args.mode == 'all':
        export_table(args.output, args.cases)


if __name__ == '__main__':
    main()
