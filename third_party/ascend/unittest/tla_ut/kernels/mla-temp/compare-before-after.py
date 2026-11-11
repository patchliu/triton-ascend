"""Development-only paired measurements against the frozen best source."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
import torch_npu


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'mla-tla'))
import benchmark as bench


def main():
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    sentinel = torch.empty(1, device='npu')
    torch.npu.synchronize()
    out = ROOT / 'mla-temp' / sys.argv[1]
    out.mkdir(exist_ok=False)
    paths = dict(previous_host=ROOT/'mla-temp/original/mla_decode.py',
                 previous_l0=ROOT/'mla-temp/original/mla_decode_explicit.py',
                 final_l0=ROOT/'mla-tla/mla.py', final_pipeline=ROOT/'mla-pipeline/mla.py')
    modules = {name: bench.load(path) for name, path in paths.items()}
    sample = Path.home()/'workspace/npuir/temp/20261110-cannbot-flash-mla.L50di4/source/samples/flash_mla_with_kvcache'
    dsl_record = bench.install_dsl_sample(sample, out/'dsl-sample')
    report = dict(dsl_sample=dsl_record, phase='precision', source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                  for name,path in paths.items()}, precision={}, formal={})

    def save():
        (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')

    def prepare(inputs, ragged=False):
        q, kv, table, lengths = inputs
        args = q, None, kv, None, table, lengths
        common = dict(head_tile=96, scale=0.04, return_lse=True, kv_l2=4)
        old = modules['previous_host']
        calls = dict(
            previous_l0=old.prepare_call(*args, split_pages=7 if ragged else None,
                         l0_mode='native112_merge_dynflags_stream_k',
                         _split_kernel=modules['previous_l0'].mla_split_explicit, **common),
            final_l0=modules['final_l0'].prepare_call(*inputs, split_pages=2 if ragged else None),
            previous_pipeline=old.prepare_call(*args, split_pages=2 if ragged else None, **common),
            final_pipeline=modules['final_pipeline'].prepare_call(*inputs, split_pages=2 if ragged else None))
        if not ragged:
            calls['ascendc'] = bench.prepare_ascendc(inputs)
            calls['cannbot'] = bench.prepare_cannbot(inputs)
        return calls

    bench.wait_exclusive()
    for case in ('ragged','full'):
        inputs = bench.make_inputs(bench.CASES[case], 'normal', 1103)
        expected = bench.reference(inputs)
        calls = prepare(inputs, case=='ragged')
        for name, call in calls.items():
            report['precision'].setdefault(name,{})[case] = bench.precision(call, expected)
        save()
        del calls, expected, inputs
    inputs = bench.make_inputs(bench.CASES['full'], 'normal', 1103)
    calls = prepare(inputs)
    report['options'] = {name: call.metadata for name,call in calls.items()}
    report['phase'] = 'performance'
    names = list(calls)
    for round_id in range(6):
        offset = round_id % len(names)
        for name in names[offset:]+names[:offset]:
            measured = bench.checked_profile(calls[name], out/f'round-{round_id}-{name}.trace.json', 40)
            report['formal'].setdefault(name,[]).append(measured)
            save()
    report['medians'] = {name: dict(
        device_us=statistics.median([v for r in runs for v in r['samples_us']]),
        wall_us=statistics.median([v for r in runs for v in r['wall_samples_us']]))
        for name,runs in report['formal'].items()}
    report['relative_to_previous'] = {
        kind: report['medians']['previous_'+kind]['device_us']/report['medians']['final_'+kind]['device_us']
        for kind in ('l0','pipeline')}
    report['phase'] = 'complete'
    save()
    print('COMPLETE',json.dumps(report['medians']),flush=True)


if __name__ == '__main__':
    main()
