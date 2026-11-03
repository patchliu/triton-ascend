"""Runtime-only DSA accuracy, eager latency and NPU profiling entry point."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

# Device imports are deferred so --help works on a host without NPU packages.
def make_inputs(shape, seed, family):
    """Generate inputs before timing; keep index order and multiplicity."""
    batch, queries, heads, length, selected = shape
    generator = torch.Generator().manual_seed(seed)

    def normal(*dims):
        return torch.randn(dims, generator=generator).bfloat16().npu()

    q = normal(batch, queries, heads, 512)
    kv = normal(batch, length, 1, 512)
    qr = normal(batch, queries, heads, 64)
    kr = normal(batch, length, 1, 64)
    ids = torch.stack([
        torch.randperm(length, generator=generator)[:selected]
        for _ in range(batch * queries)
    ]).reshape(batch, queries, 1, selected).int().npu()
    if family == "uniform":
        q.zero_()
        qr.zero_()
    elif family == "peaked":
        q.mul_(4)
        qr.mul_(4)
    elif family == "duplicates":
        ids[..., 1::2] = ids[..., 0::2]
    elif family == "invalid":
        ids[..., selected // 2:] = -1
    return q, kv, qr, kr, ids

def native(data):
    q, kv, qr, kr, ids = data
    return torch_npu.npu_sparse_flash_attention(
        q, kv, kv, ids, 576 ** -0.5, query_rope=qr, key_rope=kr,
        attention_mode=2, sparse_mode=0, return_softmax_lse=True,
    )

def accuracy(outputs, data, chunk=16):
    """Full-output FP32 reference, chunked only to bound scratch memory.

    CPU-generated BF16 inputs are converted to FP32 on NPU for the reference;
    HF32 is explicitly disabled. This is not a CPU reference or an official
    CANN acceptance rule. Max-row error supplements the attachment's global
    normalization so one erroneous row cannot hide behind another row's scale.
    """
    q, kv, qr, kr, ids = data
    batch, queries, heads, _ = q.shape
    metrics = {model: {key: dict(max_abs=0., reference_max=0., error_sq=0., ref_sq=0.,
                                max_row_normalized=0., finite=True, elements=0)
                       for key in ("output", "maximum", "denominator")}
               for model in outputs}
    for bi in range(batch):
        for start in range(0, queries, chunk):
            end = min(queries, start + chunk)
            index = ids[bi, start:end, 0].long()
            valid = (index >= 0) & (index < kv.shape[1])
            index = index.clamp(0, kv.shape[1] - 1)
            selected = kv[bi, index, 0].float()
            rope = kr[bi, index, 0].float()
            score = (torch.bmm(q[bi, start:end].float(), selected.transpose(1, 2))
                     + torch.bmm(qr[bi, start:end].float(), rope.transpose(1, 2))) / 24.
            score.masked_fill_(~valid[:, None, :], -float("inf"))
            maximum = score.amax(-1)
            exponent = (score - maximum[..., None]).exp()
            denominator = exponent.sum(-1)
            reference = (torch.bmm(exponent / denominator[..., None], selected),
                         maximum, denominator)
            pieces = {model: (value[0][bi, start:end], value[1][bi, 0, start:end],
                               value[2][bi, 0, start:end]) for model, value in outputs.items()}
            for model, values in pieces.items():
                for key, actual, expected in zip(metrics[model], values, reference):
                    if actual.shape != expected.shape:
                        raise AssertionError((model, key, actual.shape, expected.shape))
                    delta = actual.float() - expected
                    row = metrics[model][key]
                    row["max_abs"] = max(row["max_abs"], float(delta.abs().max()))
                    row["reference_max"] = max(row["reference_max"], float(expected.abs().max()))
                    row["error_sq"] += float(delta.square().sum())
                    row["ref_sq"] += float(expected.square().sum())
                    row["finite"] &= bool(actual.isfinite().all())
                    row["elements"] += expected.numel()
                    row_error = delta.abs().amax(-1) / expected.abs().amax(-1).clamp_min(1e-30)
                    row["max_row_normalized"] = max(row["max_row_normalized"], float(row_error.max()))
    for model, values in metrics.items():
        for row, tolerance in zip(values.values(), (.005, .0001, .0001)):
            row["max_normalized"] = row["max_abs"] / max(row["reference_max"], 1e-30)
            row["relative_rms"] = (row["error_sq"] / max(row["ref_sq"], 1e-30)) ** .5
            row["attachment_gate_passed"] = row["finite"] and row["max_normalized"] <= tolerance
            row["attachment_tolerance"] = tolerance
    return {"metrics": metrics}


def profile_tasks(call, path, samples, expected_names):
    for _ in range(3):
        call()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=config,
    ) as profiler:
        for _ in range(samples):
            call()
        torch.npu.synchronize()
    profiler.export_chrome_trace(str(path))
    raw = json.loads(path.read_text())
    events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
    tasks = sorted((event for event in events if event.get("ph") == "X"
                    and str((event.get("args") or {}).get("Task Type", "")).startswith(
                        ("AI_CORE", "AI_VECTOR", "MIX_", "KERNEL"))), key=lambda event: float(event["ts"]))
    if not tasks or len(tasks) % samples:
        raise RuntimeError(f"incomplete profiler record: {len(tasks)} tasks / {samples} calls")
    count = len(tasks) // samples
    names = [event["name"] for event in tasks[:count]]
    if names != expected_names:
        raise RuntimeError(f"Expected all main/merge tasks {expected_names}, got {names}")
    if len({(event.get("pid"), event.get("args", {}).get("Physic Stream Id")) for event in tasks}) != 1:
        raise RuntimeError("Expected a single device and stream for serial task grouping")
    summed, spans = [], []
    for start in range(0, len(tasks), count):
        group = tasks[start:start + count]
        if [event["name"] for event in group] != names:
            raise RuntimeError("device task sequence changed within one profile")
        summed.append(sum(float(event["dur"]) for event in group))
        spans.append(max(float(event["ts"]) + float(event["dur"]) for event in group)
                     - float(group[0]["ts"]))
    return {"task_sum_us": summed, "task_span_us": spans, "tasks_per_call": count,
            "tasks": [{"name": event["name"], "args": event.get("args", {})} for event in tasks[:count]]}

def balanced_order(names, sample, round_id):
    """Balance all six orders and predecessor pairs for three implementations."""
    if len(names) == 3:
        orders = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1))
        return [names[index] for index in orders[(sample + round_id) % 6]]
    rotate = (sample + round_id) % len(names)
    return names[rotate:] + names[:rotate]

def eager_samples(calls, samples, round_id):
    """Rotate order every sample; include wrapper allocation and all launches."""
    result = {name: [] for name in calls}
    names = list(calls)
    for sample in range(samples):
        for name in balanced_order(names, sample, round_id):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            output = calls[name]()
            torch.npu.synchronize()
            result[name].append((time.perf_counter_ns() - start) / 1000)
            del output
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['decode_b1'], help='case names, or all')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--op', choices=['dsa', 'ascendc', 'both'], default='dsa')
    parser.add_argument('--mode', choices=['accuracy', 'timing', 'profile'], default='accuracy')
    parser.add_argument('--api-mode', choices=['public', 'legacy'], default='public')
    parser.add_argument('--family', choices=['normal', 'uniform', 'peaked', 'duplicates', 'invalid'], default='normal')
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out', type=Path, default=Path('dsa-results'))
    args = parser.parse_args()
    if args.samples <= 0 or args.rounds <= 0:
        parser.error('samples and rounds must be positive')
    # Read only the literal shape table for host-side listing and validation.
    import ast
    module = ast.parse(Path(__file__).with_name('dsa.py').read_text())
    shapes = next(ast.literal_eval(n.value) for n in module.body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CASE_SHAPES' for t in n.targets))
    if args.list:
        for name, shape in shapes.items():
            print(name, shape)
        return
    cases = list(shapes) if args.cases == ['all'] else args.cases
    if any(name not in shapes for name in cases):
        parser.error('unknown case; use --list')
    if args.family == 'invalid' and args.op != 'dsa':
        parser.error('invalid indices are only tested with --op dsa')
    global torch, torch_npu
    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    torch.npu.matmul.allow_hf32 = False
    os.environ['DSA_API_MODE'] = args.api_mode
    if args.op != 'ascendc':
        import dsa
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'arguments': vars(args) | {'out': str(args.out)},
              'torch': torch.__version__, 'results': []}
    for case in cases:
        data = make_inputs(shapes[case], args.seed, args.family)
        calls = {}
        if args.op != 'ascendc':
            calls['dsa'] = lambda: dsa.forward(*data)
        if args.op != 'dsa':
            calls['ascendc'] = lambda: native(data)
        outputs = {name: call() for name, call in calls.items()}
        torch.npu.synchronize()
        result = {'case': case, 'accuracy': accuracy(outputs, data)}
        report['results'].append(result)
        passed = all(row['attachment_gate_passed'] for model in result['accuracy']['metrics'].values() for row in model.values())
        (args.out / 'results.json').write_text(json.dumps(report, indent=2))
        if not passed:
            raise AssertionError(f'{case}: accuracy failed; see results.json')
        del outputs
        if args.mode != 'accuracy':
            for call in calls.values():
                for _ in range(3):
                    call()
            torch.npu.synchronize()
            result['eager_us'] = [eager_samples(calls, args.samples, r) for r in range(args.rounds)]
            result['eager_median_us'] = {name: statistics.median(v for r in result['eager_us'] for v in r[name]) for name in calls}
        if args.mode == 'profile':
            result['profiles'] = {}
            for name, call in calls.items():
                # Discover the full device task sequence in one complete call.
                trace = args.out / f'{case}-{name}-warmup.json'
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]) as prof:
                    call()
                    torch.npu.synchronize()
                prof.export_chrome_trace(str(trace))
                raw = json.loads(trace.read_text())
                events = raw.get('traceEvents', []) if isinstance(raw, dict) else raw
                tasks = sorted((e for e in events if e.get('ph') == 'X' and str((e.get('args') or {}).get('Task Type', '')).startswith(('AI_CORE', 'AI_VECTOR', 'MIX_', 'KERNEL'))), key=lambda e: float(e['ts']))
                result['profiles'][name] = profile_tasks(call, args.out / f'{case}-{name}.json', args.samples, [e['name'] for e in tasks])
        (args.out / 'results.json').write_text(json.dumps(report, indent=2))
        print(f'{case}: accuracy PASS', result.get('eager_median_us', {}), flush=True)
        del calls, data


if __name__ == '__main__':
    main()
