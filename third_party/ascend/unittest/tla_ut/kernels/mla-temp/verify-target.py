"""Compile the delivered kernels for a specified offline Ascend target."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

from utils.tla_test_utils import Options, compile_offline, lower_offline

ROOT = Path(__file__).resolve().parent.parent


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def load(path):
    name = path.parent.name.replace('-', '_') + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target')
    parser.add_argument('cube_cores', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.cube_cores <= 0:
        parser.error('cube_cores must be positive')

    class TargetOptions(Options):
        arch = args.target

    flags = [f'--target={args.target}', '--enable-triton-kernel-compile=true', '--enable-tla-mode',
             f'--custom-aic-number={args.cube_cores}', f'--custom-aiv-number={2*args.cube_cores}',
             '--enable-optimized-metaop=true', '--enable-auto-bind-sub-block=false', '--enable-preload=false',
             '--enable-auto-multi-buffer=true', '--limit-auto-multi-buffer-of-local-buffer=no-limit',
             '--enable-vf-operand-substitution=true', '--enable-auto-blockify-loop', '--enable-vf-merge-level=1']
    signature = dict(Q='*bf16', KV='*bf16', BLOCK_TABLE='*i32', KV_LENGTHS='*i32',
                     NUMERATOR='*fp32', MAXIMUM='*fp32', DENOMINATOR='*fp32')
    compiler = ROOT/'npuir-package/bin/bishengir-compile'
    for version in ('mla-tla', 'mla-pipeline'):
        explicit = version == 'mla-tla'
        module = load(ROOT/version/'mla.py')
        kernel = module.attention if explicit else module.attention_splits
        for case, batch, pages in (('full', 8, 1120), ('ragged', 4, 9)):
            groups = args.cube_cores // math.gcd(batch, args.cube_cores)
            split_pages = 2 if case == 'ragged' else ceildiv(pages, groups)
            if explicit:
                split_pages = ceildiv(split_pages, 7) * 7
            splits = ceildiv(pages, split_pages)
            if explicit:
                request_tiles = ceildiv(pages * 128, 112)
                per_core = ceildiv(batch * request_tiles, args.cube_cores)
                stream_splits = min(args.cube_cores, ceildiv(request_tiles + per_core - 1, per_core))
                if stream_splits <= 7:
                    splits = stream_splits
            fuse = explicit and splits <= 7
            grid = args.cube_cores if fuse else min(args.cube_cores, batch * splits)
            constants = dict(BATCH=batch, QUERY_LENGTH=4, QUERY_HEADS=24, MAX_PAGES=pages,
                             SPLIT_PAGES=split_pages, SPLITS=splits, CUBE_CORES=grid,
                             SCALE=0.04, Q_TILE_ROWS=96, KV_L2=4)
            sig = signature
            extra = []
            if explicit:
                sig = dict(signature, OUT='*bf16', LSE='*fp32')
                constants.update(FUSE_MERGE=fuse, RETURN_LSE=True)
                extra = ['--disable-hivm-auto-inject-sync', '--enable-hivm-unit-flag-sync=false',
                         '--enable-hivm-graph-sync-solver=true']
            output = args.output/version/case
            output.mkdir(parents=True, exist_ok=False)
            (output/'tiling.json').write_text(json.dumps(constants, indent=2) + '\n')
            ir, context = compile_offline(kernel, sig, constants, output, options=TargetOptions())
            lower_offline(ir, context, output)
            command = [str(compiler), str(output/'01_linalg.mlir'), *flags, *extra,
                       '--save-temps='+str(output/'device-ir'), '-o', str(output/'kernel.o')]
            (output/'compiler-argv.json').write_text(json.dumps(command, indent=2) + '\n')
            with (output/'device.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (output/'kernel.o').open('rb') as binary:
                assert binary.read(4) == b'\x7fELF'
            print('ELF_PASS', args.target, version, case, 'grid', grid, 'splits', splits, flush=True)


if __name__ == '__main__':
    main()
