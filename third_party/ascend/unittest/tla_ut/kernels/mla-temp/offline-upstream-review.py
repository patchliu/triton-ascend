"""Compile the final and frozen schedules without requiring a device."""
from pathlib import Path
import importlib.util
import sys

from utils.tla_test_utils import compile_offline, lower_offline

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    name = path.parent.name.replace('-', '_') + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


signature = dict(Q='*bf16', KV='*bf16', BLOCK_TABLE='*i32', KV_LENGTHS='*i32',
                 NUMERATOR='*fp32', MAXIMUM='*fp32', DENOMINATOR='*fp32')
for version in ('mla-tla', 'mla-pipeline'):
    module = load(ROOT/version/'mla.py')
    for ragged in (False, True):
        pages = 9 if ragged else 1120
        batch = 4 if ragged else 8
        constants = dict(BATCH=batch, QUERY_LENGTH=4, QUERY_HEADS=24, MAX_PAGES=pages,
                         SPLIT_PAGES=(7 if ragged else 161) if version=='mla-tla' else (2 if ragged else 160),
                         SPLITS=(6 if ragged else 5) if version=='mla-tla' else (5 if ragged else 7),
                         CUBE_CORES=28 if version=='mla-tla' else (20 if ragged else 28),
                         SCALE=0.04, Q_TILE_ROWS=96, KV_L2=4)
        sig = signature
        function = module.attention_splits if version=='mla-pipeline' else module.attention
        if version=='mla-tla':
            sig = dict(signature, OUT='*bf16', LSE='*fp32')
            constants.update(FUSE_MERGE=True, RETURN_LSE=True)
        output = ROOT/'mla-temp/offline-upstream'/version/('ragged' if ragged else 'full')
        ir, context = compile_offline(function, sig, constants, output)
        lowered = str(lower_offline(ir, context, output))
        assert 'hivm.hir.mmad' in lowered
        if version=='mla-pipeline':
            assert 'hivm.pipeline' in lowered and 'hivm.hir.sync_block_wait' not in lowered
        print('IR_PASS', version, 'ragged' if ragged else 'full', flush=True)
