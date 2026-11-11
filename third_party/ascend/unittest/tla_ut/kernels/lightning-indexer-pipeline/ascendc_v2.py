"""Indices-only binding for the installed CANN 9.1 Lightning Indexer V2."""

from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
from time import perf_counter_ns


@lru_cache(maxsize=1)
def load_bindings():
    import torch
    import torch_npu
    from filelock import FileLock
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension, CppExtension, get_default_build_root

    cann = Path(os.environ['ASCEND_HOME_PATH']).resolve()
    package = cann / 'python/site-packages/npu_ops_transformer'
    sources = package / 'ops/csrc'
    metadata_source = sources / 'lightning_indexer_v2_metadata.cpp'
    indexer_source = sources / 'lightning_indexer_v2.cpp'
    if not metadata_source.is_file() or not indexer_source.is_file():
        raise RuntimeError(f'Expected installed CANN 9.1 V2 bindings under {sources}')

    source = indexer_source.read_text()
    empty_cpu = 'at::empty({0}, at::kFloat)'
    if empty_cpu not in source:
        raise RuntimeError(f'Unexpected CANN 9.1 indices-only binding at {indexer_source}')
    # ACLNN still takes an empty score tensor. Keep its device consistent with Q.
    source = source.replace(empty_cpu, 'at::empty({0}, query.options().dtype(at::kFloat))')
    abi = f'{cann}:{torch.__version__}:{torch_npu.__version__}:{torch._C._GLIBCXX_USE_CXX11_ABI}'
    digest = hashlib.sha256(source.encode() + metadata_source.read_bytes() + abi.encode()).hexdigest()[:12]
    cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', get_default_build_root())) / f'tla_li_v2_{digest}'
    cache.mkdir(parents=True, exist_ok=True)
    indexer_copy = cache / 'lightning_indexer_v2.cpp'
    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    includes = [torch_npu_root / name for name in
                ('include', 'include/third_party/hccl/inc', 'include/third_party/acl/inc')]
    includes.extend((cann / 'include', package / 'common/inc'))
    flags = ['-L' + str(cann / 'lib64'), '-lascendcl', '-L' + str(torch_npu_root / 'lib'), '-ltorch_npu']

    def build(name, path):
        name = f'tla_li_v2_{name}_{digest}'
        distribution = Distribution(dict(name=name, ext_modules=[CppExtension(
            name, [str(path)], include_dirs=list(map(str, includes)),
            extra_compile_args=['-O2', '-g0'], extra_link_args=flags,
        )], cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}))
        command = distribution.get_command_obj('build_ext')
        command.build_lib = str(cache)
        command.build_temp = str(cache / 'objects')
        command.ensure_finalized()
        command.run()
        spec = importlib.util.spec_from_file_location(name, command.get_ext_fullpath(name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    with FileLock(str(cache / 'build.lock')):
        if not indexer_copy.is_file() or indexer_copy.read_text() != source:
            indexer_copy.write_text(source)
        metadata = build('metadata', metadata_source).npu_lightning_indexer_v2_metadata
        indexer = build('index', indexer_copy).npu_lightning_indexer_v2
    return metadata, indexer


def prepare_call(q, k, w, *, topk=2048, reuse_metadata=True):
    """Preserve BF16 inputs, with optional reuse of shape-only V2 metadata."""
    import torch

    if q.ndim != 4 or k.ndim != 4 or q.device.type != 'npu':
        raise ValueError('Expected BSND query and key on NPU')
    b, sq, hq, dim = q.shape
    kb, sk, hkv, kd = k.shape
    if not (b == kb > 0 and sq > 0 and sk > 0 and dim == kd == 128 and 1 <= hq <= 64 and hkv == 1
            and w.shape == (b, sq, hq) and 1 <= topk <= 8192):
        raise ValueError('Expected compatible BSND shape and supported TopK')
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != q.device for t in (q, k, w)):
        raise ValueError('Expected contiguous BF16 Q, K and W on the same NPU')
    metadata, indexer = load_bindings()

    def make_plan():
        return metadata(hq, hkv, dim, topk, None, None, None, None, None,
                        b, sq, sk, 'BSND', 'BSND', 3, 1)

    plan = None
    setup_us = None
    if reuse_metadata:
        with torch.npu.device(q.device):
            torch.npu.synchronize()
            started = perf_counter_ns()
            plan = make_plan()
            torch.npu.synchronize()
            setup_us = (perf_counter_ns() - started) / 1000

    def call():
        with torch.npu.device(q.device):
            current_plan = plan if reuse_metadata else make_plan()
            indices, _ = indexer(q, k, w.float(), topk, None, None, None, None, None, None,
                                 None, current_plan, -1, 'BSND', 'BSND', 3, 1, 0)
        return indices

    call.metadata = dict(provider='ascendc_v2', metadata_reused=reuse_metadata, metadata_setup_wall_us=setup_us,
                         weight_conversion='BF16 to FP32 on every call', return_value=False)
    return call
