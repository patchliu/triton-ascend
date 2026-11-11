"""Thin binding for the CANN 9.1 SparseFlashMLA HCA baseline."""

from contextlib import ExitStack
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import platform
import re


class Unavailable(RuntimeError):
    """The expected ACLNN ABI or runtime is unavailable."""


_TENSOR = ('const aclTensor*', ct.c_void_p)
_I64 = ('int64_t', ct.c_int64)
_STRING = ('char*', ct.c_char_p)
_BOOL = ('bool', ct.c_bool)
_OUT_ARGS = (('workspaceSize', 'uint64_t*', ct.POINTER(ct.c_uint64)),
             ('executor', 'aclOpExecutor**', ct.POINTER(ct.c_void_p)))


def _params(kind, names):
    return tuple((name, *kind) for name in names.split())


_PLAN_ARGS = (
    _params(_TENSOR, 'cuSeqLensQOptional cuSeqLensOriKvOptional cuSeqLensCmpKvOptional sequsedQOptional '
                     'sequsedOriKvOptional')
    + _params(_I64, 'numHeadsQ numHeadsKv headDim batchSize maxSeqlenQ maxSeqlenKv oriTopK cmpTopK cmpRatio '
                    'oriMaskMode cmpMaskMode oriWinLeft oriWinRight')
    + _params(_STRING, 'layoutQOptional layoutKvOptional')
    + _params(_BOOL, 'hasOriKv hasCmpKv')
    + _params(_TENSOR, 'metadata') + _OUT_ARGS
)
_APPLY_ARGS = (
    _params(_TENSOR, 'q oriKvOptional cmpKvOptional oriSparseIndicesOptional cmpSparseIndicesOptional '
                     'oriBlockTableOptional cmpBlockTableOptional cuSeqlensQOptional cuSeqlensOriKvOptional '
                     'cuSeqlensCmpKvOptional sequsedQOptional sequsedOriKvOptional sequsedCmpKvOptional '
                     'cmpResidualKvOptional oriTopkLengthOptional cmpTopkLengthOptional sinksOptional metadataOptional')
    + (('softmaxScale', 'double', ct.c_double),)
    + _params(_I64, 'cmpRatio oriMaskMode cmpMaskMode oriWinLeft oriWinRight')
    + _params(_STRING, 'layoutQOptional layoutKvOptional')
    + _params(_I64, 'topkValueMode oriKvStride cmpKvStride')
    + _params(_BOOL, 'returnSoftmaxLse')
    + _params(_TENSOR, 'attnOutOut softmaxLseOutOptional') + _OUT_ARGS
)


def _check_signature(header, symbol, specification):
    source = header.read_text()
    declaration = re.search(r'\b' + symbol + r'\s*\(([^;]*)\)\s*;', source)
    if declaration is None:
        raise Unavailable(f'{symbol} is missing from {header}')
    normalized = lambda value: re.sub(r'\s+', '', value)
    actual = tuple(normalized(parameter) for parameter in declaration.group(1).split(','))
    expected = tuple(normalized(kind + name) for name, kind, _ in specification)
    if actual != expected:
        raise Unavailable(f'Unrecognized {symbol} ABI in {header}')


def _bind(library, name, arguments, result=ct.c_int):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


class _Api:
    def __init__(self, root):
        headers = root / 'opp/include/aclnnop'
        metadata = 'aclnnSparseFlashMlaMetadata'
        attention = 'aclnnSparseFlashMla'
        suffix = 'GetWorkspaceSize'
        _check_signature(headers / 'aclnn_sparse_flash_mla_metadata.h', metadata + suffix, _PLAN_ARGS)
        _check_signature(headers / 'aclnn_sparse_flash_mla.h', attention + suffix, _APPLY_ARGS)

        lib_dir = root / f'{platform.machine()}-linux/lib64'
        if not lib_dir.is_dir():
            lib_dir = root / 'lib64'
        self.library = (lib_dir / 'libopapi.so').resolve()
        opapi = ct.CDLL(str(self.library))
        tensors = ct.CDLL(str(lib_dir / 'libnnopbase.so'))
        runtime = ct.CDLL(str(lib_dir / 'libascendcl.so'))
        i64_array = ct.POINTER(ct.c_int64)
        self.create_tensor = _bind(tensors, 'aclCreateTensor', [i64_array, ct.c_uint64, ct.c_int, i64_array, ct.c_int64,
                                                               ct.c_int, i64_array, ct.c_uint64, ct.c_void_p],
                                   ct.c_void_p)
        self.destroy_tensor = _bind(tensors, 'aclDestroyTensor', [ct.c_void_p])
        self.destroy_executor = _bind(tensors, 'aclDestroyAclOpExecutor', [ct.c_void_p])
        self.recent_error = _bind(runtime, 'aclGetRecentErrMsg', [], ct.c_char_p)
        self.plan_size = _bind(opapi, metadata + suffix, [ctype for _, _, ctype in _PLAN_ARGS])
        self.apply_size = _bind(opapi, attention + suffix, [ctype for _, _, ctype in _APPLY_ARGS])
        launch_args = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
        self.plan = _bind(opapi, metadata, launch_args)
        self.apply = _bind(opapi, attention, launch_args)

    def _check_status(self, status, name):
        if status:
            detail = (self.recent_error() or b'').decode(errors='replace')
            raise RuntimeError(f'{name} failed with ACLNN status {status}\n{detail}')

    def tensor(self, torch, stack, tensor):
        if tensor is None:
            return None
        # ACL_FLOAT=0, ACL_INT32=3, ACL_BF16=27 and ACL_FORMAT_ND=2.
        dtype = {torch.float32: 0, torch.int32: 3, torch.bfloat16: 27}[tensor.dtype]
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            raise ValueError('ACLNN baseline expects contiguous ND tensors with zero storage offset')
        dims = (ct.c_int64 * tensor.ndim)(*tensor.shape)
        strides = (ct.c_int64 * tensor.ndim)(*tensor.stride())
        handle = self.create_tensor(dims, tensor.ndim, dtype, strides, 0, 2, dims, tensor.ndim, tensor.data_ptr())
        if not handle:
            raise RuntimeError('aclCreateTensor returned a null descriptor')
        stack.callback(self.destroy_tensor, handle)
        return handle

    def launch(self, torch, device, size_function, kernel, args):
        workspace_size = ct.c_uint64()
        executor = ct.c_void_p()
        self._check_status(size_function(*args, ct.byref(workspace_size), ct.byref(executor)), size_function.__name__)
        try:
            workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=device)
            stream = torch.npu.current_stream(device).npu_stream
        except BaseException:
            self.destroy_executor(executor)
            raise
        self._check_status(kernel(workspace.data_ptr(), workspace_size, executor, stream), kernel.__name__)


@lru_cache(maxsize=1)
def _load_api(root):
    try:
        return _Api(root)
    except (OSError, AttributeError) as error:
        raise Unavailable(f'Cannot load CANN SparseFlashMLA from {root}: {error}') from error


def prepare_native(torch, q, original, compact, sink, *, host_cores=None):
    """Return fixed-metadata and request-time-metadata C128 HCA calls."""
    ratio, window, head_dim = 128, 128, 512
    if q.ndim != 3 or q.shape[1:] != (64, head_dim) or sink.shape != (64,):
        raise ValueError('Expected Q [Sq,64,512] and sink [64]')
    sq, history, compact_len = q.shape[0], original.shape[0], compact.shape[0]
    if history % ratio:
        raise Unavailable('The installed ACLNN metadata ABI cannot express a compressed history residual')
    if original.shape != (history, 1, head_dim) or compact.shape != (history // ratio, 1, head_dim):
        raise ValueError('Expected the original and compressed TND KV inputs for C128 HCA')
    device = q.device
    if not 1 <= sq <= history or device.type != 'npu' or any(t.device != device for t in (original, compact, sink)):
        raise ValueError('Expected right-aligned queries and inputs on the same NPU')
    if any(t.dtype != torch.bfloat16 for t in (q, original, compact)) or sink.dtype != torch.float32:
        raise ValueError('Expected BF16 Q/KV and FP32 sink')
    path = os.environ.get('ASCEND_HOME_PATH')
    if not path:
        raise Unavailable('ASCEND_HOME_PATH must name the CANN SDK used by torch_npu')
    api = _load_api(Path(path).resolve())
    cu_q, cu_ori, cu_cmp = [torch.tensor([0, n], device=device, dtype=torch.int32) for n in (sq, history, compact_len)]
    empty_length = torch.empty(0, dtype=torch.int32, device=device)
    residual = torch.tensor([history % ratio], device=device, dtype=torch.int32)
    layout = b'TND'

    def device_metadata():
        # 1024 int32 values in the installed CANN 9.1 kernel metadata header.
        out = torch.empty(1024, dtype=torch.int32, device=device)
        with ExitStack() as stack:
            # This metadata API makes every sequence descriptor contiguous. Its absent-input sentinel is [0] int32.
            lengths = tuple(api.tensor(torch, stack, t) for t in (cu_q, cu_ori, cu_cmp, empty_length, empty_length))
            meta = api.tensor(torch, stack, out)
            args = (*lengths, 64, 1, head_dim, 1, sq, history, 0, 0, ratio, 4, 3,
                    window - 1, 0, layout, layout, True, True, meta)
            api.launch(torch, device, api.plan_size, api.plan, args)
        return out

    metadata_dir = os.environ.get('TLA_HCA_METADATA_DIR')
    host_path = None
    metadata = device_metadata
    if metadata_dir:
        if host_cores is None:
            raise ValueError('Host metadata requires the actual device core counts')
        cubes, vectors = host_cores
        host_path = Path(metadata_dir).resolve() / f'metadata-{sq}-{history}-{cubes}-{vectors}.bin'
        if not host_path.is_file():
            raise Unavailable(f'No host-produced SparseFlashMLA metadata for this shape and device at {host_path}')
        data = bytearray(host_path.read_bytes())
        if len(data) != 1024 * 4:
            raise Unavailable(f'Expected 1024 int32 metadata values in {host_path}')

        def metadata():
            return torch.frombuffer(data, dtype=torch.int32).to(device)

    def attention(plan):
        out = torch.empty_like(q)
        with ExitStack() as stack:
            tensors = (q, original, compact, None, None, None, None, cu_q, cu_ori, cu_cmp, None, None, None,
                       residual, None, None, sink, plan)
            descriptors = tuple(api.tensor(torch, stack, t) for t in tensors)
            # Contiguous TND uses an element stride of 512 between KV rows.
            args = (*descriptors, head_dim**-0.5, ratio, 4, 3, window - 1, 0, layout, layout, 1, original.stride(0),
                    compact.stride(0), False, api.tensor(torch, stack, out), None)
            api.launch(torch, device, api.apply_size, api.apply, args)
        return out

    cached = metadata()

    def call():
        return attention(cached)

    call.metadata = dict(backend='aclnn_cann_9_1', library=str(api.library), sdk=str(Path(path).resolve()),
                         metadata_device='precomputed_host' if host_path else 'npu',
                         metadata_file=str(host_path) if host_path else None)
    return call, (None if host_path else lambda: attention(metadata()))
