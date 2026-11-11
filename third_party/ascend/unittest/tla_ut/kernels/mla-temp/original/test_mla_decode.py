"""Shape-matched paged BF16 MLA against independent FP32 and CANN FIA."""

import hashlib
import os
from pathlib import Path
import statistics
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernels.utils import device_snapshot, load_kernel, profile, save_environment, save_json

OPERATOR = Path(__file__).with_name('mla_decode.py')
EXPLICIT_OPERATOR = Path(__file__).with_name('mla_decode_explicit.py')
CASES = {
    'single-token': (1, 128, (32768,) * 16, 64),
    'multi-token': (8, 64, (131072,) * 4, 64),
    'batch8-h24': (4, 24, (143360,) * 8, 160),
    'ragged': (8, 64, (8, 129, 261, 513), 2),
    # The last request needs more than seven pages so 112-token modes
    # exercise empty splits on the shorter requests too.
    'ragged-h24': (4, 24, (4, 129, 260, 1031), 2),
}
PAGE = 128
INPUTS = (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
ROUNDS = int(os.environ.get('TLA_MLA_ROUNDS', '3'))
SAMPLES = int(os.environ.get('TLA_MLA_SAMPLES', '40'))
RESULTS = Path(os.environ.get('TLA_MLA_DECODE_RESULTS', 'mla-decode-results'))


def tiling(case):
    head_tile = int(os.environ.get('TLA_MLA_HEAD_TILE', '96' if CASES[case][1] == 24 else '64'))
    q_group = int(os.environ.get('TLA_MLA_Q_GROUP', '1'))
    wide_qk = os.environ.get('TLA_MLA_WIDE_QK') == '1'
    default_split_pages = CASES[case][3]
    # Keep the short regression case split, so it still exercises empty splits.
    split_pages = (default_split_pages if case.startswith('ragged')
                   else int(os.environ.get('TLA_MLA_SPLIT_PAGES', str(default_split_pages))))
    if head_tile not in (32, 64, 96) or split_pages < 1 or q_group not in (1, 2):
        raise ValueError('Expected 32, 64 or 96 rows per tile, positive split pages and one or two Q tiles per group')
    if wide_qk and q_group != 2:
        raise ValueError('TLA_MLA_WIDE_QK requires TLA_MLA_Q_GROUP=2')
    return head_tile, split_pages, q_group, wide_qk


def softmax_scale(heads):
    return 0.04 if heads == 24 else 1.0 / 24.0


def baseline_name(case):
    default = 'flash_mla' if case == 'batch8-h24' else 'fia'
    baseline = os.environ.get('TLA_MLA_BASELINE', default)
    if baseline not in ('flash_mla', 'fia'):
        raise ValueError('TLA_MLA_BASELINE must be flash_mla or fia')
    return baseline


def kv_cache_control(case):
    default = '4' if case == 'batch8-h24' else 'default'
    mode = os.environ.get('TLA_MLA_KV_L2', default)
    return None if mode == 'default' else int(mode)


def l0_modes(kernel):
    selected = os.environ.get('TLA_MLA_L0')
    if not selected:
        return ()
    if selected == 'all':
        return kernel.BASE_L0_MODES
    if selected == 'extended':
        return kernel.L0_MODES
    if selected not in kernel.L0_MODES:
        raise ValueError(f'TLA_MLA_L0 must be all, extended or one of {kernel.L0_MODES}')
    return (selected,)


def l0_split_pages(mode, pages):
    return ((pages + 6) // 7) * 7 if mode.startswith(('l1_112', 'native112')) else pages


@pytest.fixture
def runtime():
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('MLA decode accuracy requires the run stage')
    torch = pytest.importorskip('torch')
    torch_npu = pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    previous = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    try:
        yield torch, torch_npu, load_kernel(OPERATOR)
    finally:
        torch.npu.matmul.allow_hf32 = previous


@pytest.mark.parametrize('case', tuple(CASES))
@pytest.mark.parametrize('compute_stage', (0, 1))
def test_ir(case, compute_stage, output_dir):
    from utils.tla_test_utils import compile_offline, lower_offline

    kernel = load_kernel(OPERATOR)
    sq, heads, lengths, _ = CASES[case]
    head_tile, split_pages, q_group, wide_qk = tiling(case)
    split_pv = os.environ.get('TLA_MLA_SPLIT_PV') == '1'
    stride = (max(lengths) + PAGE - 1) // PAGE
    splits = (stride + split_pages - 1) // split_pages
    schedule = os.environ.get('TLA_MLA_SCHEDULE')
    stages = kernel.SCHEDULES[schedule] if compute_stage and schedule else (compute_stage,) * 3
    signature = dict(Q='*bf16', QR='*bf16', KV='*bf16', KR='*bf16', TABLE='*i32', LENGTHS='*i32',
                     NUMERATOR='*fp32', MAXIMUM='*fp32', DENOMINATOR='*fp32')
    explicit_signature = dict(signature, OUT='*bf16', LSE='*fp32')
    constants = dict(B=len(lengths), SQ=sq, H=heads, TABLE_STRIDE=stride, SPLIT_PAGES=split_pages,
                     SPLITS=splits, CORES=min(28, len(lengths) * sq * heads // (head_tile * q_group) * splits),
                     STAGES=stages, SCALE=softmax_scale(heads), HEAD_TILE=head_tile,
                     SKIP_FULL_MASK=bool(compute_stage) and os.environ.get('TLA_MLA_SKIP_FULL_MASK', '1') == '1',
                     KV_MAJOR=bool(compute_stage) and os.environ.get('TLA_MLA_KV_MAJOR', '1') == '1',
                     Q_GROUP=q_group, WIDE_QK=wide_qk, PACKED=baseline_name(case) == 'flash_mla',
                     KV_L2=kv_cache_control(case), SPLIT_PV=split_pv)
    if output_dir is not None:
        output_dir = output_dir / f'{case}-group-{q_group}-wide-{int(wide_qk)}-stage-{compute_stage}'
    explicit_constants = dict(constants, L0_MODE=None, FUSE_MERGE=False, RETURN_LSE=False)
    modes = l0_modes(kernel)
    if os.environ.get('TLA_MLA_PIPELINE_112') == '1':
        pages = ((split_pages + 6) // 7) * 7
        parts = (stride + pages - 1) // pages
        pipeline112 = dict(constants, PACKED=True, SPLIT_PAGES=pages, SPLITS=parts,
                           CORES=min(28, len(lengths) * sq * heads // head_tile * parts),
                           Q_GROUP=1, WIDE_QK=False, SPLIT_PV=True, KV_TILE=112,
                           STAGES=kernel.SCHEDULES['overlap'])
        dump = None if output_dir is None else output_dir / 'pipeline112'
        module, context = compile_offline(kernel.mla_split, signature, pipeline112, dump)
        lowered = str(lower_offline(module, context, dump))
        assert 'hivm.pipeline' in lowered and 'hivm.pipeline_stage = 2 : i32' in lowered
        assert lowered.count('hivm.hir.mmadL1') == 3
        assert 'hivm.hir.mmadL0' not in lowered and 'hivm.hir.sync_block_wait' not in lowered
        if not modes:
            return
    if modes:
        explicit = load_kernel(EXPLICIT_OPERATOR)
        for mode in modes:
            dump = None if output_dir is None else output_dir / mode
            pages = l0_split_pages(mode, split_pages)
            parts = (stride + pages - 1) // pages
            if mode == 'native112_merge_dynflags_stream_k':
                stream_parts = kernel.stream_k_splits(stride, len(lengths) * sq * heads // head_tile, 28)
                if stream_parts <= kernel.FUSED_MERGE_MAX_SPLITS:
                    parts = stream_parts
            l0_constants = dict(explicit_constants, PACKED=True, L0_MODE=mode, SPLIT_PAGES=pages, SPLITS=parts,
                                CORES=min(28, len(lengths) * sq * heads // head_tile * parts))
            if mode in kernel.FUSED_MERGE_MODES and parts <= kernel.FUSED_MERGE_MAX_SPLITS:
                l0_constants.update(CORES=28, FUSE_MERGE=True, RETURN_LSE=True)
            module, context = compile_offline(explicit.mla_split_explicit, explicit_signature,
                                              l0_constants, dump)
            lowered = str(lower_offline(module, context, dump))
            assert 'hivm.pipeline' not in lowered
            if l0_constants['FUSE_MERGE']:
                assert 'hivm.hir.sync_block[<ALL_VECTOR>]' in lowered
            if mode.startswith('native112_merge_dynflags'):
                assert 'sync_block_wait[<CUBE>, <PIPE_MTE1>, <PIPE_MTE1>] flag = %' in lowered
                assert 'sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = %' in lowered
            if mode in kernel.DYNAMIC_112_MODES:
                assert lowered.count('hivm.hir.nd2nz') == 3  # Q and at most two KV page fragments.
            if mode in kernel.L1_112_MODES:
                assert lowered.count('hivm.hir.mmadL1') == 2
                assert 'hivm.hir.mmadL0' not in lowered
            else:
                native_chunks = mode in ('native112_k5', 'native112_bands', 'native112_dynamic') + kernel.MANUAL_L0_MODES
                if mode in ('native112_shared_p', 'native112_merge_dynflags_shared',
                            'native112_merge_dynflags_first_pv', 'native112_merge_dynflags_balanced',
                            'native112_merge_dynflags_dma', 'native112_merge_dynflags_stream',
                            'native112_merge_dynflags_vecsum', 'native112_merge_dynflags_hybrid',
                            'native112_merge_dynflags_stream_k'):
                    expected_copies = 15
                elif native_chunks:
                    expected_copies = 16
                else:
                    expected_copies = 18 if mode in ('reload', 'q') else 17
                assert lowered.count('hivm.hir.mmadL0') == (9 if native_chunks else 10)
                assert lowered.count('hivm.hir.l12l0') == expected_copies
                assert 'hivm.hir.mmadL1' not in lowered
        return
    if os.environ.get('TLA_MLA_EXPLICIT') == '1':
        explicit = load_kernel(EXPLICIT_OPERATOR)
        module, context = compile_offline(explicit.mla_split_explicit, explicit_signature, explicit_constants, output_dir)
        lowered = str(lower_offline(module, context, output_dir))
        assert 'hivm.pipeline' not in lowered
        assert 'hivm.hir.sync_block_set' in lowered and 'hivm.hir.sync_block_wait' in lowered
        assert lowered.count('hivm.hir.mmadL1') == (2 if constants['PACKED'] else 3)
        return
    module, context = compile_offline(kernel.mla_split, signature, dict(constants, KV_TILE=128), output_dir)
    lowered = str(lower_offline(module, context, output_dir))
    assert 'hivm.pipeline' in lowered
    assert 'hivm.task_name = "load_page"' in lowered
    for task in ('qk', 'softmax', 'publish_probability', 'pv', 'accumulate'):
        if task in ('pv', 'accumulate'):
            parts = 2 if split_pv else q_group
        else:
            parts = 1 if wide_qk else q_group
        for part in range(parts):
            name = f'{task}_{part}' if parts > 1 else task
            assert f'hivm.task_name = "{name}"' in lowered


def make_inputs(torch, sq, heads, lengths, family, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    b = len(lengths)
    pages = [(length + PAGE - 1) // PAGE for length in lengths]
    ids = torch.randperm(sum(pages), generator=generator, dtype=torch.int32)
    table = torch.full((b, max(pages)), -1, dtype=torch.int32)
    for row, group in enumerate(ids.split(pages)):
        table[row, :group.numel()] = group

    device = torch.device('npu', torch.npu.current_device())

    def randn(shape, *, query=False):
        if query and family == 'uniform':
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        cpu = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
        if query and family == 'peaked':
            cpu.mul_(8)
        return cpu.to(device)

    q = randn((b, sq, heads, 512), query=True)
    qr = randn((b, sq, heads, 64), query=True)
    kv = randn((ids.numel(), PAGE, 512))
    kr = randn((ids.numel(), PAGE, 64))
    return q, qr, kv, kr, table.to(device), torch.tensor(lengths, dtype=torch.int32, device=device), table


def reference(torch, q, qr, kv, kr, table_cpu, lengths, return_lse=False):
    _, sq, heads, dim = q.shape
    device = q.device
    result = []
    all_lse = []
    for batch, length in enumerate(lengths):
        pages = (length + PAGE - 1) // PAGE
        ids = table_cpu[batch, :pages].to(device=device, dtype=torch.int64)
        keys = kv.index_select(0, ids).reshape(-1, dim)[:length].float()
        keys_rope = kr.index_select(0, ids).reshape(-1, 64)[:length].float()
        key_position = torch.arange(length, device=device)
        query_end = length - sq + torch.arange(sq, device=device).repeat_interleave(heads)
        query = q[batch].reshape(-1, dim).float()
        query_rope = qr[batch].reshape(-1, 64).float()
        # Each row uses one global softmax. Split only the reference Q rows to bound memory.
        parts = []
        lse_parts = []
        for start in range(0, query.shape[0], 16):
            rows = slice(start, start + 16)
            logits = (query[rows] @ keys.T + query_rope[rows] @ keys_rope.T) * softmax_scale(heads)
            logits.masked_fill_(key_position[None, :] > query_end[rows, None], -float('inf'))
            parts.append(logits.softmax(-1) @ keys)
            if return_lse:
                lse_parts.append(logits.logsumexp(-1))
        result.append(torch.cat(parts).reshape(sq, heads, dim))
        if return_lse:
            all_lse.append(torch.cat(lse_parts).reshape(sq, heads))
    output = torch.stack(result)
    return (output, torch.stack(all_lse)) if return_lse else output


def prepare_cannbot(torch, q, kv, table, lengths, used_q, mask, scale, mask_mode, return_lse):
    import cannbotdsl
    import flash_mla_with_kvcache as sample
    from flash_mla_with_kvcache_metadata import flash_mla_with_kvcache_metadata

    batch, sq, heads, width = q.shape
    q_tnd = q.view(batch * sq, heads, width)
    offsets = torch.arange(0, (batch + 1) * sq, sq, dtype=torch.int32, device=q.device)
    metadata = flash_mla_with_kvcache_metadata(
        lengths, heads, 1, cu_seqlens_q=offsets, seqused_q=used_q,
        head_dim_qk=576, head_dim_v=512, mask_mode=mask_mode, layout_q='TND')

    def call():
        out, lse = sample.flash_mla_with_kvcache(
            q_tnd, kv, block_table=table, cache_seqlens=lengths, cu_seqlens_q=offsets, seqused_q=used_q,
            attn_mask=mask, metadata=metadata, head_dim_v=512, softmax_scale=scale, mask_mode=mask_mode,
            layout_q='TND', layout_kv='PA_BBND', layout_out='NTD', return_softmax_lse=return_lse)
        out = out.transpose(0, 1).view(batch, sq, heads, 512)
        return (out, lse.transpose(0, 1).view(batch, sq, heads)) if return_lse else out

    call.metadata = dict(source=sample.__file__, dsl_version=getattr(cannbotdsl, '__version__', None),
                         source_commit=os.environ.get('TLA_MLA_CANNBOT_COMMIT'),
                         checker_extended_h24=os.environ.get('TLA_MLA_CANNBOT_ADAPTED_H24') == '1',
                         layout='TND/PA_BBND/NTD', prepared_metadata=True)
    return call


def prepared(runtime, case, family, seed):
    torch, torch_npu, kernel = runtime
    sq, heads, lengths, _ = CASES[case]
    scale = softmax_scale(heads)
    head_tile, split_pages, q_group, wide_qk = tiling(case)
    split_pv = os.environ.get('TLA_MLA_SPLIT_PV') == '1'
    q, qr, kv, kr, table, lengths_dev, table_cpu = make_inputs(torch, sq, heads, lengths, family, seed)
    mask = None
    if sq > 1:
        mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu_(1)
    official = baseline_name(case) == 'flash_mla'
    return_lse = official and os.environ.get('TLA_MLA_RETURN_LSE', '1') == '1'
    if official:
        import cann_ops_transformer_custom as native
        from cann_ops_transformer_custom import ops

        q_packed = torch.cat((q, qr), dim=-1)
        kv_packed = torch.cat((kv, kr), dim=-1)
        inputs = (q_packed, None, kv_packed, None, table, lengths_dev)
        mask_mode = 3 if sq > 1 else 0
        native_mask = mask.to(torch.int8) if mask is not None else None
        native_kv = kv_packed.unsqueeze(2)
        used_q = torch.full_like(lengths_dev, sq)

        def make_metadata():
            return ops.flash_mla_metadata(lengths_dev, heads, 1, seqused_q=used_q,
                                          head_dim_qk=576, head_dim_v=512, mask_mode=mask_mode, layout_q='BSND')

        metadata = make_metadata()

        def native_call(plan):
            output, lse = ops.flash_mla(
                q_packed, native_kv, block_table=table, cache_seqlens=lengths_dev, seqused_q=used_q,
                attn_mask=native_mask, metadata=plan, head_dim_v=512, softmax_scale=scale, mask_mode=mask_mode,
                layout_q='BSND', layout_kv='PA_BBND', layout_out='BSND', return_softmax_lse=return_lse)
            return (output, lse.transpose(1, 2)) if return_lse else output

        def ascendc():
            return native_call(metadata)

        ascendc.metadata = dict(python_module=native.__file__, prepared_metadata=True,
                                vendor_path=os.environ.get('ASCEND_CUSTOM_OPP_PATH'))
    else:
        inputs = (q, qr, kv, kr, table, lengths_dev)
        native_kv, native_kr = kv.unsqueeze(1), kr.unsqueeze(1)
        native_lengths = list(lengths)

        def ascendc():
            output, _ = torch_npu.npu_fused_infer_attention_score(
                q, native_kv, native_kv, query_rope=qr, key_rope=native_kr, num_heads=heads, num_key_value_heads=1,
                input_layout='BSND', scale=scale, block_table=table, block_size=PAGE,
                actual_seq_lengths_kv=native_lengths, atten_mask=mask, sparse_mode=3 if sq > 1 else 0,
                softmax_lse_flag=False)
            return output

    calls = dict(
        pipeline=kernel.prepare_call(*inputs, head_tile=head_tile, return_lse=return_lse, kv_l2=kv_cache_control(case),
                                     split_pv=split_pv,
                                     split_pages=split_pages, scale=scale, q_group=q_group, wide_qk=wide_qk,
                                     schedule=os.environ.get('TLA_MLA_SCHEDULE'),
                                     skip_full_mask=os.environ.get('TLA_MLA_SKIP_FULL_MASK', '1') == '1',
                                     merge_columns=int(os.environ.get('TLA_MLA_MERGE_COLUMNS', '512')),
                                     kv_major=os.environ.get('TLA_MLA_KV_MAJOR', '1') == '1'),
        stage0=kernel.prepare_call(*inputs, prefetch=False, head_tile=head_tile, return_lse=return_lse,
                                   kv_l2=kv_cache_control(case), split_pv=split_pv,
                                   split_pages=split_pages, scale=scale, q_group=q_group, wide_qk=wide_qk,
                                   skip_full_mask=heads == 24,
                                   merge_columns=int(os.environ.get('TLA_MLA_MERGE_COLUMNS', '512')),
                                   kv_major=False),
        ascendc=ascendc,
    )
    if official:
        calls['ascendc_regenerate'] = lambda: native_call(make_metadata())
        if os.environ.get('TLA_MLA_CANNBOT') == '1':
            calls['cannbot'] = prepare_cannbot(torch, q_packed, native_kv, table, lengths_dev, used_q,
                                               native_mask, scale, mask_mode, return_lse)
    if os.environ.get('TLA_MLA_EXPLICIT') == '1':
        explicit = load_kernel(EXPLICIT_OPERATOR)
        calls['explicit'] = kernel.prepare_call(
            *inputs, head_tile=head_tile, split_pages=split_pages, scale=scale, return_lse=return_lse,
            kv_l2=kv_cache_control(case), q_group=q_group, wide_qk=wide_qk, split_pv=split_pv,
            _split_kernel=explicit.mla_split_explicit)
    modes = l0_modes(kernel)
    pipeline112 = os.environ.get('TLA_MLA_PIPELINE_112') == '1'
    if pipeline112 or modes:
        packed_inputs = inputs if official else (torch.cat((q, qr), dim=-1), None,
                                                 torch.cat((kv, kr), dim=-1), None, table, lengths_dev)
    if pipeline112:
        calls['pipeline112'] = kernel.prepare_call(
            *packed_inputs, head_tile=head_tile, split_pages=((split_pages + 6) // 7) * 7,
            scale=scale, return_lse=return_lse, kv_l2=kv_cache_control(case), kv_tile=112, split_pv=True)
    if modes:
        explicit = load_kernel(EXPLICIT_OPERATOR)
        for mode in modes:
            calls[f'l0_{mode}'] = kernel.prepare_call(
                *packed_inputs, head_tile=head_tile, split_pages=l0_split_pages(mode, split_pages),
                scale=scale, return_lse=return_lse,
                kv_l2=kv_cache_control(case), l0_mode=mode, _split_kernel=explicit.mla_split_explicit)
    return calls, reference(torch, q, qr, kv, kr, table_cpu, lengths, return_lse)


def precision(torch, call, expected):
    actual = call()
    torch.npu.synchronize()

    def check(actual, expected, dtype, max_limit, rms_limit):
        assert actual.shape == expected.shape and actual.dtype == dtype and torch.isfinite(actual).all()
        error = actual.float() - expected
        maximum = (error.abs().amax() / expected.abs().amax().clamp_min(1e-30)).item()
        rms = (error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)).item()
        assert maximum <= max_limit and rms <= rms_limit, (maximum, rms)
        return dict(normalized_max_error=maximum, relative_rms=rms)

    if isinstance(expected, tuple):
        output = check(actual[0], expected[0], torch.bfloat16, 0.005, 0.005)
        output['lse'] = check(actual[1], expected[1], torch.float32, 0.0001, 0.0001)
        return output
    return check(actual, expected, torch.bfloat16, 0.008, 0.005)


def accuracy(runtime, case):
    torch, _, _ = runtime
    records = []
    for family, seed in INPUTS:
        calls, expected = prepared(runtime, case, family, seed)
        checks = {name: [precision(torch, call, expected) for _ in range(3)] for name, call in calls.items()}
        records.append(dict(family=family, seed=seed, checks=checks))
        del calls, expected
    return records


@pytest.mark.parametrize('case', tuple(CASES))
def test_run(runtime, case):
    accuracy(runtime, case)


@pytest.mark.parametrize('case', ('single-token', 'multi-token', 'batch8-h24'))
def test_perf(request, runtime, case):
    if os.environ.get('TLA_RUN_PERF') != '1':
        pytest.skip('Set TLA_RUN_PERF=1 to profile on an idle NPU')
    if hasattr(request.config, 'workerinput'):
        pytest.skip('Run performance tests without pytest-xdist')
    import triton
    from triton.backends.ascend.utils import _get_npucompiler_path

    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'
    torch, _, kernel = runtime
    sq, heads, lengths, _ = CASES[case]
    head_tile, split_pages, q_group, wide_qk = tiling(case)
    official = baseline_name(case) == 'flash_mla'
    spec = dict(bs=len(lengths), num_head=heads, kv_head=1, head_dim_qk=576, head_dim_v=512,
                q_len=sq, kv_len=lengths[0], page=PAGE, dtype='bf16', layout='BSND/PA_BBND' if official else 'BSND/BnNBsD',
                scale=softmax_scale(heads), mask='right_aligned_causal' if sq > 1 else 'none', head_tile=head_tile,
                split_pages=split_pages, q_group=q_group, wide_qk=wide_qk, kv_l2=kv_cache_control(case),
                split_pv=os.environ.get('TLA_MLA_SPLIT_PV') == '1',
                return_lse=official and os.environ.get('TLA_MLA_RETURN_LSE', '1') == '1',
                baseline='ops-transformer/attention/flash_mla' if official else 'torch_npu.npu_fused_infer_attention_score',
                baseline_source_commit=os.environ.get('TLA_MLA_NATIVE_COMMIT') if official else None)
    RESULTS.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'{case}-', dir=RESULTS))
    save_environment(folder, kernel, spec)
    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open('rb') as stream:
        compiler_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(operator=f'dense_mla_{case}', specification=spec, passed=False, profiles={},
                  compiler=dict(path=str(compiler), sha256=compiler_sha), triton_path=triton.__file__,
                  kernel_sha256=hashlib.sha256(OPERATOR.read_bytes()).hexdigest())
    try:
        report['accuracy'] = accuracy(runtime, case)
        calls, expected = prepared(runtime, case, *INPUTS[0])
        report['launch_options'] = {name: getattr(call, 'metadata', {}) for name, call in calls.items()}
        report['performance_precheck'] = {name: precision(torch, call, expected) for name, call in calls.items()}
        names = list(calls)
        for round_id in range(ROUNDS):
            offset = round_id % len(names)
            for name in names[offset:] + names[:offset]:
                measured = profile(calls[name], folder / f'{name}-{round_id}.json', SAMPLES)
                report['profiles'].setdefault(name, []).append(measured)
                assert measured['occupancy_consistent'], f'NPU occupancy changed during {name} round {round_id}'
        medians = {name: statistics.median([t for run in runs for t in run['samples_us']])
                   for name, runs in report['profiles'].items()}
        report['device_median_us'] = medians
        report['device_min_us_by_round'] = {
            name: [min(run['samples_us']) for run in runs] for name, runs in report['profiles'].items()
        }
        report['pipeline_pct_of_ascendc'] = 100 * medians['ascendc'] / medians['pipeline']
        for name in medians:
            if name in ('explicit', 'cannbot', 'pipeline112') or name.startswith('l0_'):
                report[f'{name}_pct_of_ascendc'] = 100 * medians['ascendc'] / medians[name]
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        save_json(folder / 'results.json', report)
