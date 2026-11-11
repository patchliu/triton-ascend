def device_cores(device):
    """Read the available hardware grid through the active Triton driver."""
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        properties = driver.active.utils.get_device_properties(device)
    return properties['num_aicore'], properties['num_vectorcore']


def prepare_call(q, kv, block_table, kv_lengths, *, scale=0.04, return_lse=True,
                 head_tile=None, split_pages=None, kv_l2=4):
    """Prepare paged MLA for packed BF16 Q and a single latent KV head.

    Q is [batch, query_length, query_heads, 576], KV is [pages, 128, 576].
    The first 512 KV columns supply both latent K and V, and the final
    64 columns supply RoPE K. Causality is aligned to the end of each KV
    sequence. Every actual KV length must be in [query_length, table capacity].
    The table must address valid physical pages below that actual length.
    A prepared call captures the tensor objects, and reads their current
    contents on every invocation. It allocates fresh output and workspace.
    """
    import torch

    if q.ndim != 4 or kv.ndim != 3:
        raise ValueError('Expected Q [B,Sq,H,576] and KV [pages,128,576]')
    batch, query_length, heads, width = q.shape
    if min(batch, query_length, heads) <= 0 or width != 576 or kv.shape[0] <= 0 or kv.shape[1:] != (128, 576):
        raise ValueError('Expected nonempty packed Q/KV with 576 columns and 128-token pages')
    if block_table.ndim != 2 or block_table.shape[0] != batch or kv_lengths.shape != (batch,):
        raise ValueError('Expected block table [B,max_pages] and KV lengths [B]')
    if block_table.shape[1] * 128 < query_length:
        raise ValueError('The page table capacity must cover the query length')
    if q.dtype != torch.bfloat16 or kv.dtype != q.dtype:
        raise ValueError('Q and KV must both be BF16')
    if block_table.dtype != torch.int32 or kv_lengths.dtype != torch.int32:
        raise ValueError('Page metadata must be int32')
    tensors = (q, kv, block_table, kv_lengths)
    if q.device.type != 'npu' or any(t.device != q.device or not t.is_contiguous() for t in tensors):
        raise ValueError('Inputs must be contiguous and on the same Ascend NPU')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Softmax scale must be finite and positive')
    if head_tile is None:
        head_tile = 96 if heads % 32 and query_length * heads % 96 == 0 else (64 if heads % 64 == 0 else 32)
    if head_tile not in (32, 64, 96) or query_length * heads % head_tile:
        raise ValueError('Query rows must form complete 32-, 64- or 96-row tiles')
    if heads % head_tile and (head_tile // 2) % heads:
        raise ValueError('Each Vector sub-block must cover complete query head groups')
    if split_pages is not None and (not isinstance(split_pages, int) or split_pages <= 0):
        raise ValueError('split_pages must be a positive integer')

    cube_cores, vector_cores = device_cores(q.device)
    if cube_cores <= 0 or vector_cores != 2 * cube_cores:
        raise ValueError(f'Expected two Vector cores per Cube core, got {cube_cores}:{vector_cores}')
    rows = batch * query_length * heads
    query_tiles = rows // head_tile
    max_pages = block_table.shape[1]
    if split_pages is None:
        groups = cube_cores // math.gcd(query_tiles, cube_cores)
        split_pages = triton.cdiv(max_pages, groups) if head_tile == 96 else 64
    if EXPLICIT_L0:
        split_pages = triton.cdiv(split_pages, 7) * 7
    splits = triton.cdiv(max_pages, split_pages)
    if EXPLICIT_L0:
        request_tiles = triton.cdiv(max_pages * 128, 112)
        tiles_per_core = triton.cdiv(query_tiles * request_tiles, cube_cores)
        stream_splits = min(cube_cores, triton.cdiv(request_tiles + tiles_per_core - 1, tiles_per_core))
        if stream_splits <= 7:
            splits = stream_splits
        fuse_merge = splits <= 7
    else:
        fuse_merge = False
    launch_cores = cube_cores if fuse_merge else min(cube_cores, query_tiles * splits)
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
    if EXPLICIT_L0:
        options.update(unit_flag=False, sync_solver=True, disable_auto_inject_sync=True)

    def call():
        numerator = torch.empty((splits, rows, 512), dtype=torch.float32, device=q.device)
        maximum = torch.empty((splits, rows), dtype=torch.float32, device=q.device)
        denominator = torch.empty_like(maximum)
        output = torch.empty((batch, query_length, heads, 512), dtype=q.dtype, device=q.device)
        lse = torch.empty((batch, query_length, heads), dtype=torch.float32, device=q.device) if return_lse else maximum
        args = (q, kv, block_table, kv_lengths, numerator, maximum, denominator,
                batch, query_length, heads, max_pages, split_pages, splits, launch_cores, scale, head_tile, kv_l2)
        with torch.npu.device(q.device):
            if EXPLICIT_L0:
                attention[(launch_cores,)](*args, output, lse, fuse_merge, return_lse, **options)
            else:
                attention_splits[(launch_cores,)](*args, **options)
            if not fuse_merge:
                merge_splits[(rows,)](numerator, maximum, denominator, output, lse, rows, splits,
                                      triton.next_power_of_2(splits), 512, return_lse, sanitize_overflow=False)
        return (output, lse) if return_lse else output

    call.metadata = dict(cube_grid=launch_cores, vector_cores=vector_cores, head_tile=head_tile,
                         split_pages=split_pages, splits=splits, kv_tile=112 if EXPLICIT_L0 else 128,
                         fuse_merge=fuse_merge, return_lse=return_lse, scale=scale, kv_l2=kv_l2)
    return call
