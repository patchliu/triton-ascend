"""FP32 block inverse for independent 64-row WY systems."""

import triton
import triton.language as tl
from triton.language.extra.cann import extension as cann


@triton.jit
def _inverse_diagonal(LOWER, DIAGONAL, BT: tl.constexpr, N: tl.constexpr):
    block = tl.program_id(0)
    tiles: tl.constexpr = BT // N
    system, tile = block // tiles, block % tiles
    index = tl.arange(0, N)
    row = tile * N + index
    lower = tl.trans(tl.load(LOWER + (system * BT + row[:, None]) * BT + row[None, :]))
    inverse = (index[:, None] == index[None, :]).to(tl.float32)

    # Every pivot row is finalized by the preceding columns. Apply its contribution to later rows.
    for pivot in range(N - 1):
        column = tl.trans(cann.extract_slice(lower, (pivot, 0), (1, N), (1, 1)))
        pivot_row = cann.extract_slice(inverse, (pivot, 0), (1, N), (1, 1))
        inverse = tl.fma(-column, pivot_row, inverse)

    tl.store(DIAGONAL + (block * N + index[:, None]) * N + index[None, :], inverse)


@triton.jit
def _merge_inverse(LOWER, DIAGONAL, INVERSE, BT: tl.constexpr, N: tl.constexpr):
    block = tl.program_id(0)
    width: tl.constexpr = 2 * N
    tiles: tl.constexpr = BT // width
    system, tile = block // tiles, block % tiles
    index = tl.arange(0, N)
    block_offset = index[:, None] * N + index[None, :]
    diagonal = DIAGONAL + 2 * block * N * N
    left = tl.load(diagonal + block_offset)
    right = tl.load(diagonal + N * N + block_offset)

    row = tile * width + index
    off_diagonal = tl.load(LOWER + (system * BT + row[:, None] + N) * BT + row[None, :])
    merged = -tl.dot(tl.dot(right, off_diagonal, input_precision='ieee'), left, input_precision='ieee')

    offset = (block * width + index[:, None]) * width + index[None, :]
    tl.store(INVERSE + offset, left)
    tl.store(INVERSE + offset + N, tl.full((N, N), 0.0, tl.float32))
    tl.store(INVERSE + offset + N * width, merged)
    tl.store(INVERSE + offset + N * (width + 1), right)


def solve_tril(lower, *, output_dtype, diagonal_size=32):
    """Return the FP32 inverse of (I + L) in the packed FLA layout."""
    import torch

    batch, tokens, heads, chunk = lower.shape
    if (batch, heads, chunk) != (1, 1, 64) or tokens % chunk or not lower.is_contiguous():
        raise ValueError('Expected packed [1,systems*64,1,64] triangular coefficients')
    if lower.dtype != torch.float32 or output_dtype != torch.float32:
        raise ValueError('Expected FP32 triangular coefficients and inverse')
    if diagonal_size not in (16, 32):
        raise ValueError('Expected 16- or 32-row diagonal tiles')
    systems = tokens // chunk
    tiles = systems * (chunk // diagonal_size)
    diagonal = torch.empty((tiles, diagonal_size, diagonal_size), dtype=lower.dtype, device=lower.device)
    _inverse_diagonal[(tiles,)](lower, diagonal, chunk, diagonal_size)
    for size in (16, 32) if diagonal_size == 16 else (32,):
        width = 2 * size
        tiles = systems * (chunk // width)
        merged = torch.empty((tiles, width, width), dtype=lower.dtype, device=lower.device)
        _merge_inverse[(tiles,)](lower, diagonal, merged, chunk, size)
        diagonal = merged
    return diagonal.view_as(lower)
