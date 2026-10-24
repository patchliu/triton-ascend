# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Create buffer views: kernel pointer entry or in-buffer subview.

One API, two entry forms:

- ``tla.view(ptr, shape, strides=None, offsets=None, space="gm")`` aliases a
  kernel pointer argument as a buffer, optionally shifted by ``offsets`` (a
  ``ptr_to_int`` cast + native ``hivm.hir.pointer_cast``).  ``strides`` are the
  final memref strides, defaulting to row-major.
- ``tla.view(gm, (256,), strides=(1,), offsets=(base,))`` builds a tile of a
  buffer (the former subview role).  Strides are relative to the source view,
  as in memref.subview.

``space`` only applies to a pointer entry; a buffer's space is its own.
"""
from triton.language.core import builtin, tensor, int64, _unwrap_if_constexpr
from .core import buffer, buffer_type, _shape, _row_major_strides


def _check_scalar_int(offset):
    """Return a scalar-integer offset as an int or tensor; reject everything else."""
    offset = _unwrap_if_constexpr(offset)
    if isinstance(offset, tensor):
        if offset.shape or not offset.dtype.is_int():
            raise TypeError("view offsets must be scalar integers")
        return offset
    if type(offset) is not int:
        raise TypeError("view offsets must be scalar integers")
    return offset


def _subview(src, sizes, steps, offsets, semantic):
    """Rank-preserving memref subview with dynamic offsets and static sizes."""
    handles = []
    for offset, size, step, extent in zip(offsets, sizes, steps, src.shape):
        offset = _check_scalar_int(offset)
        if not isinstance(offset, tensor) and (offset < 0 or offset + (size - 1) * step >= extent):
            raise ValueError("static view is outside the source buffer")
        handles.append(semantic.to_tensor(offset).handle)
    ty = buffer_type(sizes, src.dtype, src.space, tuple(a * b for a, b in zip(src.strides, steps)), offset=None)
    handle = semantic.builder.create_tla_subview(src.handle, handles, list(sizes), list(steps))
    return buffer(handle, ty)


def _pointer_entry(src, sizes, strides, offsets, space, semantic):
    """Kernel pointer entry: alias pointer memory as a buffer in one step.

    ``offsets`` (element units) are folded into the pointer address, so a
    dynamic tile such as ``view(ptr, (BM, HID), offsets=(pt * BM, 0))`` emits a
    single ``hivm.hir.pointer_cast`` at the shifted address (no subview op).
    """
    if offsets is not None:
        offsets = tuple(offsets)
        if len(offsets) != len(sizes):
            raise ValueError("view offsets must match buffer rank")
        # Fold the linear element offset sum(off_i * stride_i) into the address.
        shifted = False
        linear = 0
        for off, stride in zip(offsets, strides):
            off = _check_scalar_int(off)
            term = semantic.mul(off, stride, False)
            linear = semantic.add(linear, term, False)
            shifted = shifted or (type(off) is not int or off != 0)
        if shifted:
            src = semantic.add(src, linear, False)  # pointer + element offset
    elem_ty = src.dtype.element_ty
    addr = semantic.cast(src, int64)  # tt.ptr_to_int -> i64 byte address
    # The element offset is already included in addr, so the new buffer starts
    # at offset zero even when its pointer was shifted dynamically.
    ty = buffer_type(sizes, elem_ty, space, strides, offset=0)
    handle = semantic.builder.create_tla_bind(addr.handle, ty.to_ir(semantic.builder))
    return buffer(handle, ty)


@builtin
def view(src, shape, strides=None, offsets=None, space="gm", _semantic=None):
    if isinstance(src, buffer):
        if space != "gm":
            raise ValueError("view space is ignored for a buffer source")
        sizes = _shape(shape)
        steps = (1, ) * len(sizes) if strides is None else _shape(strides)
        offsets = (0, ) * len(sizes) if offsets is None else tuple(offsets)
        if not (len(sizes) == len(src.shape) == len(steps) == len(offsets)):
            raise ValueError("view sizes, offsets and strides must match source rank")
        return _subview(src, sizes, steps, offsets, _semantic)
    if isinstance(src, tensor) and src.dtype.is_ptr():
        sizes = _shape(shape)
        if strides is None:
            strides = _row_major_strides(sizes)
        else:
            strides = _shape(strides)
        if len(strides) != len(sizes):
            raise ValueError("view strides must match buffer rank")
        return _pointer_entry(src, sizes, strides, offsets, space, _semantic)
    raise TypeError("tla.view expects a tla.buffer or a pointer tensor, "
                    f"got {type(src).__name__}")
