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
"""tla.device_print: device print via tt.print (outside scopes, UB values only)."""

from triton.language.core import builtin
from triton.language.core import device_print as _device_print

__all__ = ["device_print"]


def _in_scope(builder):
    block = builder.get_insertion_block()
    while block is not None:
        op = block.get_parent()
        if op is None:
            return False
        if op.get_name() == "scope.scope":
            return True
        block = op.get_parent()


def _is_ub(value):
    """True for scalars (no type) or UB-backed values."""
    ty = str(getattr(value, "type", ""))
    return not ty or "#hivm.address_space<ub>" in ty


@builtin
def device_print(prefix, *args, hex=False, _semantic=None):
    """Print values on device; outside tla.scope and UB-backed values only."""
    if _in_scope(_semantic.builder):
        raise RuntimeError("tla.device_print: not allowed inside tla.scope")
    if any(not _is_ub(a) for a in args):
        raise RuntimeError("tla.device_print: only UB values are supported")
    return _device_print(prefix, *args, hex=hex, _semantic=_semantic)
