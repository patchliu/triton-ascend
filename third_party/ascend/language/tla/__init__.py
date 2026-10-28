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
"""
TLA (Triton Language for Ascend) namespace.

Independent ``tla.*`` programming namespace for Ascend NPU kernels, migrated
from the TLA repo's SIMD VF additions.  It does **not** reuse the existing
``al``/``bl`` general interfaces (different programming paradigm).

Exports: ``alloc_buffer``/``alloc`` (on-chip buffer + versions),
``copy`` (GM<->buffer), ``view`` (GM view), ``scope`` (core-mode scopes:
``tla.scope.simd()`` / ``tla.scope.cube()``),
``simd`` (vector-register ops), ``annotate_buffer_versions``.
"""

import triton

from .core import buffer, buffer_type, vector, vector_type
from .scope import scope
from .alloc import alloc_buffer, alloc
from .copy import copy
from .view import view, reshape
from .aux_ops import annotate_buffer_versions
from . import math
from .math import compensated_rstd
from .debug import device_assert, device_print

# SIMD vector-register ops.  ``simd`` is exported as a module so its ``cast``
# (ave.hir.vextf) does not shadow unrelated top-level names.
from . import simd

# Cube-core ops (``mmad``/``tla.copy`` L1/L0C routes).
from . import cube

# Explicit pipeline / global synchronization.
from . import sync


def jit(fn=None, **kwargs):
    """Thin wrapper over ``triton.jit`` that marks the kernel as a TLA kernel.

    Marked kernels automatically get ``enable_tla_mode=True`` injected into
    their launch kwargs (see ``_install_tla_option_injection``), so callers do
    not need to pass the option explicitly.  Supports the same call forms as
    ``@triton.jit``: bare (``@tla.jit``) or with arguments
    (``@tla.jit(do_not_specialize=[...])``).
    """

    def _wrap(f):
        jf = triton.jit(f, **kwargs)
        jf._is_tla_kernel = True
        return jf

    return _wrap(fn) if fn is not None else _wrap


def _install_tla_option_injection():
    """Inject ``enable_tla_mode=True`` into launches of ``@tla.jit`` kernels.

    The compile flow snapshots ``metadata = {**options.__dict__}`` before
    codegen, so TLA mode cannot be auto-enabled from codegen time.  Instead we
    patch ``JITFunction.run`` so TLA-marked kernels carry the option in their
    launch kwargs before ``parse_options`` builds the options object.  Non-TLA
    kernels (no ``_is_tla_kernel`` marker) are untouched.
    """
    from triton.runtime.jit import JITFunction

    if getattr(JITFunction, "_tla_inject_applied", False):
        return
    _original_run = JITFunction.run

    def _run(self, *args, grid, warmup, **kwargs):
        if getattr(self, "_is_tla_kernel", False):
            kwargs.setdefault("enable_tla_mode", True)
        return _original_run(self, *args, grid=grid, warmup=warmup, **kwargs)

    JITFunction.run = _run
    JITFunction._tla_inject_applied = True


_install_tla_option_injection()

__all__ = [
    "buffer",
    "buffer_type",
    "vector",
    "vector_type",

    # buffer alloc / copy / view
    "alloc_buffer",
    "alloc",
    "copy",
    "view",
    "reshape",

    # core-mode scopes (scope.simd / scope.cube)
    "scope",

    # math helpers / diagnostics
    "math",
    "compensated_rstd",
    "device_assert",
    "device_print",

    # aux ops
    "annotate_buffer_versions",

    # simd ops
    "simd",

    # cube ops
    "cube",

    # explicit synchronization
    "sync",

    # kernel decorator (marks a kernel as TLA and auto-enables TLA mode)
    "jit",
]
