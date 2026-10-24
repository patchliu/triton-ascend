"""Dispatch table for TLA ``with`` statement context managers.

``register_with_dispatch()`` merges ``TLA_WITH_DISPATCH`` into
``triton.compiler.code_generator.WITH_DISPATCH``.  It is invoked from
``tla.builder.setup_unified_builder``, which the build-time triton-ascend
patch mounts in ``CodeGenerator.__init__``, so every TLA compilation (offline
``ast_to_ttir`` or the runtime affinity pipeline) has the handlers registered
before ``visit_With`` runs.  Deferring the import of ``code_generator`` into
the function body is required: this package is exec'd while ``triton.language``
is still initializing (``triton.language.extra`` auto-loads it), and importing
``triton.compiler`` at module level would deadlock on ``constexpr``.
"""

from .code_generator import handle_scope_with
from .scope import simd, cube

__all__ = ["TLA_WITH_DISPATCH", "register_with_dispatch"]

# Registry of 'with' statement handlers for the TLA namespace, keyed by the
# context-manager class itself.
TLA_WITH_DISPATCH = {
    simd: handle_scope_with,
    cube: handle_scope_with,
}


def register_with_dispatch():
    # ``WITH_DISPATCH`` is read dynamically by ``visit_With`` at codegen time,
    # so merging before the first kernel compiles is equivalent to the previous
    # build-time patch wiring in community ``code_generator.py``.
    from triton.compiler import code_generator as _code_generator

    _code_generator.WITH_DISPATCH.update(TLA_WITH_DISPATCH)
