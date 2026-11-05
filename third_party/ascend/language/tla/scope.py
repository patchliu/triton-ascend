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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Core-mode scopes and compile-time task/stage annotations for TLA.

Task annotations describe a logical schedule, not an executable pipeline.
The NPU-IR scheduling backend must resolve them before device compilation.
"""
from dataclasses import dataclass
from triton.language.core import _unwrap_if_constexpr

__all__ = ["scope", "vector", "simd", "cube", "pipeline"]


@dataclass(frozen=True)
class _TaskScope:
    kind: type
    name: str
    stage: object = None
    order: object = None

    def __post_init__(self):
        if self.kind not in (vector, cube):
            raise ValueError("task() is only supported on scope.vector() and scope.cube()")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("task name must be a nonempty compile-time string")
        if self.stage is not None and (type(self.stage) is not int or not 0 <= self.stage < 64):
            raise ValueError("task stage must be a compile-time integer in [0, 64) or None")
        if self.order is not None and (type(self.order) is not int or not 0 <= self.order < 2**31):
            raise ValueError("task order must be a nonnegative compile-time i32 integer or None")

    def mlir_attrs(self, builder):
        attrs = self.kind.mlir_attrs(builder)
        attrs["hivm.task_name"] = builder.get_string_attr(self.name)
        if self.stage is not None:
            attrs["hivm.pipeline_stage"] = builder.parse_attr(f"{self.stage} : i32")
        if self.order is not None:
            attrs["hivm.task_order"] = builder.parse_attr(f"{self.order} : i32")
        return attrs


class pipeline:
    """Loop marker: ``for i in tla.pipeline(stop)`` starts a pipeline domain.

    The current interface uses a zero lower bound and unit step. Each pipeline
    needs direct vector/cube scope hints; task names, stages and orders can be
    specified on those scopes. This marker is lowered by the TLA frontend.
    """

    def __init__(self, stop):
        self.stop = stop

    def __iter__(self):
        raise RuntimeError("tla.pipeline can only be used in a @tla.jit for loop")


class scope:
    """Base marker for a core-mode region.

    Subclasses fix the core type (:class:`simd` -> vector core,
    :class:`cube` -> cube core) and the attribute set for the emitted
    ``scope.scope`` (``mlir_attrs``). The Triton code generator routes ``with``
    statements whose context expression evaluates to one of these classes to
    ``code_generator.handle_scope_with`` via ``TLA_WITH_DISPATCH``, so the
    plain and task-annotated scopes share the same core-mode attributes.
    """

    core_mode = "vector"

    @classmethod
    def mlir_attrs(cls, builder):
        """Attributes stamped on the emitted ``scope.scope`` op."""
        return {"hivm.tcore_type": builder.parse_attr("#hivm.tcore_type<VECTOR>")}


class _TaskScopeKind(scope):
    """Execution-core scopes that can own a pipeline task."""

    @classmethod
    def task(cls, name, *, stage=None, order=None):
        """Name a task and optionally specify its stage and mainloop order.

        A task may contain several pipes; the scope itself is not a barrier.
        Missing stages are left unspecified, never defaulted to zero.
        Optional order is a mainloop ordering hint, independent of stage.
        It is passed through as metadata; this interface does not reorder ops.
        """
        return _TaskScope(cls, _unwrap_if_constexpr(name), _unwrap_if_constexpr(stage), _unwrap_if_constexpr(order))


class vector(_TaskScopeKind):
    """``with tla.scope.vector():`` -- vector-core region for non-SIMD ops.

    A vector scope is the vector core's inline region: it hosts non-SIMD
    vector-core ops (``tla.copy`` GM<->UB, control flow) and can nest
    :class:`simd` scopes for the register work. Only ``hivm.tcore_type<VECTOR>``
    is stamped (no VF markers), so ``outline-scope`` never extracts it;
    ``InlineScope(forceInline=true)`` folds it back at the end of the HIVM
    pipeline.
    """

    core_mode = "vector"


class simd(scope):
    """``with tla.scope.simd():`` -- SIMD vector-function region (vector core).

    Emitted as a manual vector function: the bishengir ``outline-scope`` pass
    extracts this marked scope into ``func.func`` + ``func.call``. See
    :meth:`mlir_attrs` for the role of each stamped attribute. SIMD scopes do
    not own tasks; nest them inside ``scope.vector().task(...)`` instead.
    """

    core_mode = "vector"

    @classmethod
    def mlir_attrs(cls, builder):
        attrs = super().mlir_attrs(builder)
        attrs.update({
            "no_inline": builder.get_unit_attr(),
            "outline": builder.get_unit_attr(),
            "vector_mode": builder.get_string_attr("vec"),
            "hivm.func_core_type": builder.parse_attr("#hivm.func_core_type<AIV>"),
            "hivm.vf_mode": builder.parse_attr("#hivm.vf_mode<SIMD>"),
            "hivm.vector_function": builder.get_unit_attr(),
        })
        return attrs


class cube(_TaskScopeKind):
    """``with tla.scope.cube():`` -- cube-core region.

    AIC scopes are not vector functions: only ``hivm.tcore_type<CUBE>`` is
    stamped, so outline-scope never extracts them into an AIV vector function.
    """

    core_mode = "cube"

    @classmethod
    def mlir_attrs(cls, builder):
        return {"hivm.tcore_type": builder.parse_attr("#hivm.tcore_type<CUBE>")}


# ``tla.scope`` resolves as the namespace holding the fixed-core-mode classes.
scope.vector = vector
scope.simd = simd
scope.cube = cube
