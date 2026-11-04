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
Ascend-specific code generation handlers for 'with' statement context managers.
"""

__all__ = ["handle_scope_with"]


def handle_scope_with(generator, node):
    """Emit a plain scope, task group, or logical scheduling domain."""
    from triton.compiler.code_generator import enter_sub_region
    from triton.language.core import _unwrap_if_constexpr
    from .scope import _TaskScope, schedule

    if node.items[0].optional_vars is not None:
        raise ValueError("TLA scopes do not support 'as' bindings")
    spec = _unwrap_if_constexpr(generator.visit(node.items[0].context_expr))
    domains = getattr(generator, "_tla_schedule_domains", [])
    is_domain = isinstance(spec, schedule)
    if is_domain:
        if domains:
            raise ValueError("nested schedule domains are not supported yet")
        domains = domains + [set()]
    elif isinstance(spec, _TaskScope):
        if not domains:
            raise ValueError("task() requires an enclosing tla.scope.schedule('auto')")
        if getattr(generator, "_tla_in_task", False):
            raise ValueError("nested task annotations are not supported")
        # The Triton loop carry analysis may visit one AST node repeatedly.
        task_key = (spec.name, node.lineno, node.col_offset)
        if any(name == spec.name and (line, col) != (node.lineno, node.col_offset) for name, line, col in domains[-1]):
            raise ValueError(f"duplicate task name in schedule domain: {spec.name}")
        domains[-1].add(task_key)
    old_domains = getattr(generator, "_tla_schedule_domains", [])
    old_task = getattr(generator, "_tla_in_task", False)
    generator._tla_schedule_domains = domains
    generator._tla_in_task = old_task or isinstance(spec, _TaskScope)
    try:
        with enter_sub_region(generator):
            attrs = spec.mlir_attrs(generator.builder)
            scope_op = generator.builder.create_scope_op(attrs, [])
            body = generator.builder.get_scope_entry_block(scope_op)
            generator.builder.set_insertion_point_to_start(body)
            generator.visit_compound_statement(node.body)
            generator.builder.set_insertion_point_to_end(body)
            if not body.has_terminator():
                generator.builder.scope_return([])
    finally:
        generator._tla_schedule_domains = old_domains
        generator._tla_in_task = old_task
    return None
