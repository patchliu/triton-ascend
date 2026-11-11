"""Development-only extraction of the two already validated schedules."""

import ast
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'mla-temp/original'


class Schedule(ast.NodeTransformer):
    def __init__(self, fixed):
        self.fixed = dict(fixed)

    def value(self, node):
        expression = copy.deepcopy(node)
        for name in ast.walk(expression):
            if isinstance(name, ast.Name) and name.id not in self.fixed and name.id not in ('min', 'max'):
                raise ValueError(name.id)
        return eval(compile(ast.Expression(expression), '<constexpr>', 'eval'),
                    {'__builtins__': {}, 'min': min, 'max': max}, self.fixed)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.fixed:
            return ast.copy_location(ast.parse(repr(self.fixed[node.id]), mode='eval').body, node)
        return node

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.value is not None:
            try:
                value = self.value(node.value)
            except (ValueError, TypeError, NameError):
                return self.generic_visit(node)
            if isinstance(value, (bool, str)) or node.target.id in self.fixed:
                self.fixed[node.target.id] = value
                return None
        return self.generic_visit(node)

    def visit_If(self, node):
        node.test = self.visit(node.test)
        try:
            chosen = node.body if self.value(node.test) else node.orelse
        except (ValueError, TypeError, NameError):
            return self.generic_visit(node)
        result = []
        for statement in chosen:
            rewritten = self.visit(statement)
            if isinstance(rewritten, list):
                result.extend(rewritten)
            elif rewritten is not None:
                result.append(rewritten)
        return result

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        try:
            return node.body if self.value(node.test) else node.orelse
        except (ValueError, TypeError, NameError):
            return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        try:
            value = self.value(node)
            return ast.copy_location(ast.parse(repr(value), mode='eval').body, node)
        except (ValueError, TypeError, NameError):
            return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        try:
            value = self.value(node)
            if isinstance(value, int):
                return ast.copy_location(ast.Constant(value), node)
        except (ValueError, TypeError, NameError):
            pass
        if isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and node.left.value in (0, 1):
                return node.right if node.left.value == 1 else node.left
            if isinstance(node.right, ast.Constant) and node.right.value in (0, 1):
                return node.left if node.right.value == 1 else node.right
        if isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                return node.left
        if isinstance(node.op, ast.FloorDiv) and isinstance(node.right, ast.Constant) and node.right.value == 1:
            return node.left
        return node

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'static_assert' and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value is True):
            return None
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                and isinstance(node.iter.func.value, ast.Name)
                and node.iter.func.value.id == 'tl' and node.iter.func.attr == 'static_range'):
            try:
                values = range(*(self.value(a) for a in node.iter.args))
            except (ValueError, TypeError, NameError):
                return node
            if len(values) == 1:
                replacement = Schedule({node.target.id: values[0]})
                return [replacement.visit(s) for s in node.body]
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        values = []
        for value in node.values:
            try:
                constant = self.value(value)
            except (ValueError, TypeError, NameError):
                values.append(value)
                continue
            if isinstance(node.op, ast.And) and not constant:
                return ast.copy_location(ast.Constant(False), node)
            if isinstance(node.op, ast.Or) and constant:
                return ast.copy_location(ast.Constant(True), node)
        if len(values) == 1:
            return values[0]
        if not values:
            return ast.copy_location(ast.Constant(isinstance(node.op, ast.And)), node)
        node.values = values
        return node


RENAMES = dict(B='BATCH', SQ='QUERY_LENGTH', H='QUERY_HEADS', TABLE_STRIDE='MAX_PAGES',
               CORES='CUBE_CORES', HEAD_TILE='Q_TILE_ROWS', ROWS='VECTOR_ROWS', QUERY_ROWS='CAUSAL_ROWS',
               KV_TILE='KEY_ROWS', WIDTH='QK_DIM', LATENT='VALUE_DIM', ROPE='ROPE_DIM',
               TABLE='BLOCK_TABLE', LENGTHS='KV_LENGTHS', QR='UNUSED_Q_ROPE', KR='UNUSED_K_ROPE')


class Names(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = RENAMES.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = RENAMES.get(node.arg, node.arg)
        return self.generic_visit(node)


def kernel(filename, function, fixed, arguments):
    tree = ast.parse((SOURCE / filename).read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    result = copy.deepcopy(original)
    for _ in range(3):
        result = Schedule(fixed).visit(result)
    for _ in range(4):
        reads = {n.id for n in ast.walk(result) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        class Unused(ast.NodeTransformer):
            def visit_AnnAssign(self, node):
                if isinstance(node.target, ast.Name) and node.target.id not in reads:
                    return None
                return node
        result = Unused().visit(result)
    result.args.args = [a for a in result.args.args if a.arg not in arguments]
    result.args.defaults = []
    result.name = 'attention' if 'explicit' in function else 'attention_splits'
    result = Names().visit(result)
    ast.fix_missing_locations(result)
    return ast.unparse(result)


def main():
    fixed = dict(Q_GROUP=1, WIDE_QK=False, SPLIT_PV=False, PACKED=True,
                 SKIP_FULL_MASK=True, KV_MAJOR=True)
    removed = set(fixed) | {'STAGES', 'QR', 'KR'}
    tla = kernel('mla_decode_explicit.py', 'mla_split_explicit',
                 dict(fixed, L0_MODE='native112_merge_dynflags_stream_k'), removed | {'L0_MODE'})
    pipeline = kernel('mla_decode.py', 'mla_split',
                      dict(fixed, KV_TILE=128, STAGES=(0, 0, 1), PARTS=1,
                           VALUE_PARTS=1, PV_TASKS=1), removed | {'KV_TILE'})
    for directory, body in (('mla-tla', tla), ('mla-pipeline', pipeline)):
        target = ROOT / 'mla-temp/generated' / directory
        target.mkdir(parents=True, exist_ok=True)
        (target / 'mla.py').write_text('"""BF16 paged FlashMLA, with FP32 online state and LSE."""\n\n'
                                      'import math\n\nimport triton\nimport triton.language as tl\n'
                                      'import triton.language.extra.tla as tla\n\n\n' + body + '\n')


if __name__ == '__main__':
    main()
