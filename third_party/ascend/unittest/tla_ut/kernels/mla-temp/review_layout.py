"""Development-only presentation changes after schedule extraction."""
import ast
from pathlib import Path

from extract_final import Schedule

ROOT = Path(__file__).resolve().parent.parent


class Readable(ast.NodeTransformer):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.aliases = {
            'GROUP_ROWS': 'Q_TILE_ROWS', 'TILE_ROWS': 'Q_TILE_ROWS',
            'STATE_ROWS': 'VECTOR_ROWS', 'VALUE_TILE': 'VALUE_DIM',
            'q_lat': 'query', 'kv_lat': 'key_value',
            'max_state': 'maximum_state', 'sum_state': 'denominator_state',
        } if pipeline else {}

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in self.aliases:
            return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        node.id = self.aliases.get(node.id, node.id)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if not self.pipeline and isinstance(node.test, ast.Name) and node.test.id in ('FUSE_MERGE', 'MERGE_ARENA'):
            transform = Schedule(dict(FUSE_MERGE=True, MERGE_ARENA=True, STREAM_MERGE=True))
            node.body = [transform.visit(statement) for statement in node.body]
            node.body = [statement for part in node.body for statement in (part if isinstance(part, list) else [part])
                         if statement is not None]
        return node


for directory in ('mla-tla', 'mla-pipeline'):
    file = ROOT / 'mla-temp/generated' / directory / 'mla.py'
    tree = Readable(directory == 'mla-pipeline').visit(ast.parse(file.read_text()))
    tree = Schedule(dict(load_query=True)).visit(tree)
    class Remove(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'load_query':
                return None
            return node
    tree = Remove().visit(tree)
    ast.fix_missing_locations(tree)
    file.write_text(ast.unparse(tree) + '\n')
