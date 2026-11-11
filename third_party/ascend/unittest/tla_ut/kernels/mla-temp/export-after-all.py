"""Replay verified targets and capture every AscendNPU-IR pass output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def export(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('00_ast.mlir', '01_symbol_dce.mlir', '01_linalg.mlir', 'tiling.json'):
        shutil.copy2(source / name, destination / name)
    original = json.loads((source / 'compiler-argv.json').read_text())
    command = [original[0], str(destination / '01_linalg.mlir')]
    command.extend(arg for arg in original[2:-2] if not arg.startswith('--save-temps='))
    command.extend(('--mlir-print-ir-after-all', '--mlir-print-ir-module-scope',
                    '--mlir-disable-threading', '--save-temps=' + str(destination / 'device-ir'),
                    '-o', str(destination / 'kernel.o')))
    (destination / 'compiler-argv.json').write_text(json.dumps(command, indent=2) + '\n')
    dump = destination / 'ir-after-all.mlir'
    with dump.open('w') as stderr, (destination / 'compiler-stdout.log').open('w') as stdout:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    headings = []
    with dump.open() as stream:
        for line in stream:
            if 'IR Dump After' in line:
                headings.append(line.strip())
    (destination / 'pass-index.txt').write_text('\n'.join(headings) + '\n')
    record = dict(exit_code=result.returncode, dumps=len(headings), dump_bytes=dump.stat().st_size,
                  source_sha256=hashlib.sha256((destination/'01_linalg.mlir').read_bytes()).hexdigest())
    if result.returncode == 0:
        assert len(headings) > 20, 'Expected pass dumps across the full compilation'
        elf = (destination / 'kernel.o').read_bytes()
        assert elf.startswith(b'\x7fELF')
        record['elf_sha256'] = hashlib.sha256(elf).hexdigest()
        record['distinct_dump_headings'] = len(set(headings))
    (destination / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(destination, json.dumps(record), flush=True)
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    for version in ('mla-tla', 'mla-pipeline'):
        for case in ('full', 'ragged'):
            export(args.input/version/case, args.output/version/case)


if __name__ == '__main__':
    main()
