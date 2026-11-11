#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python="$HOME/miniconda3/envs/dev/bin/python"
output="$root/final-delivery/toolchain"
mkdir -p "$output/wheels" "$root/final-wheel-unpack"
original=("$root"/package/staging-wheels/*.whl)
test "${#original[@]}" -eq 1
"$python" -m wheel unpack --dest "$root/final-wheel-unpack" "${original[0]}"
tree=("$root"/final-wheel-unpack/triton_ascend-*)
test "${#tree[@]}" -eq 1
embedded="${tree[0]}/triton/backends/ascend/bishengir"
mv "$embedded" "$root/final-wheel-unpack/bishengir-before-sync-fix"
mkdir -p "$embedded"
cp -a "$root/npuir-package/bin" "$root/npuir-package/lib" "$embedded/"
"$python" -m wheel pack --build-number 1 --dest-dir "$output/wheels" "${tree[0]}"

cp -a "$root/npuir-package" "$output/npuir"
cp "$root/npuir-source/LICENSE" "$output/npuir/" 2>/dev/null || true
{
    printf '80514fbe68ce52af666f642be2194ff9a2762e79 frontend build\n'
    printf '008dff39bdcc908d6706946189b0c53c5257e4d4 published frontend, identical compiler source\n'
    printf '10c0b8183 NPU-IR source, including equal preload index fix\n'
} > "$output/npuir/build-info/final-revisions.txt"
(cd "$output/npuir"; find bin lib build-info -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

final=("$output"/wheels/*.whl)
test "${#final[@]}" -eq 1
"$python" -m pip install --no-deps --target "$root/python-final" "${final[0]}"
installed="$root/python-final/triton/backends/ascend/bishengir"
for name in bishengir-compile bishengir-opt hivmc; do
    cmp "$installed/bin/$name" "$output/npuir/bin/$name"
    test -x "$installed/bin/$name"
done
for template in "$output"/npuir/lib/*.bc; do
    cmp "$template" "$installed/lib/$(basename -- "$template")"
done
(cd "$output"; sha256sum wheels/*.whl npuir/bin/bishengir-compile npuir/bin/bishengir-opt npuir/bin/hivmc > SHA256SUMS)
cat "$output/SHA256SUMS"
