#!/usr/bin/env bash
set -e
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /data/Ascend/cann-9.2.1/set_env.sh
compile="$root/npuir-package/bin/bishengir-compile"
flags=(--target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode
       --custom-aic-number=28 --custom-aiv-number=56 --enable-optimized-metaop=true
       --enable-auto-bind-sub-block=false --enable-preload=false --enable-auto-multi-buffer=true
       --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-vf-operand-substitution=true
       --enable-auto-blockify-loop --enable-vf-merge-level=1)
for version in mla-tla mla-pipeline; do
    extra=()
    if [ "$version" = mla-tla ]; then
        extra=(--disable-hivm-auto-inject-sync --enable-hivm-unit-flag-sync=false
               --enable-hivm-graph-sync-solver=true)
    fi
    for case in full ragged; do
        dir="$root/mla-temp/offline-upstream/$version/$case"
        "$compile" "$dir/01_linalg.mlir" "${flags[@]}" "${extra[@]}" \
            --save-temps="$dir/device-ir" -o "$dir/kernel.o" > "$dir/device.log" 2>&1
        test -s "$dir/kernel.o"
        printf 'ELF_PASS %s %s\n' "$version" "$case"
    done
done
affinity="$root/npuir-source/bishengir/test/bishengir-compile/affinity/tla-stages.mlir"
work="$root/mla-temp/affinity-upstream"
mkdir -p "$work"
"$compile" "$affinity" --target=Ascend950PR_9579 --enable-triton-kernel-compile=true \
    --enable-tla-mode --save-temps="$work" -o "$work/affinity.o" > "$work/device.log" 2>&1
test -s "$work/affinity.o"
"$root/npuir-build/bin/FileCheck" "$affinity" --input-file="$work/module.hivm.opt.mlir" \
    --implicit-check-not=hivm.pipeline --implicit-check-not=hivm.preload_pipeline --implicit-check-not=scope.scope
printf 'AFFINITY_ELF_PASS\n'
