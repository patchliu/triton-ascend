#!/usr/bin/env bash
set -euo pipefail
umask 077
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
trap 'printf "%s\n" "$?" > "$root/mla-temp/npuir-build-r2.exit"' EXIT
src="$root/npuir-source"
llvm="$HOME/workspace/AscendNPU-IR/third-party/llvm-project"
build="$root/npuir-build"
prefix="$root/npuir-package"
dev="$HOME/miniconda3/envs/dev/bin"
cann=/data/Ascend/cann-9.2.1
set +u
source "$cann/set_env.sh"
set -u
export PATH="$dev:$PATH"
export BISHENG_INSTALL_PATH="$cann/bin"
expected=$(git -C "$src" rev-parse HEAD:third-party/llvm-project)
actual=$(git -C "$llvm" rev-parse HEAD)
test "$expected" = "$actual"
git -C "$src" log -2 --format='%H %s'
printf 'LLVM=%s\n' "$actual"
test -z "$(git -C "$llvm" status --porcelain)"
test -x "$BISHENG_INSTALL_PATH/bisheng"
mkdir -p "$build" "$prefix"
cmake -S "$llvm/llvm" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_MAKE_PROGRAM="$dev/ninja" \
  -DCMAKE_C_COMPILER_LAUNCHER="$dev/ccache" -DCMAKE_CXX_COMPILER_LAUNCHER="$dev/ccache" \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_EXTERNAL_PROJECTS=bishengir \
  -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$src" -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_LLD=ON -DLLVM_PARALLEL_LINK_JOBS=2 -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF \
  -DBSPUB_DAVINCI_BISHENGIR=ON -DLLVM_BSPUB_DAVINCI=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON -DBISHENGIR_ENABLE_PM_CL_OPTIONS=ON -DBISHENGIR_ENABLE_TRITON_COMPILE=ON -DBISHENGIR_PUBLISH=ON -DBISHENGIR_BUILD_HIVMC=ON -DBISHENGIR_BUILD_TEMPLATE=ON \
  -DBISHENGIR_BUILD_TEMPLATE_ASCEND960=OFF \
  -DBISHENG_COMPILER_PATH="$BISHENG_INSTALL_PATH" -DBISHENG_COMPILER_EXECUTABLE="$cann/bin/bisheng" \
  -DBISHENG_LLC_INCLUDE_DIR=/data/Ascend/cann-9.3.0/include
cmake --build "$build" --parallel 40 --target bishengir-compile bishengir-opt hivmc FileCheck
cmake --build "$build" --parallel 12 --target install-bishengir-publish-products
test -x "$prefix/bin/bishengir-compile"
test -x "$prefix/bin/bishengir-opt"
test -x "$prefix/bin/hivmc"
test -s "$prefix/lib/meta_op.aic.c310.O2.bc"
mkdir -p "$prefix/build-info"
cp "$build/CMakeCache.txt" "$prefix/build-info/"
git -C "$src" log -2 --format='%H %s' > "$prefix/build-info/npuir-revisions.txt"
printf '%s\n' "$actual" > "$prefix/build-info/llvm-revision.txt"
"$prefix/bin/bishengir-compile" --version | tee "$prefix/build-info/version.txt"
(cd "$prefix"; find bin lib build-info -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo NPUIR_BUILD_PASS
