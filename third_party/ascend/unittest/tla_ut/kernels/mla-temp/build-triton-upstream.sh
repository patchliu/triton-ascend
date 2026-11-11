#!/usr/bin/env bash
set -euo pipefail
umask 077
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
trap 'printf "%s\n" "$?" > "$root/mla-temp/triton-build.exit"' EXIT
src="$root/triton-source"
npu="$root/npuir-source"
previous="$HOME/workspace/npuir/temp/20261109-tla-ops-paged-gqa-43df"
dev="$HOME/miniconda3/envs/dev/bin"
cann=/data/Ascend/cann-9.2.1
set +u
source "$cann/set_env.sh"
set -u
export PATH="$dev:$PATH"
export TRITON_ASCEND_BISHENGIR_PATH="$root/npuir-package"
export TRITON_BUILD_NPUIR=OFF TRITON_OFFLINE_BUILD=1 MAX_JOBS=24
export TRITON_BUILD_DIR="$root/build/triton"
export LLVM_SYSPATH="$HOME/.triton/llvm/llvm-f6ded0be-3f5016d1-ubuntu-x64"
cd "$src"
sub="$src/third_party/ascend/AscendNPU-IR"
test -d "$sub"
rmdir "$sub"
ln -s "$npu" "$sub"
git log -2 --format='%H %s'
printf 'Dialect source %s\n' "$(git -C "$npu" rev-parse HEAD)"
mkdir -p "$root/package/staging-wheels"
"$dev/python" -m pip wheel -v . --no-build-isolation --no-deps -w "$root/package/staging-wheels"
ls -lh "$root/package/staging-wheels"
echo TRITON_STAGING_WHEEL_PASS
