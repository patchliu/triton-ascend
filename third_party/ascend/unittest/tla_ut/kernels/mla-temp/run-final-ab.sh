#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
base="$HOME/workspace/npuir/temp"
native="$base/20261110-mla-ops-transformer.cXaT4q"
dsl="$base/20261110-cannbot-flash-mla.L50di4"
set +u
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source "$native/vendor/vendors/custom_transformer/bin/set_env.bash"
set -u
export PYTHONPATH="$dsl/python:$native/python:$base/20261111-flashmla-l0.iggOBZ/python-bf16-dynflags-o2:$base/20261111-flashmla-l0.iggOBZ/pipeline112-v3.p4BgGE/tla_ut"
export PATH="$native/build/python/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$native/torch-extensions-cann91" MAX_JOBS=4
export TLA_MLA_PROBE_O2=1 TLA_TEST_STAGE=run PYTHONDONTWRITEBYTECODE=1
export CANNBOTDSL_BISHENG_PATH="$base/20261110-mla-cannbot-bench.uvCRab/bisheng-cann92"
export TRITON_CACHE_DIR="$root/mla-temp/cache"
python="$HOME/.conda/envs/tla-lpc/bin/python"
mkdir -p "$root/mla-temp/logs"

trap 'code=$?; printf "%s\n" "$code" > "$root/mla-temp/logs/ab.exit"' EXIT
while [ ! -f "$root/mla-temp/logs/run-r2.exit" ]; do sleep 8; done
while :; do
    card=
    for candidate in 2 1; do
        snap=$(npu-smi info -t proc-mem -i "$candidate" 2>&1) || continue
        if ! grep -q 'Process id:' <<<"$snap"; then card=$candidate; break; fi
    done
    if [ -n "$card" ]; then break; fi
    sleep 8
done
export ASCEND_RT_VISIBLE_DEVICES="$card"
cd "$root"
"$python" -u mla-temp/compare-before-after.py before-after > mla-temp/logs/ab.log 2>&1
"$python" -m pytest -q mla-tla/test_mla_tla.py mla-pipeline/test_mla_pipeline.py -k prepared > mla-temp/logs/mutation.log 2>&1
