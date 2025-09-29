#!/usr/bin/env bash
set -euo pipefail
: "${CONDA_PREFIX:?activate your 'nlp' env first}"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/llamacpp_cuda12.sh" <<'SH'
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
SH

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/llamacpp_cuda12.sh" <<'SH'
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
unset _OLD_LD_LIBRARY_PATH
SH

echo "Hook installed. Deactivate/activate your env to apply."