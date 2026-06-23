#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  if [[ -f "$CONDA_SH" ]]; then
    set +u
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
    set -u
  else
    echo "CONDA_SH not found: $CONDA_SH" >&2
    exit 1
  fi
fi

export GMAI_VL_MODEL_PATH="${GMAI_VL_MODEL_PATH:-$SCRIPT_DIR/model_weight}"
export PORT="${PORT:-10083}"
export GRADIO_ANALYTICS_ENABLED=False
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-$SCRIPT_DIR/gradio_tmp}"
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""
export all_proxy=""
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"

cd "$SCRIPT_DIR"
python app.py
