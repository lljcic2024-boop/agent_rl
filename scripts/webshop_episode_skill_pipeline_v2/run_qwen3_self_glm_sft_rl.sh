#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONDA_ENV_WEBSHOP="${CONDA_ENV_WEBSHOP:-sgop-webshop}"
CONDA_ENV_SFT="${CONDA_ENV_SFT:-copd}"

cd "$PROJECT_ROOT"

echo "[webshop-qwen3-self] Step 1/3: generate Qwen3 self baseline rollouts and GLM summary+skill data."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_WEBSHOP"
bash "$SCRIPT_DIR/run_qwen3_self_glm.sh" "$@"

echo "[webshop-qwen3-self] Step 2/3: SFT Qwen3 on self-generated WebShop data."
conda activate "$CONDA_ENV_SFT"
bash "$SCRIPT_DIR/run_sft_qwen3_self_glm.sh"

echo "[webshop-qwen3-self] Step 3/3: start WebShop RL from self SFT checkpoint."
conda activate "$CONDA_ENV_WEBSHOP"
bash "$PROJECT_ROOT/examples/seed/run_webshop_qwen3_episode_no_skill_loss_sft_self.sh"
