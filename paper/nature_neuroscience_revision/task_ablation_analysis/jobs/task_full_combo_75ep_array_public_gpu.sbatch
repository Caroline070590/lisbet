#!/bin/bash
#SBATCH --job-name=fullcombo75
#SBATCH --output=fullcombo75_%A_%a.out
#SBATCH --error=fullcombo75_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=public-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --array=0-74%5

set -euo pipefail

module purge
module load Anaconda3/2024.02-1

# Set DATA and OUTBASE before submission. BETMAN defaults to the executable on PATH.
: "${DATA:?Set DATA to the prepared LISBET training-data directory}"
: "${OUTBASE:?Set OUTBASE to the output directory}"
BETMAN="${BETMAN:-betman}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX="${MATRIX:-${SCRIPT_DIR}/task_full_combo_75ep_matrix.tsv}"

# +2 because line 1 is the header and SLURM_ARRAY_TASK_ID starts at 0
LINE_NO=$((SLURM_ARRAY_TASK_ID + 2))
LINE=$(sed -n "${LINE_NO}p" "${MATRIX}")

IFS=$'\t' read -r RUN_LABEL TASK_IDS SEED <<< "${LINE}"

RUN_ID="${RUN_LABEL}_seed${SEED}_75ep"
OUT="${OUTBASE}/${RUN_ID}"

echo "============================================================"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID}"
echo "Run label: ${RUN_LABEL}"
echo "Task ids: ${TASK_IDS}"
echo "Seed: ${SEED}"
echo "Epochs: 75"
echo "Data: ${DATA}"
echo "Output: ${OUT}"
echo "============================================================"

nvidia-smi

${BETMAN} train_model ${DATA} \
  --window_size 200 \
  --select_coords "experimental,stimuli;x,y;nose,earL,earR,neck,hipsL,hipsR,tail" \
  --rename_coords "experimental:resident,stimuli:intruder;x:x,y:y;earL:left_ear,earR:right_ear,hipsL:left_hip,hipsR:right_hip" \
  --output_path ${OUT} \
  --epochs 75 \
  --batch_size 32 \
  --task_ids ${TASK_IDS} \
  --seed ${SEED} \
  --run_id ${RUN_ID} \
  --learning_rate 0.0001 \
  --save_weights last \
  --save_history

echo "Finished ${RUN_ID} at $(date)"
