#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

device=${DEVICE:-0}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

# ==============================
# ==============================
EXP_TAG=${EXP_TAG:-pace_clip}
TEST_EPOCH=${TEST_EPOCH:-5}
PYTHON_FILE=${PYTHON_FILE:-test.py}

# ==============================
# ==============================
DATA_PATH_DTD=${DATA_PATH_DTD:-./data/DTD}

# ==============================
# ==============================
DATASET_NAME_DTD=${DATASET_NAME_DTD:-dtd-synthetic}

# ==============================
# ==============================
IMAGE_SIZE=${IMAGE_SIZE:-518}
GAMMA_CA=${GAMMA_CA:-0.1}
CPG_TOPK_RATIO=${CPG_TOPK_RATIO:-0.05}
SIGMA=${SIGMA:-4}
PATCH_ROUTER_TAU=${PATCH_ROUTER_TAU:-0.1}
PIXEL_PATCH_RATIO=${PIXEL_PATCH_RATIO:-0.30}
IMAGE_PATCH_RATIO=${IMAGE_PATCH_RATIO:-0.10}
FUSION_FLOOR_RATIO=${FUSION_FLOOR_RATIO:-0.10}
FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}

LAMBDA_ROUTER_MVTEC2DTD=${LAMBDA_ROUTER_MVTEC2DTD:-0.7}

TOPK_RATIO_DTD=${TOPK_RATIO_DTD:-0.001}

FEATURES_LIST="6 12 18 24"
FEATURE_MAP_LAYER="0 1 2 3"

echo "=========================================="
echo "PACE-CLIP Test: MVTec-trained checkpoint -> DTD-Synthetic"
echo "EXP_TAG = ${EXP_TAG}"
echo "TEST_EPOCH = ${TEST_EPOCH}"
echo "PYTHON_FILE = ${PYTHON_FILE}"
echo "=========================================="

# ============================================================
#   5: topk_ratio       top-k ratio
# ============================================================
run_test_one_dataset() {
  target_tag=$1
  dataset_name=$2
  data_path=$3
  lambda_router=$4
  topk_ratio=$5
  metrics_mode=${6:-image-pixel-level}

  for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
      # ======================================================
      # ======================================================
      train_base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2Visa
      ckpt_dir=./checkpoints/${train_base_dir}/
      ckpt_path=${ckpt_dir}/epoch_${TEST_EPOCH}.pth

      if [ ! -f "${ckpt_path}" ]; then
        echo "Error: checkpoint not found: ${ckpt_path}"
        echo "Please check your MVTec-trained checkpoint directory."
        exit 1
      fi

      result_dir=./results/${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2${target_tag}/epoch_${TEST_EPOCH}_pace_clip_topk${topk_ratio}
      mkdir -p ${result_dir}
      LOG=${result_dir}/res.log

      echo "=========================================="
      echo "Testing MVTec -> ${target_tag}"
      echo "Dataset arg: ${dataset_name}"
      echo "Data path: ${data_path}"
      echo "Checkpoint: ${ckpt_path}"
      echo "Lambda router: ${lambda_router}"
      echo "Top-k ratio: ${topk_ratio}"
      echo "Log file: ${LOG}"
      echo "=========================================="

      CUDA_VISIBLE_DEVICES=${device} python ${PYTHON_FILE} \
        --dataset ${dataset_name} \
        --data_path ${data_path} \
        --save_path ${result_dir} \
        --checkpoint_path ${ckpt_path} \
        --features_list ${FEATURES_LIST} \
        --feature_map_layer ${FEATURE_MAP_LAYER} \
        --image_size ${IMAGE_SIZE} \
        --depth ${depth[i]} \
        --n_ctx ${n_ctx[j]} \
        --t_n_ctx ${t_n_ctx[0]} \
        --metrics ${metrics_mode} \
        --sigma ${SIGMA} \
        --gamma_ca ${GAMMA_CA} \
        --lambda_router ${lambda_router} \
        --topk_ratio ${topk_ratio} \
        --cpg_topk_ratio ${CPG_TOPK_RATIO} \
        --patch_router_tau ${PATCH_ROUTER_TAU} \
        --pixel_patch_ratio ${PIXEL_PATCH_RATIO} \
        --image_patch_ratio ${IMAGE_PATCH_RATIO} \
        --fusion_floor_ratio ${FUSION_FLOOR_RATIO} \
        --fusion_temperature ${FUSION_TEMPERATURE} \
        2>&1 | tee ${LOG}
      wait
    done
  done
}

# ============================================================
# MVTec -> DTD-Synthetic
# ============================================================
run_test_one_dataset \
  "DTDSynthetic" \
  "${DATASET_NAME_DTD}" \
  "${DATA_PATH_DTD}" \
  "${LAMBDA_ROUTER_MVTEC2DTD}" \
  "${TOPK_RATIO_DTD}"

echo "=========================================="
echo "DTD test finished."
echo "Results saved to: ./results/9_12_4_${EXP_TAG}_MVTec2DTDSynthetic/"
echo "=========================================="
