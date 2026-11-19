#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

device=${DEVICE:-0}

depth=(9)
n_ctx=(12)
t_n_ctx=(4)

EXP_TAG=${EXP_TAG:-pace_clip}
TEST_EPOCH=${TEST_EPOCH:-5}
PYTHON_FILE=${PYTHON_FILE:-test.py}

# ==============================
# ==============================
DATA_PATH_BTAD=${DATA_PATH_BTAD:-./data/BTAD/BTech_Dataset_transformed}
DATA_PATH_MPDD=${DATA_PATH_MPDD:-./data/MPDD}

IMAGE_SIZE=${IMAGE_SIZE:-518}
GAMMA_CA=${GAMMA_CA:-0.1}
CPG_TOPK_RATIO=${CPG_TOPK_RATIO:-0.05}
SIGMA=${SIGMA:-4}
PATCH_ROUTER_TAU=${PATCH_ROUTER_TAU:-0.1}
PIXEL_PATCH_RATIO=${PIXEL_PATCH_RATIO:-0.30}
IMAGE_PATCH_RATIO=${IMAGE_PATCH_RATIO:-0.10}
FUSION_FLOOR_RATIO=${FUSION_FLOOR_RATIO:-0.10}
FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}

LAMBDA_ROUTER_MVTEC2BTAD=${LAMBDA_ROUTER_MVTEC2BTAD:-0.7}
LAMBDA_ROUTER_MVTEC2MPDD=${LAMBDA_ROUTER_MVTEC2MPDD:-0.7}

TOPK_RATIO_BTAD=${TOPK_RATIO_BTAD:-0.001}
TOPK_RATIO_MPDD=${TOPK_RATIO_MPDD:-0.001}

FEATURES_LIST="6 12 18 24"
FEATURE_MAP_LAYER="0 1 2 3"

echo "=========================================="
echo "PACE-CLIP Test: MVTec-trained checkpoint -> BTAD / MPDD"
echo "EXP_TAG = ${EXP_TAG}"
echo "TEST_EPOCH = ${TEST_EPOCH}"
echo "PYTHON_FILE = ${PYTHON_FILE}"
echo "BTAD path = ${DATA_PATH_BTAD}"
echo "MPDD path = ${DATA_PATH_MPDD}"
echo "BTAD topk = ${TOPK_RATIO_BTAD}"
echo "MPDD topk = ${TOPK_RATIO_MPDD}"
echo "=========================================="

# ============================================================
# 1. MVTec -> BTAD
# ============================================================
for i in "${!depth[@]}"; do
  for j in "${!n_ctx[@]}"; do

    train_base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2Visa
    ckpt_dir=./checkpoints/${train_base_dir}/

    result_dir=./results/${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2BTAD/epoch_${TEST_EPOCH}_pace_clip_topk001
    mkdir -p ${result_dir}

    ckpt_path=${ckpt_dir}/epoch_${TEST_EPOCH}.pth
    if [ ! -f "${ckpt_path}" ]; then
      echo "Error: checkpoint not found: ${ckpt_path}"
      echo "Please check whether your MVTec-trained checkpoint directory is correct."
      exit 1
    fi

    LOG=${result_dir}/res.log
    echo "=========================================="
    echo "Testing MVTec -> BTAD"
    echo "Checkpoint: ${ckpt_path}"
    echo "Log file: ${LOG}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${device} python ${PYTHON_FILE} \
      --dataset btad \
      --data_path ${DATA_PATH_BTAD} \
      --save_path ${result_dir} \
      --checkpoint_path ${ckpt_path} \
      --features_list ${FEATURES_LIST} \
      --feature_map_layer ${FEATURE_MAP_LAYER} \
      --image_size ${IMAGE_SIZE} \
      --depth ${depth[i]} \
      --n_ctx ${n_ctx[j]} \
      --t_n_ctx ${t_n_ctx[0]} \
      --metrics image-pixel-level \
      --sigma ${SIGMA} \
      --gamma_ca ${GAMMA_CA} \
      --lambda_router ${LAMBDA_ROUTER_MVTEC2BTAD} \
      --topk_ratio ${TOPK_RATIO_BTAD} \
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


# ============================================================
# 2. MVTec -> MPDD
# ============================================================
for i in "${!depth[@]}"; do
  for j in "${!n_ctx[@]}"; do

    train_base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2Visa
    ckpt_dir=./checkpoints/${train_base_dir}/

    result_dir=./results/${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2MPDD/epoch_${TEST_EPOCH}_pace_clip_topk001
    mkdir -p ${result_dir}

    ckpt_path=${ckpt_dir}/epoch_${TEST_EPOCH}.pth
    if [ ! -f "${ckpt_path}" ]; then
      echo "Error: checkpoint not found: ${ckpt_path}"
      echo "Please check whether your MVTec-trained checkpoint directory is correct."
      exit 1
    fi

    LOG=${result_dir}/res.log
    echo "=========================================="
    echo "Testing MVTec -> MPDD"
    echo "Checkpoint: ${ckpt_path}"
    echo "Log file: ${LOG}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${device} python ${PYTHON_FILE} \
      --dataset mpdd \
      --data_path ${DATA_PATH_MPDD} \
      --save_path ${result_dir} \
      --checkpoint_path ${ckpt_path} \
      --features_list ${FEATURES_LIST} \
      --feature_map_layer ${FEATURE_MAP_LAYER} \
      --image_size ${IMAGE_SIZE} \
      --depth ${depth[i]} \
      --n_ctx ${n_ctx[j]} \
      --t_n_ctx ${t_n_ctx[0]} \
      --metrics image-pixel-level \
      --sigma ${SIGMA} \
      --gamma_ca ${GAMMA_CA} \
      --lambda_router ${LAMBDA_ROUTER_MVTEC2MPDD} \
      --topk_ratio ${TOPK_RATIO_MPDD} \
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
