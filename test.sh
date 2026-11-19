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

DATA_PATH_VISA=${DATA_PATH_VISA:-./data/visa}
DATA_PATH_MVTEC=${DATA_PATH_MVTEC:-./data/mvtec}

IMAGE_SIZE=${IMAGE_SIZE:-518}
GAMMA_CA=${GAMMA_CA:-0.1}
CPG_TOPK_RATIO=${CPG_TOPK_RATIO:-0.05}
SIGMA=${SIGMA:-4}
PATCH_ROUTER_TAU=${PATCH_ROUTER_TAU:-0.1}
PIXEL_PATCH_RATIO=${PIXEL_PATCH_RATIO:-0.30}
IMAGE_PATCH_RATIO=${IMAGE_PATCH_RATIO:-0.10}
FUSION_FLOOR_RATIO=${FUSION_FLOOR_RATIO:-0.10}
FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}

LAMBDA_ROUTER_VISA2MVTEC=${LAMBDA_ROUTER_VISA2MVTEC:-0.9}
LAMBDA_ROUTER_MVTEC2VISA=${LAMBDA_ROUTER_MVTEC2VISA:-0.7}

TOPK_RATIO_MVTEC=${TOPK_RATIO_MVTEC:-0.001}
TOPK_RATIO_VISA=${TOPK_RATIO_VISA:-0.001}

FEATURES_LIST="6 12 18 24"
FEATURE_MAP_LAYER="0 1 2 3"

echo "=========================================="
echo "PACE-CLIP Test: Pixel-Anchored Dual-Granularity Evidence Fusion"
echo "EXP_TAG = ${EXP_TAG}"
echo "TEST_EPOCH = ${TEST_EPOCH}"
echo "PYTHON_FILE = ${PYTHON_FILE}"
echo "PIXEL/IMAGE PATCH_RATIO = ${PIXEL_PATCH_RATIO}/${IMAGE_PATCH_RATIO} (image ratio kept but local maps use pixel detach)"
echo "MVTec test topk = ${TOPK_RATIO_MVTEC}"
echo "VisA test topk = ${TOPK_RATIO_VISA}"
echo "=========================================="

# ============================================================
# 1. MVTec -> VisA
# ============================================================
for i in "${!depth[@]}"; do
  for j in "${!n_ctx[@]}"; do
    base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2Visa
    ckpt_dir=./checkpoints/${base_dir}/
    result_dir=./results/${base_dir}/epoch_${TEST_EPOCH}_pace_clip_topk001
    mkdir -p ${result_dir}

    ckpt_path=${ckpt_dir}/epoch_${TEST_EPOCH}.pth
    if [ ! -f "${ckpt_path}" ]; then
      echo "Error: checkpoint not found: ${ckpt_path}"
      exit 1
    fi

    LOG=${result_dir}/res.log
    echo "Log file: ${LOG}"
    CUDA_VISIBLE_DEVICES=${device} python ${PYTHON_FILE} \
      --dataset visa \
      --data_path ${DATA_PATH_VISA} \
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
      --lambda_router ${LAMBDA_ROUTER_MVTEC2VISA} \
      --topk_ratio ${TOPK_RATIO_VISA} \
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
# 2. VisA -> MVTec
# ============================================================
for i in "${!depth[@]}"; do
  for j in "${!n_ctx[@]}"; do
    base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_Visa2MVTec
    ckpt_dir=./checkpoints/${base_dir}/
    result_dir=./results/${base_dir}/epoch_${TEST_EPOCH}_pace_clip_topk001
    mkdir -p ${result_dir}

    ckpt_path=${ckpt_dir}/epoch_${TEST_EPOCH}.pth
    if [ ! -f "${ckpt_path}" ]; then
      echo "Error: checkpoint not found: ${ckpt_path}"
      exit 1
    fi

    LOG=${result_dir}/res.log
    echo "Log file: ${LOG}"
    CUDA_VISIBLE_DEVICES=${device} python ${PYTHON_FILE} \
      --dataset mvtec \
      --data_path ${DATA_PATH_MVTEC} \
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
      --lambda_router ${LAMBDA_ROUTER_VISA2MVTEC} \
      --topk_ratio ${TOPK_RATIO_MVTEC} \
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
