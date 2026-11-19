#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

device=${DEVICE:-0}

depth=(9)
n_ctx=(12)
t_n_ctx=(4)

EXP_TAG=${EXP_TAG:-pace_clip}
TARGET_EPOCH=${TARGET_EPOCH:-5}
RESUME_EPOCH=${RESUME_EPOCH:-0}
PYTHON_FILE=${PYTHON_FILE:-train.py}

DATA_PATH_VISA=${DATA_PATH_VISA:-./data/visa}
DATA_PATH_MVTEC=${DATA_PATH_MVTEC:-./data/mvtec}

# PACE-CLIP uses two independent prompt learners.
# Use 4 by default to reduce OOM risk. If memory is enough, set BATCH_SIZE=8.
BATCH_SIZE=${BATCH_SIZE:-4}
IMAGE_SIZE=${IMAGE_SIZE:-518}
GAMMA_CA=${GAMMA_CA:-0.1}
PIXEL_PATCH_RATIO=${PIXEL_PATCH_RATIO:-0.30}
IMAGE_PATCH_RATIO=${IMAGE_PATCH_RATIO:-0.10}
PATCH_ROUTER_TAU=${PATCH_ROUTER_TAU:-0.1}

# Direction-specific router weights
LAMBDA_ROUTER_VISA2MVTEC=${LAMBDA_ROUTER_VISA2MVTEC:-0.9}
LAMBDA_ROUTER_MVTEC2VISA=${LAMBDA_ROUTER_MVTEC2VISA:-0.7}

# Training image top-k. Testing still uses target-specific top-k in test script.
TOPK_RATIO=${TOPK_RATIO:-0.001}
CPG_TOPK_RATIO=${CPG_TOPK_RATIO:-0.05}

PIXEL_PROMPT_LR=${PIXEL_PROMPT_LR:-5e-5}
PIXEL_CROSS_LR=${PIXEL_CROSS_LR:-5e-5}
IMAGE_PROMPT_LR=${IMAGE_PROMPT_LR:-5e-5}
IMAGE_CROSS_LR=${IMAGE_CROSS_LR:-5e-5}
FUSION_LR=${FUSION_LR:-5e-3}

LAMBDA_PIXEL=${LAMBDA_PIXEL:-4.0}
LAMBDA_IMAGE_START=${LAMBDA_IMAGE_START:-0.1}
LAMBDA_IMAGE_END=${LAMBDA_IMAGE_END:-0.4}
LAMBDA_BALANCE=${LAMBDA_BALANCE:-0.05}
IMAGE_DELAY_EPOCHS=${IMAGE_DELAY_EPOCHS:-1}

FUSION_FLOOR_RATIO=${FUSION_FLOOR_RATIO:-0.10}
FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}

FEATURES_LIST="6 12 18 24"
FEATURE_MAP_LAYER="0 1 2 3"

echo "=========================================="
echo "PACE-CLIP Train: Pixel-Anchored Dual-Granularity Evidence Fusion"
echo "EXP_TAG = ${EXP_TAG}"
echo "TARGET_EPOCH = ${TARGET_EPOCH}"
echo "RESUME_EPOCH = ${RESUME_EPOCH}"
echo "PYTHON_FILE = ${PYTHON_FILE}"
echo "BATCH_SIZE = ${BATCH_SIZE}"
echo "PIXEL/IMAGE PATCH_RATIO = ${PIXEL_PATCH_RATIO}/${IMAGE_PATCH_RATIO} (image ratio kept but local maps use pixel detach)"
echo "Pixel LR prompt/cross = ${PIXEL_PROMPT_LR}/${PIXEL_CROSS_LR}"
echo "Image LR prompt/cross/fusion = ${IMAGE_PROMPT_LR}/${IMAGE_CROSS_LR}/${FUSION_LR}"
echo "LAMBDA_PIXEL = ${LAMBDA_PIXEL}"
echo "LAMBDA_IMAGE = ${LAMBDA_IMAGE_START} -> ${LAMBDA_IMAGE_END}, delay=${IMAGE_DELAY_EPOCHS}"
echo "LAMBDA_BALANCE = ${LAMBDA_BALANCE}"
echo "Image local evidence = pixel_maps.detach() top-k"
echo "=========================================="

# ============================================================
# 1. Train on VisA, test on MVTec later
# ============================================================
for i in "${!depth[@]}"; do
  for j in "${!n_ctx[@]}"; do
    base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_Visa2MVTec
    save_dir=./checkpoints/${base_dir}/
    mkdir -p ${save_dir}

    if [ "${RESUME_EPOCH}" -gt 0 ]; then
      resume_path=${save_dir}/epoch_${RESUME_EPOCH}.pth
      if [ ! -f "${resume_path}" ]; then
        echo "Error: resume checkpoint not found: ${resume_path}"
        exit 1
      fi
      RESUME_ARGS="--resume_path ${resume_path}"
      LOG=${save_dir}/train_epoch_$((RESUME_EPOCH + 1))_${TARGET_EPOCH}.log
      echo "Resume VisA -> MVTec from ${resume_path}"
    else
      RESUME_ARGS=""
      LOG=${save_dir}/train_epoch_1_${TARGET_EPOCH}.log
      echo "Train VisA -> MVTec from scratch"
    fi

    echo "Log file: ${LOG}"
    CUDA_VISIBLE_DEVICES=${device} python ${PYTHON_FILE} \
      --dataset visa \
      --train_data_path ${DATA_PATH_VISA} \
      --save_path ${save_dir} \
      --features_list ${FEATURES_LIST} \
      --feature_map_layer ${FEATURE_MAP_LAYER} \
      --image_size ${IMAGE_SIZE} \
      --batch_size ${BATCH_SIZE} \
      --print_freq 1 \
      --epoch ${TARGET_EPOCH} \
      --save_freq 1 \
      --depth ${depth[i]} \
      --n_ctx ${n_ctx[j]} \
      --t_n_ctx ${t_n_ctx[0]} \
      --gamma_ca ${GAMMA_CA} \
      --lambda_router ${LAMBDA_ROUTER_VISA2MVTEC} \
      --topk_ratio ${TOPK_RATIO} \
      --cpg_topk_ratio ${CPG_TOPK_RATIO} \
      --patch_router_tau ${PATCH_ROUTER_TAU} \
      --pixel_patch_ratio ${PIXEL_PATCH_RATIO} \
      --image_patch_ratio ${IMAGE_PATCH_RATIO} \
      --pixel_prompt_lr ${PIXEL_PROMPT_LR} \
      --pixel_cross_lr ${PIXEL_CROSS_LR} \
      --image_prompt_lr ${IMAGE_PROMPT_LR} \
      --image_cross_lr ${IMAGE_CROSS_LR} \
      --fusion_lr ${FUSION_LR} \
      --lambda_pixel ${LAMBDA_PIXEL} \
      --lambda_image_start ${LAMBDA_IMAGE_START} \
      --lambda_image_end ${LAMBDA_IMAGE_END} \
      --lambda_balance ${LAMBDA_BALANCE} \
      --fusion_floor_ratio ${FUSION_FLOOR_RATIO} \
      --fusion_temperature ${FUSION_TEMPERATURE} \
      --image_delay_epochs ${IMAGE_DELAY_EPOCHS} \
      ${RESUME_ARGS} \
      2>&1 | tee ${LOG}
    wait
  done
done

# ============================================================
# 2. Train on MVTec, test on VisA later
# ============================================================
for i in "${!depth[@]}"; do
  for j in "${!n_ctx[@]}"; do
    base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_${EXP_TAG}_MVTec2Visa
    save_dir=./checkpoints/${base_dir}/
    mkdir -p ${save_dir}

    if [ "${RESUME_EPOCH}" -gt 0 ]; then
      resume_path=${save_dir}/epoch_${RESUME_EPOCH}.pth
      if [ ! -f "${resume_path}" ]; then
        echo "Error: resume checkpoint not found: ${resume_path}"
        exit 1
      fi
      RESUME_ARGS="--resume_path ${resume_path}"
      LOG=${save_dir}/train_epoch_$((RESUME_EPOCH + 1))_${TARGET_EPOCH}.log
      echo "Resume MVTec -> VisA from ${resume_path}"
    else
      RESUME_ARGS=""
      LOG=${save_dir}/train_epoch_1_${TARGET_EPOCH}.log
      echo "Train MVTec -> VisA from scratch"
    fi

    echo "Log file: ${LOG}"
    CUDA_VISIBLE_DEVICES=${device} python ${PYTHON_FILE} \
      --dataset mvtec \
      --train_data_path ${DATA_PATH_MVTEC} \
      --save_path ${save_dir} \
      --features_list ${FEATURES_LIST} \
      --feature_map_layer ${FEATURE_MAP_LAYER} \
      --image_size ${IMAGE_SIZE} \
      --batch_size ${BATCH_SIZE} \
      --print_freq 1 \
      --epoch ${TARGET_EPOCH} \
      --save_freq 1 \
      --depth ${depth[i]} \
      --n_ctx ${n_ctx[j]} \
      --t_n_ctx ${t_n_ctx[0]} \
      --gamma_ca ${GAMMA_CA} \
      --lambda_router ${LAMBDA_ROUTER_MVTEC2VISA} \
      --topk_ratio ${TOPK_RATIO} \
      --cpg_topk_ratio ${CPG_TOPK_RATIO} \
      --patch_router_tau ${PATCH_ROUTER_TAU} \
      --pixel_patch_ratio ${PIXEL_PATCH_RATIO} \
      --image_patch_ratio ${IMAGE_PATCH_RATIO} \
      --pixel_prompt_lr ${PIXEL_PROMPT_LR} \
      --pixel_cross_lr ${PIXEL_CROSS_LR} \
      --image_prompt_lr ${IMAGE_PROMPT_LR} \
      --image_cross_lr ${IMAGE_CROSS_LR} \
      --fusion_lr ${FUSION_LR} \
      --lambda_pixel ${LAMBDA_PIXEL} \
      --lambda_image_start ${LAMBDA_IMAGE_START} \
      --lambda_image_end ${LAMBDA_IMAGE_END} \
      --lambda_balance ${LAMBDA_BALANCE} \
      --fusion_floor_ratio ${FUSION_FLOOR_RATIO} \
      --fusion_temperature ${FUSION_TEMPERATURE} \
      --image_delay_epochs ${IMAGE_DELAY_EPOCHS} \
      ${RESUME_ARGS} \
      2>&1 | tee ${LOG}
    wait
  done
done
