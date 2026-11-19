# PACE-CLIP

**PACE-CLIP: Pixel-Anchored Cross-granularity Evidence Learning for Zero-Shot Industrial Anomaly Detection**

This repository contains the PACE-CLIP implementation prepared for confidential editorial and peer-review access. It is not a public code release. The five datasets listed below are publicly available from their respective sources; this repository does not redistribute dataset images or annotations.

## Introduction

PACE-CLIP is a zero-shot industrial anomaly detection framework. Its implementation is organized around three method components:

- **DCPG**: Defect-aware Conditional Prompt Generation.
- **DGDL**: Dual-Granularity Decoupled Learning.
- **PABEF**: Pixel-Anchored Balanced Evidence Fusion.

The project uses an AnomalyCLIP/CLIP visual-language backbone.

## Project structure

```text
PACE-CLIP/
├── AnomalyCLIP_lib/
├── generate_dataset_json/
│   ├── mvtec.py
│   ├── visa.py
│   ├── btad.py
│   ├── mpdd.py
│   └── DTD.py
├── dcpg.py
├── dgdl.py
├── pabef.py
├── train.py
├── test.py
├── dataset.py
├── logger.py
├── loss.py
├── metrics.py
├── utils.py
├── train.sh
├── test.sh
├── test_btad_mpdd.sh
├── test_dtd_synthetic.sh
├── requirements.txt
├── README.md
├── METHOD_CODE_AUDIT.md
└── THIRD_PARTY_NOTICES.md
```

## Method-to-code mapping

- `dcpg.py`: DCPG prompt-generation implementation.
- `dgdl.py`: DGDL shallow-deep semantic enhancement helper. The independent image-level and pixel-level branches and their training objectives are implemented in `train.py` and evaluated in `test.py`.
- `pabef.py`: PABEF image-level evidence fusion.
- `AnomalyCLIP_lib/AnomalyCLIP.py`: modified visual attention implementation using V-V self-attention.
- `train.py`: training entry point.
- `test.py`: evaluation entry point.

## How to run

### 1. Environment

The supplied environment uses PyTorch 2.0.0 and torchvision 0.15.1. Python 3.10 is recommended.

```bash
conda create -n paceclip python=3.10 -y
conda activate paceclip
pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with the CUDA version available on the machine.

### 2. Prepare the datasets

The experiments use five public datasets:

- **MVTec AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **VisA**: https://github.com/amazon-science/spot-diff
- **BTAD**: http://avires.dimi.uniud.it/papers/btad/btad.zip
- **MPDD**: https://github.com/stepanje/MPDD
- **DTD-Synthetic**: https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1

Download the datasets from their respective sources. Arrange the files in the directory structures expected by the corresponding metadata-generation scripts. Dataset files are not included in this repository.

### 3. Generate dataset metadata

Each dataset directory should contain the original images and masks. Run the corresponding script from the repository root. For example:

```bash
python generate_dataset_json/mvtec.py --root ./data/mvtec
python generate_dataset_json/visa.py --root ./data/visa
python generate_dataset_json/btad.py --root ./data/BTAD/BTech_Dataset_transformed
python generate_dataset_json/mpdd.py --root ./data/MPDD
python generate_dataset_json/DTD.py --root ./data/DTD
```

Each script creates a `meta.json` file in the specified dataset root. The data loader reads the `train` split during training and the `test` split during evaluation.

### 4. Configure paths

The shell scripts use environment variables for dataset locations. The default paths are relative to the repository root:

```bash
export DATA_PATH_MVTEC=./data/mvtec
export DATA_PATH_VISA=./data/visa
export DATA_PATH_BTAD=./data/BTAD/BTech_Dataset_transformed
export DATA_PATH_MPDD=./data/MPDD
export DATA_PATH_DTD=./data/DTD
```

### 5. Train PACE-CLIP

```bash
bash train.sh
```

The training script runs the two auxiliary-domain training directions. Checkpoints are written under `./checkpoints/`.

### 6. Evaluate PACE-CLIP

Evaluate MVTec AD and VisA cross-domain settings:

```bash
bash test.sh
```

Evaluate BTAD and MPDD:

```bash
bash test_btad_mpdd.sh
```

Evaluate DTD-Synthetic:

```bash
bash test_dtd_synthetic.sh
```

The evaluation scripts expect the corresponding trained checkpoints under `./checkpoints/`. Results and logs are written under `./results/`.

## Pretrained CLIP weights

The model uses the `ViT-L/14@336px` backbone. Pretrained weights are not included. The model loader uses the configured CLIP download/cache mechanism. The default cache directory is `~/.cache/clip`; set `CLIP_CACHE_DIR` to use another location.

## Code and data availability

The PACE-CLIP source code in this package is provided only for confidential editorial and peer-review access. Do not publish this package to a public GitHub repository. The five datasets are publicly available from the external sources listed above and remain subject to their respective terms and licenses.

## Citation

Please cite the PACE-CLIP manuscript when using this implementation. Add the final bibliographic information after publication.
