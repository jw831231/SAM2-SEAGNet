# SAM2-SEAGNet

**SAM2-SEAGNet: SAM2 Network with Speckle-Edge Aware Attention Gate for Cardiac Structure Segmentation in Echocardiographic Images**

Official PyTorch implementation of the paper.

**Authors**: Wei Jiang*, Wenzhu Wu, Yu Li, Yongxi Qin, Xi Li (corresponding author)
**Affiliation**: Department of Basic Medical Sciences, Chongqing Medical and Pharmaceutical College | Key Laboratory of Brain-Machine Fusion and Intelligent Medical Equipment

---

## Abstract

Echocardiography is essential for clinical cardiovascular disease diagnosis, but accurate segmentation of cardiac structures (left ventricle LV, left atrium LA, left ventricular myocardium MYO) remains challenging due to **multi-scale structural variations**, **severe speckle noise**, and **blurred tissue boundaries**.

We propose **SAM2-SEAGNet**, a hierarchical U-shaped encoder-decoder segmentation network built upon Segment Anything Model 2 (SAM2). 

**Key contributions**:
- A customized **Atrous Spatial Pyramid Pooling (ASPP)** module to enhance multi-scale feature representation.
- A novel lightweight **Speckle-Edge Aware Attention Gate (SEAG)** designed to suppress speckle noise and refine ambiguous boundaries in skip connections.

Extensive experiments on three subsets of the public **CAMUS** dataset demonstrate state-of-the-art performance:
- **CAMUS-LV**: Dice **93.23%**, IoU **87.57%**, HD **8.78 mm**
- Significant boundary accuracy improvement (HD reduced by up to 46% on MYO compared to previous SOTA).

Ablation studies confirm the effectiveness and synergy of ASPP and SEAG modules.

## Highlights

- Novel **SEAG** module: Combines local variance (speckle-aware) + Sobel gradients (edge-aware) to generate spatial attention for robust feature fusion in ultrasound.
- Customized **ASPP** + 4-stage decoder with CBAM.
- Frozen SAM2 Hiera-Base-Plus encoder for strong pre-trained features + efficient domain adaptation.
- Complete training pipeline with checkpointing, mixed precision, boundary loss, and visualization.

## Results Summary (CAMUS Dataset)

| Task       | Dice (%) | IoU (%) | HD (mm) | Improvement vs SOTA (HD) |
|------------|----------|---------|---------|---------------------------|
| CAMUS-LV  | **93.23** | **87.57** | **8.78** | -22.2%                   |
| CAMUS-MYO | 86.96    | 77.28   | 10.36   | **-46.3%**               |
| CAMUS-LA  | 88.99    | 81.32   | 10.19   | -12.1%                   |

## Project Structure (Clean & Modular)

```
SAM2-SEAGNet/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── train.py                 # Clean training script with argparse
├── infer.py                 # Inference & visualization demo
├── sam2_seagnet/            # Core package
│   ├── __init__.py
│   ├── models.py            # SEAG, ASPP, CBAM, SAM2PromptGenerator
│   ├── dataset.py           # CAMUSDataset
│   ├── losses.py            # CombinedLoss (BCE + Dice + Boundary)
│   ├── metrics.py           # Dice, IoU, HD, HD95
│   └── utils.py             # Visualization helpers
├── configs/                 # (Future) default.yaml
├── paper/                   # Manuscript (.docx)
├── assets/
└── checkpoints/             # Trained weights (gitignored)
```

## Installation

### 1. Clone & Environment
```bash
git clone https://github.com/jw831231/SAM2-SEAGNet.git
cd SAM2-SEAGNet

conda create -n sam2seg python=3.10 -y
conda activate sam2seg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Install SAM2 (Critical Step)

SAM2 requires specific setup:

```bash
# Clone SAM2
cd ..
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Download checkpoint (sam2.1_hiera_base_plus.pt) from 
# https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt
# Place it somewhere accessible, e.g. ~/checkpoints/sam2.1_hiera_base_plus.pt

# Copy config yaml (important!)
mkdir -p sam2/configs/sam2.1
# Copy sam2.1_hiera_b.yaml from your SAM2 download or Kaggle input to the above path
```

Return to the project folder.

### 3. Prepare CAMUS Data

1. Download CAMUS dataset (registration required): https://humanheart-project.creatis.insa-lyon.fr/
2. Preprocess into PNG format:
   - One folder `images/` containing `patientID_xxx.png`
   - One folder `masks/` containing `patientID_xxx_mask.png`
3. Recommended split: patient-level 80% train / 10% val / 10% test

The code supports training separate models for **LV**, **MYO**, or **LA**.

## Usage

### Training (Recommended)

```bash
python train.py \
    --image_dir /path/to/camus/images \
    --mask_dir /path/to/camus/masks \
    --output_dir ./output \
    --sam2_checkpoint /path/to/sam2.1_hiera_base_plus.pt \
    --sam2_config sam2/configs/sam2.1/sam2.1_hiera_b.yaml \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --resume
```

Key features in `train.py`:
- argparse for all important parameters
- Automatic checkpointing & best model saving
- Resume training support
- Mixed precision (AMP)
- Training curves + test evaluation + visualization

### Inference & Visualization

```bash
python infer.py \
    --checkpoint ./output/best_prompt_generator.pth \
    --image_dir /path/to/test/images \
    --output_dir ./results
```

## Citation

If you use this work, please cite:

```bibtex
@article{Jiang2026SAM2SEAGNet,
  title={SAM2-SEAGNet: SAM2 Network with Speckle-Edge Aware Attention Gate for Cardiac Structure Segmentation in Echocardiographic Images},
  author={Jiang, Wei and Wu, Wenzhu and Li, Yu and Qin, Yongxi and Li, Xi},
  year={2026}
}
```

## Acknowledgments

This work was supported in part by the Science and Technology Research Program of Chongqing Education Commission of China.

**Contact**: 10720@cqmpc.edu.cn

---

*Repository created for reproducibility and advancing medical ultrasound AI research.*