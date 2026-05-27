# SAM2-SEAGNet

**SAM2-SEAGNet: SAM2 Network with Speckle-Edge Aware Attention Gate for Cardiac Structure Segmentation in Echocardiographic Images**

Official PyTorch implementation of the paper.

**Authors**: Wei Jiang* (corresponding author), Wenzhu Wu, Yu Li, Yongxi Qin
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
├── infer.py                 # Full inference script
├── scripts/
│   └── prepare_camus.py     # CAMUS data preprocessing script
├── configs/
│   └── default.yaml
├── sam2_seagnet/            # Core package
│   ├── __init__.py
│   ├── models.py            # SEAG, ASPP, CBAM, SAM2PromptGenerator
│   ├── dataset.py           # CAMUSDataset
│   ├── losses.py            # CombinedLoss (BCE + Dice + Boundary)
│   ├── metrics.py           # Dice, IoU, HD, HD95
│   └── utils.py             # Visualization helpers
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

# Download checkpoint (sam2.1_hiera_base_plus.pt)
# Place it somewhere accessible, e.g. ~/checkpoints/sam2.1_hiera_base_plus.pt

# Copy config yaml
mkdir -p sam2/configs/sam2.1
# Copy sam2.1_hiera_b.yaml to the above path
```

Return to the project folder.

### 3. Prepare CAMUS Data

Use the provided script:
```bash
python scripts/prepare_camus.py --input_dir /path/to/raw/camus --output_dir /path/to/processed/camus
```

Or manually convert to PNG pairs (images/ + masks/).

## Usage

### Training

```bash
python train.py \
    --image_dir /path/to/camus/images \
    --mask_dir /path/to/camus/masks \
    --output_dir ./output \
    --sam2_checkpoint /path/to/sam2.1_hiera_base_plus.pt \
    --sam2_config /path/to/sam2/configs/sam2.1/sam2.1_hiera_b.yaml \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

### Inference (New!)

```bash
# Single image
python infer.py \
    --checkpoint ./output/best_prompt_generator.pth \
    --sam2_checkpoint /path/to/sam2.1_hiera_base_plus.pt \
    --image_path /path/to/test/image.png \
    --output_dir ./results

# Folder (batch)
python infer.py \
    --checkpoint ./output/best_prompt_generator.pth \
    --sam2_checkpoint /path/to/sam2.1_hiera_base_plus.pt \
    --image_path /path/to/test_folder \
    --output_dir ./results
```

## Citation

```bibtex
@article{Jiang2026SAM2SEAGNet,
  title={SAM2-SEAGNet: SAM2 Network with Speckle-Edge Aware Attention Gate for Cardiac Structure Segmentation in Echocardiographic Images},
  author={Jiang, Wei and Wu, Wenzhu and Li, Yu and Qin, Yongxi},
  year={2026}
}
```

## Acknowledgments

Supported in part by the Science and Technology Research Program of Chongqing Education Commission of China.

**Contact**: 10720@cqmpc.edu.cn

---

*Repository created for reproducibility in medical ultrasound AI.*