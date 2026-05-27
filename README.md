# SAM2-SEAGNet

**SAM2-SEAGNet: SAM2 Network with Speckle-Edge Aware Attention Gate for Cardiac Structure Segmentation in Echocardiographic Images**

Official PyTorch implementation for the research paper.

**Authors**: Wei Jiang*, Wenzhu Wu, Yu Li, Yongxi Qin, Xi Li (corresponding)
**Affiliation**: Chongqing Medical and Pharmaceutical College, Key Laboratory of Brain-Machine Fusion and Intelligent Medical Equipment

---

## Abstract

Echocardiography is essential for clinical cardiovascular disease diagnosis, but accurate segmentation of cardiac structures (LV, LA, MYO) remains challenging due to **multi-scale variations**, **severe speckle noise**, and **blurred tissue boundaries**.

We propose **SAM2-SEAGNet**, a U-shaped encoder-decoder built on SAM2:
- Customized **ASPP** module for multi-scale feature representation.
- Novel lightweight **Speckle-Edge Aware Attention Gate (SEAG)** embedded in skip connections to suppress noise and refine boundaries.

On CAMUS dataset (patient-level split):
- **LV**: Dice 93.23%, IoU 87.57%, HD 8.78mm (best)
- Strong improvements in boundary accuracy (HD reduced significantly vs SOTA).

Ablation confirms ASPP + SEAG synergy.

## Key Results

| Dataset    | Dice (%) | IoU (%) | HD (mm) | Notes                  |
|------------|----------|---------|---------|------------------------|
| CAMUS-LV  | **93.23** | **87.57** | **8.78** | Best overall          |
| CAMUS-MYO | 86.96    | 77.28   | 10.36   | Good myocardium       |
| CAMUS-LA  | 88.99    | 81.32   | 10.19   | Competitive LA        |

Compared to MSA, SwinUNet, MedSAM, SAMed, original SAM.

## Highlights & Innovations

1. **Speckle-and-Edge Aware Attention Gate (SEAG)**
   - Computes speckle intensity map (local variance from 3x3 pooling).
   - Computes edge strength map (Sobel gradients).
   - Concatenates with current decoder features + skip features.
   - Generates spatial attention weights for noise-robust, edge-preserving fusion.

2. **Customized ASPP** in bottleneck (atrous + global avg pool branches).

3. **4-stage Decoder** with CBAM attention + SEAG skip connections (first two layers).

4. Frozen SAM2 Hiera encoder + lightweight adapters for domain adaptation to echocardiography.

## Repository Structure (Recommended)

```
SAM2-SEAGNet/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── index.py                 # Your full training script (upload it!)
├── paper/
│   └── SAM2-SEAGNet_paper.docx   # Upload your manuscript
├── assets/                  # Add figures from paper if desired
├── checkpoints/             # gitignored - put your .pth here
└── docs/                    # Future: extended docs
```

## How to Use This Repo

### 1. Setup

```bash
git clone https://github.com/jw831231/SAM2-SEAGNet.git
cd SAM2-SEAGNet
```

Install dependencies (see requirements.txt). **SAM2 installation is the trickiest part** — follow the steps in your original `index.py` (it has a robust subprocess + yaml copy method).

### 2. Data

Prepare CAMUS as PNG pairs (images/ + masks/). Use patient-level 8:1:1 split as in the code.

### 3. Training

**Option A (Recommended for now)**: 
Upload your working `index.py` to the repo root (or rename to `train.py`).
Edit the path variables at the top to point to your local SAM2 folder, CAMUS data, and output directory.
Run `python index.py` (set RESUME_TRAINING as needed).

**Option B (Future)**: We can modularize into `sam2_seagnet/models.py`, `train.py` with argparse for cleaner CLI.

## Citation

Please cite the paper if you use this code or method:

```bibtex
@article{Jiang2026SAM2SEAGNet,
  title={SAM2-SEAGNet: SAM2 Network with Speckle-Edge Aware Attention Gate for Cardiac Structure Segmentation in Echocardiographic Images},
  author={Jiang, Wei and Wu, Wenzhu and Li, Yu and Qin, Yongxi and Li, Xi},
  year={2026}
}
```

## Next Steps & Contributions Welcome

- [ ] Upload full `index.py` and paper .docx
- [ ] Add modular Python package structure
- [ ] Add inference demo script
- [ ] Upload example pretrained weights (if sharing allowed)
- [ ] Add data preprocessing script for CAMUS

**Contact / Issues**: Open an issue or email 10720@cqmpc.edu.cn

---
*Built to promote reproducibility in medical ultrasound AI.*