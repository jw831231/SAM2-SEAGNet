import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    from sam2.build_sam import build_sam2
except ImportError:
    print("Please install segment-anything-2")

from sam2_seagnet.models import SAM2PromptGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sam2_checkpoint', type=str, required=True)
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1/sam2.1_hiera_b.yaml')
    parser.add_argument('--image_path', type=str, required=True, help='Single image or folder')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    sam = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device, apply_postprocessing=False)
    model = SAM2PromptGenerator(sam).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # TODO: Add proper inference loop for single image or folder
    print("Inference script created. You can extend it for batch inference.")
    print(f"Model loaded from {args.checkpoint}")

if __name__ == "__main__":
    main()
