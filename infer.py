import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

try:
    from sam2.build_sam import build_sam2
except ImportError:
    print("[Error] Please install segment-anything-2 first.")

from sam2_seagnet.models import SAM2PromptGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2-SEAGNet Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_prompt_generator.pth')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to sam2.1_hiera_base_plus.pt')
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1/sam2.1_hiera_b.yaml')
    parser.add_argument('--image_path', type=str, required=True, help='Path to single image or folder')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_overlay', action='store_true', default=True)
    return parser.parse_args()


def preprocess_image(image_path, device):
    """Read image and preprocess exactly like training."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize to 1024x1024
    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # Normalize like SAM2
    image_tensor = image_tensor.repeat(1, 3, 1, 1) * 255.0
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    return image_tensor, image  # return preprocessed + original for visualization


def postprocess_mask(pred_logits, original_size, threshold=0.5):
    """Resize prediction back to original size and binarize."""
    pred = torch.sigmoid(pred_logits)
    pred_resized = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
    mask = (pred_resized > threshold).float()
    return mask.squeeze().cpu().numpy().astype(np.uint8)


def save_results(original_image, pred_mask, output_path, filename, save_overlay=True):
    """Save prediction mask and optional overlay."""
    os.makedirs(output_path, exist_ok=True)

    # Save binary mask
    mask_path = os.path.join(output_path, f"{filename}_pred.png")
    cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))

    if save_overlay:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        overlay[pred_mask == 1] = [0, 255, 0]  # Green
        overlay_path = os.path.join(output_path, f"{filename}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

    return mask_path


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SAM2 and model...")
    sam = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device, apply_postprocessing=False)
    model = SAM2PromptGenerator(sam).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    model.eval()
    print("Model loaded successfully.")

    image_path = Path(args.image_path)

    if image_path.is_file():
        image_files = [image_path]
    elif image_path.is_dir():
        image_files = sorted(list(image_path.glob("*.png")) + list(image_path.glob("*.jpg")))
    else:
        raise FileNotFoundError(f"Path not found: {args.image_path}")

    print(f"Found {len(image_files)} image(s). Starting inference...")

    for img_file in image_files:
        try:
            image_tensor, original_image = preprocess_image(img_file, device)
            original_size = original_image.shape  # (H, W)

            with torch.no_grad():
                pred_logits = model(image_tensor.to(device))

            pred_mask = postprocess_mask(pred_logits, original_size, threshold=args.threshold)

            filename = img_file.stem
            save_results(original_image, pred_mask, args.output_dir, filename, save_overlay=args.save_overlay)

            print(f"Processed: {img_file.name}")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print("Inference completed! Results saved to", args.output_dir)


if __name__ == "__main__":
    main()
