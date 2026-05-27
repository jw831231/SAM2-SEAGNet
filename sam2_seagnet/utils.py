import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def process_and_visualize_simple(image_tensor, mask_tensor, pred_tensor, output_dir, img_file_list, visualize=True):
    """Process one batch and save visualization + compute metrics."""
    from .metrics import compute_metrics, hausdorff_distance, hausdorff_distance_95

    batch_size = image_tensor.shape[0]
    results = []

    for i in range(batch_size):
        img = image_tensor[i]
        gt = mask_tensor[i]
        pred_logits = pred_tensor[i]

        img_file = img_file_list[i] if isinstance(img_file_list, (list, tuple)) else str(img_file_list)
        H_orig, W_orig = gt.shape[-2:]

        # Resize prediction to original size
        pred_resized = F.interpolate(
            pred_logits.unsqueeze(0).float(),
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Convert to numpy
        image_np = img.squeeze(0).cpu().numpy()
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[None, ...], 3, axis=0)
        image_np = (image_np.transpose(1, 2, 0) * 255).astype(np.uint8)

        gt_mask = gt.squeeze(0).cpu().numpy()
        pred_prob = torch.sigmoid(pred_resized).squeeze(0).cpu().numpy()
        pred_mask = (pred_prob > 0.5).astype(np.uint8)

        dice, iou = compute_metrics(pred_resized.unsqueeze(0), gt.unsqueeze(0))
        hd = hausdorff_distance(pred_mask, gt_mask)
        hd95 = hausdorff_distance_95(pred_mask, gt_mask)

        if visualize:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title("Original Ultrasound")
            axes[0, 0].axis("off")

            gt_overlay = image_np.copy()
            gt_overlay[gt_mask == 1] = [255, 0, 255]
            axes[0, 1].imshow(gt_overlay)
            axes[0, 1].set_title("Ground Truth")
            axes[0, 1].axis("off")

            pred_overlay = image_np.copy()
            pred_overlay[pred_mask == 1] = [0, 255, 0]
            axes[1, 0].imshow(pred_overlay)
            axes[1, 0].set_title("Prediction")
            axes[1, 0].axis("off")

            overlay = cv2.addWeighted(image_np, 0.7, pred_overlay, 0.3, 0)
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title(f"Overlay\nDice: {dice:.4f} | IoU: {iou:.4f}\nHD: {hd:.2f} | HD95: {hd95:.2f}")
            axes[1, 1].axis("off")

            plt.suptitle(f"Result: {img_file}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{img_file}_result.png"), dpi=150, bbox_inches="tight")
            plt.close()

        results.append({'dice': dice, 'iou': iou, 'hd': hd, 'hd95': hd95})

    return results
