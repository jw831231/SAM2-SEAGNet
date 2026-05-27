import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff


def compute_metrics(pred, target):
    """Compute Dice and IoU (tensor version)"""
    pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection) / (union + 1e-8)
    iou = intersection / (union - intersection + 1e-8)
    return dice.mean().item(), iou.mean().item()


def dice_coefficient(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return 2 * intersection / (union + 1e-8) if union > 0 else 0.0


def iou_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-8) if union > 0 else 0.0


def hausdorff_distance(pred, gt):
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('inf')
    pred_coords = np.argwhere(pred > 0)
    gt_coords = np.argwhere(gt > 0)
    hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
    hd2 = directed_hausdorff(gt_coords, pred_coords)[0]
    return max(hd1, hd2)


def hausdorff_distance_95(pred, gt):
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('inf')
    pred_coords = np.argwhere(pred > 0)
    gt_coords = np.argwhere(gt > 0)
    dist_matrix = np.sqrt(np.sum((pred_coords[:, np.newaxis, :] - gt_coords[np.newaxis, :, :])**2, axis=2))
    hd1 = np.percentile(np.min(dist_matrix, axis=1), 95)
    hd2 = np.percentile(np.min(dist_matrix, axis=0), 95)
    return max(hd1, hd2)
