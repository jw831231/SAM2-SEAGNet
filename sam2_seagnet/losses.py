import torch
import torch.nn as nn
import torch.nn.functional as F

# Sobel kernels for boundary loss
sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float()
sobel_kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float()


def boundary_loss(pred, target):
    """Boundary loss using Sobel gradients"""
    pred = torch.sigmoid(pred.float())
    target = target.float()

    pred_dx = F.conv2d(pred, sobel_kernel_x.to(pred.device), padding=1)
    pred_dy = F.conv2d(pred, sobel_kernel_y.to(pred.device), padding=1)
    target_dx = F.conv2d(target, sobel_kernel_x.to(target.device), padding=1)
    target_dy = F.conv2d(target, sobel_kernel_y.to(target.device), padding=1)

    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def dice_loss(pred, target):
    smooth = 1e-8
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)


class CombinedLoss(nn.Module):
    """Combined loss: BCE + Dice + Boundary"""
    def __init__(self, boundary_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        bce_dice = self.bce(pred, target) + dice_loss(pred, target)
        return bce_dice + self.boundary_weight * boundary_loss(pred, target)
