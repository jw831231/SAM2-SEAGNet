import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt

try:
    from sam2.build_sam import build_sam2
except ImportError:
    print("Please install segment-anything-2 first (see README)")

from sam2_seagnet.models import SAM2PromptGenerator
from sam2_seagnet.dataset import CAMUSDataset, get_train_transform, get_val_transform
from sam2_seagnet.losses import CombinedLoss
from sam2_seagnet.metrics import compute_metrics
from sam2_seagnet.utils import process_and_visualize_simple


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2-SEAGNet Training")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to images folder')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to masks folder')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to sam2.1_hiera_base_plus.pt')
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1/sam2.1_hiera_b.yaml', help='SAM2 config yaml path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device

    # ==================== Load SAM2 ====================
    print("Loading SAM2...")
    sam = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device, apply_postprocessing=False)

    # ==================== Model ====================
    model = SAM2PromptGenerator(sam, num_classes=args.num_classes).to(device)
    for param in model.feature_extractor.image_encoder.parameters():
        param.requires_grad = False

    # ==================== Data ====================
    # Simple patient split (you can improve this)
    all_patients = sorted(list(set([f.split('_')[0] for f in os.listdir(args.image_dir) if f.endswith('.png')])))
    random.shuffle(all_patients)
    n = len(all_patients)
    train_patients = all_patients[:int(0.8 * n)]
    val_patients = all_patients[int(0.8 * n):int(0.9 * n)]
    test_patients = all_patients[int(0.9 * n):]

    train_dataset = CAMUSDataset(args.image_dir, args.mask_dir, transform=get_train_transform(), patient_list=train_patients)
    val_dataset = CAMUSDataset(args.image_dir, args.mask_dir, transform=get_val_transform(), patient_list=val_patients)
    test_dataset = CAMUSDataset(args.image_dir, args.mask_dir, transform=get_val_transform(), patient_list=test_patients)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    # ==================== Training Setup ====================
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = CombinedLoss()
    scaler = GradScaler(device='cuda' if device == 'cuda' else 'cpu')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_dice = -1.0
    start_epoch = 0
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    best_model_path = os.path.join(args.output_dir, 'best_prompt_generator.pth')

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_dice = checkpoint.get('best_val_dice', -1.0)
        print(f"Resumed from epoch {start_epoch}")

    # ==================== Training Loop ====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device).float()

            # Preprocess (resize to 1024x1024 and normalize like SAM2)
            images_resized = F.interpolate(images, size=(1024, 1024), mode='bicubic', align_corners=False)
            images_resized = images_resized.repeat(1, 3, 1, 1) * 255.0
            mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
            images_resized = (images_resized - mean) / std
            masks_resized = F.interpolate(masks, size=(1024, 1024), mode='nearest')

            optimizer.zero_grad()
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                pred = model(images_resized)
                loss = criterion(pred, masks_resized)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device).float()
                images_resized = F.interpolate(images, size=(1024, 1024), mode='bicubic', align_corners=False).repeat(1, 3, 1, 1) * 255.0
                images_resized = (images_resized - mean) / std
                masks_resized = F.interpolate(masks, size=(1024, 1024), mode='nearest')
                pred = model(images_resized)
                dice, _ = compute_metrics(pred, masks_resized)
                val_dice += dice
        val_dice /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
        scheduler.step(-val_dice)

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({'model_state_dict': model.state_dict(), 'val_dice': val_dice}, best_model_path)
            print(f"  -> New best model saved (Val Dice: {val_dice:.4f})")

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'best_val_dice': best_val_dice
        }, checkpoint_path)

    print("Training finished!")

    # ==================== Final Test Evaluation ====================
    print("Running final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
    model.eval()

    metrics = {'dice': [], 'iou': [], 'hd': [], 'hd95': []}
    with torch.no_grad():
        for idx, (images, masks, img_files) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device).float()
            images_resized = F.interpolate(images, size=(1024, 1024), mode='bicubic', align_corners=False).repeat(1, 3, 1, 1) * 255.0
            images_resized = (images_resized - mean) / std
            pred = model(images_resized)

            batch_results = process_and_visualize_simple(
                images, masks, pred, args.output_dir, img_files, visualize=(idx < 3)
            )
            for r in batch_results:
                metrics['dice'].append(r['dice'])
                metrics['iou'].append(r['iou'])
                metrics['hd'].append(r['hd'])
                metrics['hd95'].append(r['hd95'])

    print(f"Test Dice: {np.mean(metrics['dice']):.4f} \u00b1 {np.std(metrics['dice']):.4f}")
    print(f"Test IoU : {np.mean(metrics['iou']):.4f}")
    print(f"Test HD  : {np.mean(metrics['hd']):.2f}")


if __name__ == "__main__":
    main()
