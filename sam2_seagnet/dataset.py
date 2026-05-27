import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CAMUSDataset(Dataset):
    """CAMUS Dataset for cardiac structure segmentation"""
    def __init__(self, image_dir, mask_dir, transform=None, patient_list=None):
        all_image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.endswith('.png') and '_mask' not in f])

        if patient_list is not None:
            selected = set(patient_list)
            all_image_files = [f for f in all_image_files if f.split('_')[0] in selected]

        self.image_files = all_image_files
        self.mask_files = [f.replace('.png', '_mask.png') for f in all_image_files]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        assert len(self.image_files) == len(self.mask_files), "Image and mask counts do not match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)

        img_file = self.image_files[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask, img_file


def get_train_transform():
    return A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2()
    ])
