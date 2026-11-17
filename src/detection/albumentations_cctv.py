"""
src/detection/albumentations_cctv.py
====================================
CCTV-realistic data augmentations using Albumentations:
- Motion blur (fast moving weapons / hands)
- Gaussian noise (low-light CCTV sensor grain)
- Random brightness / contrast (harsh lights, shadows)
- Image compression (JPEG CCTV streaming artifacts)
"""

import albumentations as A
import numpy as np


def get_cctv_train_transform():
    """Returns an Albumentations composition for CCTV training."""
    return A.Compose(
        [
            A.MotionBlur(blur_limit=(3, 7), p=0.25),
            A.GaussNoise(std_range=(0.05, 0.20), p=0.20),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.30),
            A.ImageCompression(quality_range=(40, 85), p=0.25),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.015, p=0.20),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1),
    )


def apply_cctv_transform(image: np.ndarray, bboxes: list, class_labels: list):
    """Safely applies CCTV augmentations with fallback if bounding boxes fail."""
    transform = get_cctv_train_transform()
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return augmented["image"], augmented["bboxes"], augmented["class_labels"]
    except Exception:
        return image, bboxes, class_labels
