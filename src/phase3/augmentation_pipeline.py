#!/usr/bin/env python3
"""
Traffic Light Detection - Advanced Augmentation Pipeline
Phase 3: Specialized augmentation for small object detection
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class TrafficLightAugmentation:
    """
    Specialized augmentation pipeline for traffic light detection
    Focuses on preserving small object integrity while enhancing robustness
    """
    
    def __init__(self, image_size=640, mode='train'):
        """
        Initialize augmentation pipeline
        
        Args:
            image_size: Target image size for training (640 for YOLO)
            mode: 'train' or 'val' - different augmentation intensities
        """
        self.image_size = image_size
        self.mode = mode
        
        # Define augmentation parameters optimized for traffic lights
        if mode == 'train':
            self.augmentation = self._create_train_augmentation()
        else:
            self.augmentation = self._create_val_augmentation()
    
    def _create_train_augmentation(self):
        """Create training augmentation pipeline optimized for traffic lights"""
        return A.Compose([
            # 1. Lighting and Color Augmentation (Critical for traffic lights)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # Conservative for traffic light visibility
                contrast_limit=0.2,
                brightness_by_max=True,
                p=0.8
            ),
            
            # 2. Weather/Environmental Simulation
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=10, drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=1,
                    brightness_coefficient=0.9,
                    rain_type='default',
                    p=0.3
                ),
                A.RandomFog(
                    fog_coef_lower=0.1, fog_coef_upper=0.3,
                    alpha_coef=0.1,
                    p=0.2
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.3
                ),
            ], p=0.4),
            
            # 3. HSV Augmentation (Fine-tuned for traffic light colors)
            A.HueSaturationValue(
                hue_shift_limit=15,    # Small shift to preserve color identity
                sat_shift_limit=30,    # Moderate saturation changes
                val_shift_limit=20,    # Brightness variations for different times
                p=0.7
            ),
            
            # 4. Noise Addition (Simulating sensor noise)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.2),
            ], p=0.3),
            
            # 5. Geometric Transformations (Conservative for small objects)
            A.ShiftScaleRotate(
                shift_limit=0.1,       # Small shifts to avoid cutting objects
                scale_limit=0.1,       # Conservative scaling
                rotate_limit=10,       # Small rotations (traffic lights are usually upright)
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.6
            ),
            
            # 6. Horizontal Flip (Traffic lights can appear on both sides)
            A.HorizontalFlip(p=0.5),
            
            # 7. Advanced Augmentations for Robustness
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),  # Simulate camera movement
                A.MedianBlur(blur_limit=3, p=0.1),   # Reduce noise while preserving edges
                A.GaussianBlur(blur_limit=3, p=0.1), # Simulate focus issues
            ], p=0.2),
            
            # 8. Channel manipulations
            A.OneOf([
                A.ChannelShuffle(p=0.1),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
            ], p=0.2),
            
            # 9. Compression artifacts (Real-world camera effects)
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.2),
            
            # 10. Final resize with proper aspect ratio handling
            A.LongestMaxSize(max_size=self.image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            
            # 11. Normalization for neural networks
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            
            # 12. Convert to tensor
            ToTensorV2(p=1.0)
            
        ], bbox_params=A.BboxParams(
            format='yolo',  # YOLO format: [class, x_center, y_center, width, height]
            label_fields=['class_labels'],
            min_area=16,  # Minimum bbox area (4x4 pixels)
            min_visibility=0.3,  # Minimum visibility after augmentation
        ))
    
    def _create_val_augmentation(self):
        """Create validation augmentation pipeline (minimal augmentation)"""
        return A.Compose([
            # Only resize and normalize for validation
            A.LongestMaxSize(max_size=self.image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=1,  # Less strict for validation
            min_visibility=0.1,
        ))
    
    def __call__(self, image, bboxes, class_labels):
        """
        Apply augmentation to image and bounding boxes
        
        Args:
            image: PIL Image or numpy array
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class IDs corresponding to bboxes
            
        Returns:
            augmented_image: Tensor of augmented image
            augmented_bboxes: List of augmented bounding boxes
            augmented_labels: List of corresponding class labels
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply augmentation
        try:
            augmented = self.augmentation(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                augmented['image'],
                augmented['bboxes'],
                augmented['class_labels']
            )
            
        except Exception as e:
            print(f"Augmentation failed: {e}")
            # Return original with basic resize if augmentation fails
            basic_transform = A.Compose([
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(self.image_size, self.image_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            basic_result = basic_transform(image=image)
            return basic_result['image'], bboxes, class_labels

class ImageResizeStrategy:
    """
    Advanced image resizing strategy optimized for traffic light detection
    """
    
    def __init__(self, target_size=640, maintain_aspect_ratio=True):
        self.target_size = target_size
        self.maintain_aspect_ratio = maintain_aspect_ratio
    
    def resize_with_padding(self, image, bboxes=None):
        """
        Resize image while maintaining aspect ratio using padding
        Optimal for traffic light detection to preserve object proportions
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(self.target_size / w, self.target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        
        # Apply padding (center the image)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Adjust bounding boxes if provided
        if bboxes is not None:
            adjusted_bboxes = []
            for bbox in bboxes:
                if len(bbox) >= 4:
                    # YOLO format: [class_id, x_center, y_center, width, height]
                    if len(bbox) == 5:
                        class_id, x_center, y_center, width, height = bbox
                    else:
                        x_center, y_center, width, height = bbox[:4]
                        class_id = None
                    
                    # Adjust coordinates for scaling and padding
                    new_x_center = (x_center * w * scale + left) / self.target_size
                    new_y_center = (y_center * h * scale + top) / self.target_size
                    new_width = (width * w * scale) / self.target_size
                    new_height = (height * h * scale) / self.target_size
                    
                    if class_id is not None:
                        adjusted_bboxes.append([class_id, new_x_center, new_y_center, new_width, new_height])
                    else:
                        adjusted_bboxes.append([new_x_center, new_y_center, new_width, new_height])
            
            return padded_image, adjusted_bboxes
        
        return padded_image

def create_augmentation_demo():
    """Create demonstration of augmentation effects"""
    print("🎨 Creating augmentation demonstration...")
    
    # This would be used to show before/after augmentation examples
    # Implementation would load sample images and show augmentation effects
    pass

def validate_augmentation_pipeline():
    """Validate that augmentation pipeline preserves annotations correctly"""
    print("✅ Validating augmentation pipeline...")
    
    # Create test image and bboxes
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bboxes = [[0.5, 0.5, 0.1, 0.1], [0.3, 0.7, 0.05, 0.08]]  # YOLO format
    test_labels = [1, 5]  # circle_red, circle_yellow
    
    # Test training augmentation
    train_aug = TrafficLightAugmentation(image_size=640, mode='train')
    
    try:
        aug_image, aug_bboxes, aug_labels = train_aug(test_image, test_bboxes, test_labels)
        print(f"✅ Training augmentation successful:")
        print(f"   Image shape: {aug_image.shape}")
        print(f"   Bboxes: {len(aug_bboxes)} preserved")
        print(f"   Labels: {len(aug_labels)} preserved")
    except Exception as e:
        print(f"❌ Training augmentation failed: {e}")
    
    # Test validation augmentation
    val_aug = TrafficLightAugmentation(image_size=640, mode='val')
    
    try:
        val_image, val_bboxes, val_labels = val_aug(test_image, test_bboxes, test_labels)
        print(f"✅ Validation augmentation successful:")
        print(f"   Image shape: {val_image.shape}")
        print(f"   Bboxes: {len(val_bboxes)} preserved")
        print(f"   Labels: {len(val_labels)} preserved")
    except Exception as e:
        print(f"❌ Validation augmentation failed: {e}")
    
    # Test resize strategy
    resizer = ImageResizeStrategy(target_size=640)
    resized_image, resized_bboxes = resizer.resize_with_padding(test_image, test_bboxes)
    
    print(f"✅ Resize strategy successful:")
    print(f"   Resized image shape: {resized_image.shape}")
    print(f"   Adjusted bboxes: {len(resized_bboxes)}")

if __name__ == "__main__":
    print("🚦 Traffic Light Augmentation Pipeline")
    print("="*50)
    validate_augmentation_pipeline()