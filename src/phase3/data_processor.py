#!/usr/bin/env python3
"""
Traffic Light Detection - Data Processing Pipeline
Phase 3 Implementation: Train/Val Split, Augmentation, Resizing Strategy
"""

import os
import shutil
import random
from pathlib import Path
import json
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

class TrafficLightDataProcessor:
    def __init__(self, source_data_path="./ML2-tld-colorization-dataset", 
                 output_path="./atlas_dataset", split_ratio=0.8):
        """
        Initialize data processor for traffic light detection
        
        Args:
            source_data_path: Path to extracted ATLAS dataset
            output_path: Path for processed YOLO format dataset
            split_ratio: Train/validation split ratio (0.8 = 80% train, 20% val)
        """
        self.source_path = Path(source_data_path)
        self.output_path = Path(output_path)
        self.split_ratio = split_ratio
        
        # Create output directory structure
        self.setup_directories()
        
        # Class mapping (25 traffic light classes)
        self.class_names = {
            0: "circle_green",
            1: "circle_red", 
            2: "off",
            3: "circle_red_yellow",
            4: "arrow_left_green",
            5: "circle_yellow",
            6: "arrow_right_red",
            7: "arrow_left_red",
            8: "arrow_straight_red",
            9: "arrow_left_red_yellow",
            10: "arrow_left_yellow",
            11: "arrow_straight_yellow",
            12: "arrow_right_red_yellow",
            13: "arrow_right_green",
            14: "arrow_right_yellow",
            15: "arrow_straight_green",
            16: "arrow_straight_left_green",
            17: "arrow_straight_red_yellow",
            18: "arrow_straight_left_red",
            19: "arrow_straight_left_yellow",
            20: "arrow_straight_left_red_yellow",
            21: "arrow_straight_right_red",
            22: "arrow_straight_right_red_yellow",
            23: "arrow_straight_right_yellow",
            24: "arrow_straight_right_green"
        }
        
        # Statistics tracking
        self.stats = {
            'class_distribution': Counter(),
            'image_sizes': [],
            'bbox_sizes': [],
            'total_images': 0,
            'total_annotations': 0
        }

    def setup_directories(self):
        """Create YOLO format directory structure"""
        dirs = [
            self.output_path / "images" / "train",
            self.output_path / "images" / "val", 
            self.output_path / "labels" / "train",
            self.output_path / "labels" / "val"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")

    def analyze_dataset_structure(self):
        """Analyze the source dataset to understand structure and find training data"""
        print("🔍 Analyzing dataset structure...")
        
        # Look for training images and labels
        possible_paths = [
            self.source_path / "train",
            self.source_path / "training", 
            self.source_path / "atlas_train",
            self.source_path / "images",
            self.source_path
        ]
        
        found_structure = {}
        for path in possible_paths:
            if path.exists():
                print(f"Found: {path}")
                for item in path.iterdir():
                    if item.is_dir():
                        count = len(list(item.iterdir())) if item.is_dir() else 0
                        print(f"  - {item.name}/: {count} items")
                        found_structure[str(item)] = count
                    elif item.suffix in ['.jpg', '.png', '.txt']:
                        print(f"  - {item.name}")
        
        return found_structure

    def load_yolo_annotations(self, label_file):
        """Load YOLO format annotations from .txt file"""
        annotations = []
        if not label_file.exists():
            return annotations
            
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return annotations
            for line in content.split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        annotations.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
        return annotations

    def analyze_image_and_annotations(self, image_path, annotation_path):
        """Analyze single image and its annotations for statistics"""
        try:
            # Load image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                self.stats['image_sizes'].append((img_width, img_height))
            
            # Load annotations
            annotations = self.load_yolo_annotations(annotation_path)
            self.stats['total_annotations'] += len(annotations)
            
            for ann in annotations:
                self.stats['class_distribution'][ann['class_id']] += 1
                
                # Convert normalized coordinates to absolute for size analysis
                abs_width = ann['width'] * img_width
                abs_height = ann['height'] * img_height
                self.stats['bbox_sizes'].append((abs_width, abs_height))
                
            return True, len(annotations)
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return False, 0

    def create_stratified_split(self, image_annotation_pairs):
        """
        Create stratified train/validation split ensuring class balance
        
        Args:
            image_annotation_pairs: List of (image_path, label_path, annotations) tuples
        
        Returns:
            train_pairs, val_pairs: Split data with balanced class representation
        """
        print("📊 Creating stratified train/validation split...")
        
        # Group images by their class combinations
        class_groups = defaultdict(list)
        
        for img_path, label_path, annotations in image_annotation_pairs:
            # Create signature of classes present in this image
            classes_in_image = sorted(set(ann['class_id'] for ann in annotations))
            class_signature = '_'.join(map(str, classes_in_image)) if classes_in_image else 'empty'
            class_groups[class_signature].append((img_path, label_path, annotations))
        
        train_pairs = []
        val_pairs = []
        
        # Split each group maintaining the ratio
        for signature, pairs in class_groups.items():
            random.shuffle(pairs)  # Shuffle for randomness
            split_idx = int(len(pairs) * self.split_ratio)
            
            train_pairs.extend(pairs[:split_idx])
            val_pairs.extend(pairs[split_idx:])
            
            print(f"Class group '{signature}': {len(pairs)} images -> "
                  f"{len(pairs[:split_idx])} train, {len(pairs[split_idx:])} val")
        
        # Final shuffle
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
        
        print(f"✅ Split complete: {len(train_pairs)} train, {len(val_pairs)} val images")
        return train_pairs, val_pairs

    def copy_image_and_label(self, src_img_path, src_label_path, dst_img_dir, dst_label_dir, filename):
        """Copy image and label files to destination with new filename"""
        try:
            # Copy image
            dst_img_path = dst_img_dir / f"{filename}.jpg"
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy label if exists
            if src_label_path.exists():
                dst_label_path = dst_label_dir / f"{filename}.txt"
                shutil.copy2(src_label_path, dst_label_path)
            else:
                # Create empty label file if no annotations
                dst_label_path = dst_label_dir / f"{filename}.txt"
                dst_label_path.touch()
            
            return True
        except Exception as e:
            print(f"❌ Error copying {src_img_path}: {e}")
            return False

    def process_dataset(self, images_dir, labels_dir):
        """
        Main processing function: analyze, split, and organize dataset
        
        Args:
            images_dir: Directory containing training images
            labels_dir: Directory containing YOLO format label files
        """
        print("🚀 Starting dataset processing...")
        
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        # Find all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"📁 Found {len(image_files)} image files")
        
        # Process each image and collect valid pairs
        valid_pairs = []
        processed_count = 0
        
        for img_path in image_files:
            # Find corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # Analyze image and annotations
            success, ann_count = self.analyze_image_and_annotations(img_path, label_path)
            
            if success:
                annotations = self.load_yolo_annotations(label_path)
                valid_pairs.append((img_path, label_path, annotations))
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count}/{len(image_files)} images...")
        
        self.stats['total_images'] = len(valid_pairs)
        print(f"✅ Successfully processed {len(valid_pairs)} valid image-annotation pairs")
        
        # Create stratified split
        train_pairs, val_pairs = self.create_stratified_split(valid_pairs)
        
        # Copy files to output structure
        print("📁 Copying files to YOLO format structure...")
        
        train_copied = 0
        for i, (img_path, label_path, _) in enumerate(train_pairs):
            filename = f"train_{i:06d}"
            if self.copy_image_and_label(img_path, label_path, 
                                       self.output_path / "images" / "train",
                                       self.output_path / "labels" / "train",
                                       filename):
                train_copied += 1
        
        val_copied = 0
        for i, (img_path, label_path, _) in enumerate(val_pairs):
            filename = f"val_{i:06d}"
            if self.copy_image_and_label(img_path, label_path,
                                       self.output_path / "images" / "val", 
                                       self.output_path / "labels" / "val",
                                       filename):
                val_copied += 1
        
        print(f"✅ Copied {train_copied} training and {val_copied} validation samples")
        
        # Generate statistics report
        self.generate_statistics_report()
        
        # Update YAML configuration
        self.update_yaml_config()
        
        return {
            'train_count': train_copied,
            'val_count': val_copied,
            'total_annotations': self.stats['total_annotations'],
            'class_distribution': dict(self.stats['class_distribution'])
        }

    def generate_statistics_report(self):
        """Generate comprehensive dataset statistics"""
        print("📊 Generating dataset statistics...")
        
        # Class distribution analysis
        total_annotations = sum(self.stats['class_distribution'].values())
        class_percentages = {
            class_id: (count / total_annotations * 100) if total_annotations > 0 else 0
            for class_id, count in self.stats['class_distribution'].items()
        }
        
        # Image size analysis
        if self.stats['image_sizes']:
            img_widths, img_heights = zip(*self.stats['image_sizes'])
            avg_width = np.mean(img_widths)
            avg_height = np.mean(img_heights)
            
        # Bounding box size analysis
        if self.stats['bbox_sizes']:
            bbox_widths, bbox_heights = zip(*self.stats['bbox_sizes'])
            avg_bbox_width = np.mean(bbox_widths)
            avg_bbox_height = np.mean(bbox_heights)
        
        # Create statistics report
        stats_report = {
            'dataset_summary': {
                'total_images': self.stats['total_images'],
                'total_annotations': self.stats['total_annotations'],
                'avg_annotations_per_image': self.stats['total_annotations'] / max(self.stats['total_images'], 1),
                'train_split': self.split_ratio,
                'val_split': 1 - self.split_ratio
            },
            'image_statistics': {
                'avg_width': float(avg_width) if self.stats['image_sizes'] else 0,
                'avg_height': float(avg_height) if self.stats['image_sizes'] else 0,
                'total_unique_sizes': len(set(self.stats['image_sizes']))
            },
            'bbox_statistics': {
                'avg_width': float(avg_bbox_width) if self.stats['bbox_sizes'] else 0,
                'avg_height': float(avg_bbox_height) if self.stats['bbox_sizes'] else 0,
                'total_bboxes': len(self.stats['bbox_sizes'])
            },
            'class_distribution': {
                'counts': dict(self.stats['class_distribution']),
                'percentages': class_percentages,
                'class_names': self.class_names
            }
        }
        
        # Save statistics to JSON
        stats_file = self.output_path / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_report, f, indent=2)
        
        # Print summary
        print(f"\n📈 DATASET STATISTICS SUMMARY")
        print(f"{'='*50}")
        print(f"Total Images: {self.stats['total_images']}")
        print(f"Total Annotations: {self.stats['total_annotations']}")
        print(f"Avg Annotations/Image: {self.stats['total_annotations'] / max(self.stats['total_images'], 1):.2f}")
        
        if self.stats['image_sizes']:
            print(f"Avg Image Size: {avg_width:.0f}x{avg_height:.0f}")
        
        if self.stats['bbox_sizes']:
            print(f"Avg BBox Size: {avg_bbox_width:.1f}x{avg_bbox_height:.1f}")
        
        print(f"\n🎯 CLASS DISTRIBUTION (Top 10):")
        sorted_classes = sorted(self.stats['class_distribution'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for class_id, count in sorted_classes:
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            percentage = class_percentages.get(class_id, 0)
            print(f"  {class_id:2d} ({class_name:25s}): {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n💾 Statistics saved to: {stats_file}")
        
        return stats_report

    def update_yaml_config(self):
        """Update the YAML configuration file with correct paths"""
        yaml_content = f"""# Traffic Light Detection Dataset Configuration
# ATLAS Dataset - 25 Traffic Light Classes - Auto-generated

# Dataset paths
path: {self.output_path.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 25

# Class names (exactly as specified in assignment)
names:
"""
        
        for class_id, class_name in self.class_names.items():
            yaml_content += f"  {class_id}: {class_name}\n"
        
        yaml_content += f"""
# Dataset Statistics (Auto-generated)
# Total Images: {self.stats['total_images']}
# Total Annotations: {self.stats['total_annotations']}
# Train/Val Split: {self.split_ratio:.1f}/{1-self.split_ratio:.1f}

# Training hyperparameters optimized for traffic light detection
hsv_h: 0.015        # HSV-Hue augmentation (conservative for small objects)
hsv_s: 0.7          # HSV-Saturation augmentation  
hsv_v: 0.4          # HSV-Value augmentation
degrees: 10         # Image rotation (+/- deg)
translate: 0.1      # Image translation (+/- fraction)
scale: 0.5          # Image scale (+/- gain)
fliplr: 0.5         # Image flip left-right (probability)
flipud: 0.0         # Image flip up-down (disabled for traffic lights)
mosaic: 1.0         # Image mosaic (probability)
mixup: 0.1          # Image mixup (probability) - light mixing
"""
        
        yaml_file = Path("traffic_lights.yaml")
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Updated YAML configuration: {yaml_file}")

    def create_visualization_samples(self, num_samples=10):
        """Create visualization of sample images with annotations"""
        print(f"🎨 Creating visualization of {num_samples} sample images...")
        
        # Get sample images from train set
        train_images = list((self.output_path / "images" / "train").glob("*.jpg"))[:num_samples]
        
        if not train_images:
            print("❌ No training images found for visualization")
            return
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, img_path in enumerate(train_images):
            if i >= num_samples:
                break
            
            # Load image
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Load annotations
            label_path = self.output_path / "labels" / "train" / f"{img_path.stem}.txt"
            annotations = self.load_yolo_annotations(label_path)
            
            # Plot image
            axes[i].imshow(img)
            axes[i].set_title(f"{img_path.stem}\n{len(annotations)} objects", fontsize=10)
            axes[i].axis('off')
            
            # Draw bounding boxes
            for ann in annotations:
                # Convert normalized to absolute coordinates
                x_center = ann['x_center'] * img_width
                y_center = ann['y_center'] * img_height
                width = ann['width'] * img_width
                height = ann['height'] * img_height
                
                # Calculate corner coordinates
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), width, height, 
                                   fill=False, color='red', linewidth=2)
                axes[i].add_patch(rect)
                
                # Add class label
                class_name = self.class_names.get(ann['class_id'], f"cls_{ann['class_id']}")
                axes[i].text(x1, y1-5, f"{ann['class_id']}:{class_name[:10]}", 
                           color='red', fontsize=8, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        viz_path = self.output_path / "sample_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to: {viz_path}")

def main():
    """Main execution function"""
    print("🚦 Traffic Light Detection - Data Processing Pipeline")
    print("="*60)
    
    # Initialize processor
    processor = TrafficLightDataProcessor(
        source_data_path="./ML2-tld-colorization-dataset",
        output_path="./atlas_dataset",
        split_ratio=0.8  # 80% train, 20% validation
    )
    
    # Set random seed for reproducible splits
    random.seed(42)
    np.random.seed(42)
    
    # First, analyze the dataset structure to find training data
    print("🔍 Step 1: Analyzing dataset structure...")
    structure = processor.analyze_dataset_structure()
    
    # Note: You'll need to specify the correct paths based on the actual structure
    # This is a template - adjust paths based on what's found
    print("\n⚠️  MANUAL INPUT REQUIRED:")
    print("Please specify the paths to your training images and labels directories")
    print("Example paths to check:")
    print("  - Images: ./ML2-tld-colorization-dataset/train/images")
    print("  - Labels: ./ML2-tld-colorization-dataset/train/labels")
    
    # For now, return the structure analysis
    return structure

if __name__ == "__main__":
    main()