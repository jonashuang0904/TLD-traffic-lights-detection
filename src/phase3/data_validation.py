#!/usr/bin/env python3
"""
Traffic Light Detection - Data Pipeline Validation
Phase 3: Comprehensive validation of data processing and augmentation
"""

import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import cv2

class DataPipelineValidator:
    """
    Comprehensive validation system for traffic light detection data pipeline
    """
    
    def __init__(self, dataset_path="./atlas_dataset"):
        self.dataset_path = Path(dataset_path)
        self.class_names = {
            0: "circle_green", 1: "circle_red", 2: "off", 3: "circle_red_yellow",
            4: "arrow_left_green", 5: "circle_yellow", 6: "arrow_right_red",
            7: "arrow_left_red", 8: "arrow_straight_red", 9: "arrow_left_red_yellow",
            10: "arrow_left_yellow", 11: "arrow_straight_yellow", 12: "arrow_right_red_yellow",
            13: "arrow_right_green", 14: "arrow_right_yellow", 15: "arrow_straight_green",
            16: "arrow_straight_left_green", 17: "arrow_straight_red_yellow",
            18: "arrow_straight_left_red", 19: "arrow_straight_left_yellow",
            20: "arrow_straight_left_red_yellow", 21: "arrow_straight_right_red",
            22: "arrow_straight_right_red_yellow", 23: "arrow_straight_right_yellow",
            24: "arrow_straight_right_green"
        }
        
        self.validation_results = {
            'file_structure': {},
            'annotation_integrity': {},
            'class_distribution': {},
            'image_properties': {},
            'bbox_quality': {},
            'augmentation_tests': {}
        }

    def validate_file_structure(self):
        """Validate YOLO dataset file structure"""
        print("📁 Validating file structure...")
        
        required_dirs = [
            "images/train", "images/val",
            "labels/train", "labels/val"
        ]
        
        structure_valid = True
        for req_dir in required_dirs:
            dir_path = self.dataset_path / req_dir
            exists = dir_path.exists()
            if exists:
                file_count = len(list(dir_path.glob("*")))
                self.validation_results['file_structure'][req_dir] = {
                    'exists': True,
                    'file_count': file_count
                }
                print(f"  ✅ {req_dir}: {file_count} files")
            else:
                self.validation_results['file_structure'][req_dir] = {'exists': False}
                print(f"  ❌ {req_dir}: Missing")
                structure_valid = False
        
        return structure_valid

    def validate_image_label_pairs(self):
        """Validate that each image has corresponding label file"""
        print("🔗 Validating image-label pairs...")
        
        for split in ['train', 'val']:
            images_dir = self.dataset_path / "images" / split
            labels_dir = self.dataset_path / "labels" / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            image_files = set(f.stem for f in images_dir.glob("*.jpg"))
            label_files = set(f.stem for f in labels_dir.glob("*.txt"))
            
            matched_pairs = len(image_files & label_files)
            unmatched_images = len(image_files - label_files)
            unmatched_labels = len(label_files - image_files)
            
            self.validation_results['annotation_integrity'][split] = {
                'total_images': len(image_files),
                'total_labels': len(label_files),
                'matched_pairs': matched_pairs,
                'unmatched_images': unmatched_images,
                'unmatched_labels': unmatched_labels
            }
            
            print(f"  {split.upper()} SET:")
            print(f"    ✅ Matched pairs: {matched_pairs}")
            if unmatched_images > 0:
                print(f"    ⚠️  Unmatched images: {unmatched_images}")
            if unmatched_labels > 0:
                print(f"    ⚠️  Unmatched labels: {unmatched_labels}")

    def validate_annotation_format(self, max_files_to_check=100):
        """Validate YOLO annotation format correctness"""
        print("📋 Validating annotation format...")
        
        format_errors = []
        bbox_errors = []
        class_errors = []
        
        for split in ['train', 'val']:
            labels_dir = self.dataset_path / "labels" / split
            if not labels_dir.exists():
                continue
            
            label_files = list(labels_dir.glob("*.txt"))[:max_files_to_check]
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                    
                    if not content:  # Empty file is valid (no objects)
                        continue
                    
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if not line.strip():
                            continue
                        
                        parts = line.strip().split()
                        
                        # Check format: class_id x_center y_center width height
                        if len(parts) != 5:
                            format_errors.append(f"{label_file}:{line_num} - Wrong format: {len(parts)} values")
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                        except ValueError:
                            format_errors.append(f"{label_file}:{line_num} - Invalid number format")
                            continue
                        
                        # Validate class ID
                        if class_id < 0 or class_id > 24:
                            class_errors.append(f"{label_file}:{line_num} - Invalid class ID: {class_id}")
                        
                        # Validate bbox coordinates (should be normalized 0-1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                            bbox_errors.append(f"{label_file}:{line_num} - Center out of bounds: ({x_center}, {y_center})")
                        
                        if not (0 < width <= 1 and 0 < height <= 1):
                            bbox_errors.append(f"{label_file}:{line_num} - Size out of bounds: ({width}, {height})")
                
                except Exception as e:
                    format_errors.append(f"{label_file} - Read error: {e}")
        
        self.validation_results['annotation_integrity']['format_errors'] = len(format_errors)
        self.validation_results['annotation_integrity']['bbox_errors'] = len(bbox_errors)
        self.validation_results['annotation_integrity']['class_errors'] = len(class_errors)
        
        print(f"  ✅ Format errors: {len(format_errors)}")
        print(f"  ✅ Bbox errors: {len(bbox_errors)}")
        print(f"  ✅ Class errors: {len(class_errors)}")
        
        if format_errors:
            print("  ⚠️  First few format errors:")
            for error in format_errors[:3]:
                print(f"    - {error}")
        
        return len(format_errors) == 0 and len(bbox_errors) == 0 and len(class_errors) == 0

    def analyze_class_distribution(self):
        """Analyze class distribution across train/val splits"""
        print("📊 Analyzing class distribution...")
        
        class_counts = {'train': Counter(), 'val': Counter()}
        
        for split in ['train', 'val']:
            labels_dir = self.dataset_path / "labels" / split
            if not labels_dir.exists():
                continue
            
            for label_file in labels_dir.glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.strip().split()[0])
                                class_counts[split][class_id] += 1
                except Exception as e:
                    print(f"  ⚠️  Error reading {label_file}: {e}")
        
        # Calculate statistics
        total_train = sum(class_counts['train'].values())
        total_val = sum(class_counts['val'].values())
        
        self.validation_results['class_distribution'] = {
            'train_total': total_train,
            'val_total': total_val,
            'train_counts': dict(class_counts['train']),
            'val_counts': dict(class_counts['val'])
        }
        
        print(f"  📈 Training annotations: {total_train}")
        print(f"  📈 Validation annotations: {total_val}")
        
        # Check for class imbalance
        if total_train > 0:
            print(f"  🎯 Top 5 classes in training:")
            for class_id, count in class_counts['train'].most_common(5):
                percentage = count / total_train * 100
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                print(f"    {class_id:2d} ({class_name[:20]:20s}): {count:4d} ({percentage:5.1f}%)")
        
        # Check for missing classes
        all_classes = set(range(25))
        present_classes = set(class_counts['train'].keys()) | set(class_counts['val'].keys())
        missing_classes = all_classes - present_classes
        
        if missing_classes:
            print(f"  ⚠️  Missing classes: {sorted(missing_classes)}")
        else:
            print(f"  ✅ All 25 classes present")

    def validate_image_properties(self, sample_size=50):
        """Validate image properties and detect issues"""
        print("🖼️  Validating image properties...")
        
        image_stats = {
            'sizes': [],
            'corrupted': [],
            'formats': Counter(),
            'color_modes': Counter()
        }
        
        for split in ['train', 'val']:
            images_dir = self.dataset_path / "images" / split
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg"))[:sample_size]
            
            for img_file in image_files:
                try:
                    with Image.open(img_file) as img:
                        image_stats['sizes'].append(img.size)
                        image_stats['formats'][img.format] += 1
                        image_stats['color_modes'][img.mode] += 1
                except Exception as e:
                    image_stats['corrupted'].append(str(img_file))
                    print(f"  ❌ Corrupted image: {img_file} - {e}")
        
        if image_stats['sizes']:
            widths, heights = zip(*image_stats['sizes'])
            self.validation_results['image_properties'] = {
                'sample_size': len(image_stats['sizes']),
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_size': (min(widths), min(heights)),
                'max_size': (max(widths), max(heights)),
                'corrupted_count': len(image_stats['corrupted']),
                'formats': dict(image_stats['formats']),
                'color_modes': dict(image_stats['color_modes'])
            }
            
            print(f"  📐 Average size: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
            print(f"  📐 Size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
            print(f"  🎨 Formats: {dict(image_stats['formats'])}")
            print(f"  🎨 Color modes: {dict(image_stats['color_modes'])}")
            print(f"  ❌ Corrupted images: {len(image_stats['corrupted'])}")

    def validate_bbox_quality(self, sample_size=20):
        """Validate bounding box quality and detect potential issues"""
        print("📦 Validating bounding box quality...")
        
        bbox_stats = {
            'areas': [],
            'aspect_ratios': [],
            'tiny_boxes': [],
            'large_boxes': [],
            'edge_boxes': []
        }
        
        for split in ['train', 'val']:
            images_dir = self.dataset_path / "images" / split
            labels_dir = self.dataset_path / "labels" / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg"))[:sample_size]
            
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                try:
                    # Get image dimensions
                    with Image.open(img_file) as img:
                        img_width, img_height = img.size
                    
                    # Read annotations
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        _, x_center, y_center, width, height = map(float, parts[:5])
                                        
                                        # Calculate absolute dimensions
                                        abs_width = width * img_width
                                        abs_height = height * img_height
                                        area = abs_width * abs_height
                                        aspect_ratio = abs_width / abs_height if abs_height > 0 else 0
                                        
                                        bbox_stats['areas'].append(area)
                                        bbox_stats['aspect_ratios'].append(aspect_ratio)
                                        
                                        # Check for potential issues
                                        if area < 100:  # Very small objects
                                            bbox_stats['tiny_boxes'].append((img_file.name, area))
                                        
                                        if area > 50000:  # Very large objects
                                            bbox_stats['large_boxes'].append((img_file.name, area))
                                        
                                        # Check if bbox is near image edges
                                        x1 = x_center - width/2
                                        y1 = y_center - height/2
                                        x2 = x_center + width/2
                                        y2 = y_center + height/2
                                        
                                        if x1 < 0.01 or y1 < 0.01 or x2 > 0.99 or y2 > 0.99:
                                            bbox_stats['edge_boxes'].append((img_file.name, (x1, y1, x2, y2)))
                
                except Exception as e:
                    print(f"  ⚠️  Error processing {img_file}: {e}")
        
        if bbox_stats['areas']:
            self.validation_results['bbox_quality'] = {
                'total_boxes': len(bbox_stats['areas']),
                'avg_area': np.mean(bbox_stats['areas']),
                'min_area': min(bbox_stats['areas']),
                'max_area': max(bbox_stats['areas']),
                'avg_aspect_ratio': np.mean(bbox_stats['aspect_ratios']),
                'tiny_boxes_count': len(bbox_stats['tiny_boxes']),
                'large_boxes_count': len(bbox_stats['large_boxes']),
                'edge_boxes_count': len(bbox_stats['edge_boxes'])
            }
            
            print(f"  📊 Total bounding boxes analyzed: {len(bbox_stats['areas'])}")
            print(f"  📏 Average area: {np.mean(bbox_stats['areas']):.1f} pixels²")
            print(f"  📏 Area range: {min(bbox_stats['areas']):.1f} - {max(bbox_stats['areas']):.1f}")
            print(f"  📐 Average aspect ratio: {np.mean(bbox_stats['aspect_ratios']):.2f}")
            print(f"  ⚠️  Tiny boxes (<100px²): {len(bbox_stats['tiny_boxes'])}")
            print(f"  ⚠️  Large boxes (>50k px²): {len(bbox_stats['large_boxes'])}")
            print(f"  ⚠️  Edge boxes: {len(bbox_stats['edge_boxes'])}")

    def create_validation_visualization(self, num_samples=6):
        """Create visualization of validation results"""
        print("🎨 Creating validation visualization...")
        
        # Sample images from training set
        train_images_dir = self.dataset_path / "images" / "train"
        train_labels_dir = self.dataset_path / "labels" / "train"
        
        if not train_images_dir.exists():
            print("  ❌ No training images found for visualization")
            return
        
        image_files = list(train_images_dir.glob("*.jpg"))[:num_samples]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, img_file in enumerate(image_files):
            if i >= num_samples:
                break
            
            # Load image
            img = Image.open(img_file)
            axes[i].imshow(img)
            axes[i].set_title(f"{img_file.stem}", fontsize=10)
            axes[i].axis('off')
            
            # Load and draw annotations
            label_file = train_labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    bbox_count = 0
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # Convert to pixel coordinates
                                img_width, img_height = img.size
                                x1 = (x_center - width/2) * img_width
                                y1 = (y_center - height/2) * img_height
                                box_width = width * img_width
                                box_height = height * img_height
                                
                                # Draw rectangle
                                rect = patches.Rectangle(
                                    (x1, y1), box_width, box_height,
                                    linewidth=2, edgecolor='red', facecolor='none'
                                )
                                axes[i].add_patch(rect)
                                
                                # Add class label
                                class_name = self.class_names.get(class_id, f"cls_{class_id}")
                                axes[i].text(x1, y1-5, f"{class_id}:{class_name[:8]}", 
                                           color='red', fontsize=8, weight='bold',
                                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                                bbox_count += 1
                    
                    axes[i].set_title(f"{img_file.stem}\n{bbox_count} objects", fontsize=9)
        
        plt.tight_layout()
        viz_path = self.dataset_path / "validation_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Validation visualization saved: {viz_path}")

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("📋 Generating validation report...")
        
        report = {
            'validation_summary': {
                'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A',
                'dataset_path': str(self.dataset_path),
                'total_classes': 25
            },
            'results': self.validation_results
        }
        
        # Calculate overall health score
        health_score = 100
        
        # Deduct points for issues
        if self.validation_results.get('annotation_integrity', {}).get('format_errors', 0) > 0:
            health_score -= 20
        
        if self.validation_results.get('annotation_integrity', {}).get('bbox_errors', 0) > 0:
            health_score -= 15
        
        if self.validation_results.get('image_properties', {}).get('corrupted_count', 0) > 0:
            health_score -= 10
        
        # Check for severe class imbalance
        train_counts = self.validation_results.get('class_distribution', {}).get('train_counts', {})
        if train_counts:
            max_count = max(train_counts.values())
            min_count = min(train_counts.values())
            if max_count / max(min_count, 1) > 100:  # 100:1 ratio
                health_score -= 25
        
        report['validation_summary']['health_score'] = max(0, health_score)
        
        # Save report
        report_path = self.dataset_path / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✅ Validation report saved: {report_path}")
        print(f"  🏥 Dataset health score: {health_score}/100")
        
        return report

    def run_full_validation(self):
        """Run complete validation pipeline"""
        print("🚦 Starting comprehensive data validation...")
        print("="*60)
        
        try:
            # Step 1: File structure
            structure_valid = self.validate_file_structure()
            
            # Step 2: Image-label pairs
            self.validate_image_label_pairs()
            
            # Step 3: Annotation format
            format_valid = self.validate_annotation_format()
            
            # Step 4: Class distribution
            self.analyze_class_distribution()
            
            # Step 5: Image properties
            self.validate_image_properties()
            
            # Step 6: Bounding box quality
            self.validate_bbox_quality()
            
            # Step 7: Create visualization
            self.create_validation_visualization()
            
            # Step 8: Generate report
            report = self.generate_validation_report()
            
            print("\n🎉 Validation complete!")
            print(f"Dataset health score: {report['validation_summary']['health_score']}/100")
            
            return report
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None

def main():
    """Main execution function"""
    print("🔍 Traffic Light Detection - Data Validation Pipeline")
    print("="*60)
    
    # Initialize validator
    validator = DataPipelineValidator(dataset_path="./atlas_dataset")
    
    # Run full validation
    results = validator.run_full_validation()
    
    if results:
        print("\n✅ Validation completed successfully!")
        print("Review the validation_report.json for detailed results.")
    else:
        print("\n❌ Validation failed. Check the error messages above.")

if __name__ == "__main__":
    main()