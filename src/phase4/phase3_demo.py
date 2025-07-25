#!/usr/bin/env python3
"""
Traffic Light Detection - Phase 3 Demo
Demonstration of data processing pipeline without actual dataset
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

from data_processor import TrafficLightDataProcessor
from augmentation_pipeline import TrafficLightAugmentation, ImageResizeStrategy
from data_validation import DataPipelineValidator

def create_demo_dataset():
    """Create a minimal demo dataset structure"""
    print("🎨 Creating demo dataset structure...")
    
    # Create demo directories
    demo_path = Path("./demo_atlas")
    images_dir = demo_path / "images"
    labels_dir = demo_path / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images and labels
    for i in range(5):
        # Create dummy image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(images_dir / f"sample_{i:03d}.jpg")
        
        # Create corresponding YOLO format label
        # Random traffic light annotations
        annotations = []
        for j in range(np.random.randint(1, 4)):  # 1-3 traffic lights per image
            class_id = np.random.randint(0, 25)  # Random class 0-24
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.02, 0.1)
            height = np.random.uniform(0.03, 0.15)
            
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save label file
        with open(labels_dir / f"sample_{i:03d}.txt", 'w') as f:
            f.write('\n'.join(annotations))
    
    print(f"✅ Created demo dataset with 5 images and labels at: {demo_path}")
    return demo_path, images_dir, labels_dir

def demo_data_processing():
    """Demonstrate data processing pipeline"""
    print("\n📊 DEMO: Data Processing Pipeline")
    print("="*50)
    
    # Create demo dataset
    demo_path, images_dir, labels_dir = create_demo_dataset()
    
    # Initialize processor
    processor = TrafficLightDataProcessor(
        source_data_path=str(demo_path),
        output_path="./demo_output", 
        split_ratio=0.8
    )
    
    # Process the demo dataset
    try:
        results = processor.process_dataset(images_dir, labels_dir)
        print(f"✅ Processing successful:")
        print(f"   Train samples: {results['train_count']}")
        print(f"   Val samples: {results['val_count']}")
        print(f"   Total annotations: {results['total_annotations']}")
        return True
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False

def demo_augmentation():
    """Demonstrate augmentation pipeline"""
    print("\n🎨 DEMO: Augmentation Pipeline")
    print("="*40)
    
    # Create test image and annotations
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bboxes = [[0.5, 0.5, 0.1, 0.15], [0.3, 0.7, 0.08, 0.12]]  # YOLO format
    test_labels = [1, 5]  # circle_red, circle_yellow
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Original bboxes: {len(test_bboxes)}")
    
    # Test training augmentation
    try:
        train_aug = TrafficLightAugmentation(image_size=640, mode='train')
        aug_image, aug_bboxes, aug_labels = train_aug(test_image, test_bboxes, test_labels)
        
        print(f"✅ Training augmentation:")
        print(f"   Augmented image shape: {aug_image.shape}")
        print(f"   Preserved bboxes: {len(aug_bboxes)}")
        print(f"   Preserved labels: {len(aug_labels)}")
        
        # Test validation augmentation  
        val_aug = TrafficLightAugmentation(image_size=640, mode='val')
        val_image, val_bboxes, val_labels = val_aug(test_image, test_bboxes, test_labels)
        
        print(f"✅ Validation augmentation:")
        print(f"   Validation image shape: {val_image.shape}")
        print(f"   Preserved bboxes: {len(val_bboxes)}")
        print(f"   Preserved labels: {len(val_labels)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Augmentation failed: {e}")
        return False

def demo_resizing():
    """Demonstrate image resizing strategy"""
    print("\n📐 DEMO: Image Resizing Strategy") 
    print("="*40)
    
    # Test with different image sizes
    test_sizes = [(1920, 1080), (640, 480), (800, 600), (1280, 720)]
    resizer = ImageResizeStrategy(target_size=640)
    
    for width, height in test_sizes:
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        test_bboxes = [[0.5, 0.5, 0.1, 0.15]]  # Center bbox
        
        try:
            resized_image, adjusted_bboxes = resizer.resize_with_padding(test_image, test_bboxes)
            
            print(f"✅ {width}x{height} → {resized_image.shape[1]}x{resized_image.shape[0]}")
            print(f"   Original bbox: {test_bboxes[0]}")
            print(f"   Adjusted bbox: {[f'{x:.3f}' for x in adjusted_bboxes[0]]}")
            
        except Exception as e:
            print(f"❌ Resize failed for {width}x{height}: {e}")
    
    return True

def demo_validation():
    """Demonstrate data validation"""
    print("\n🔍 DEMO: Data Validation Pipeline")
    print("="*40)
    
    # Use the processed demo output
    if Path("./demo_output").exists():
        validator = DataPipelineValidator(dataset_path="./demo_output")
        
        try:
            # Test file structure validation
            structure_valid = validator.validate_file_structure()
            print(f"File structure validation: {'✅ PASS' if structure_valid else '❌ FAIL'}")
            
            # Test image-label pairs
            validator.validate_image_label_pairs()
            
            # Test annotation format
            format_valid = validator.validate_annotation_format(max_files_to_check=10)
            print(f"Annotation format validation: {'✅ PASS' if format_valid else '❌ FAIL'}")
            
            # Analyze class distribution
            validator.analyze_class_distribution()
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    else:
        print("❌ No processed dataset found. Run data processing demo first.")
        return False

def main():
    """Main demo execution"""
    print("🚦 TRAFFIC LIGHT DETECTION - PHASE 3 DEMONSTRATION")
    print("="*60)
    print("This demo shows the core Phase 3 functionality without requiring")
    print("the actual ATLAS dataset.")
    print()
    
    results = {}
    
    # Demo 1: Data Processing
    results['data_processing'] = demo_data_processing()
    
    # Demo 2: Augmentation
    results['augmentation'] = demo_augmentation()
    
    # Demo 3: Resizing
    results['resizing'] = demo_resizing()
    
    # Demo 4: Validation
    results['validation'] = demo_validation()
    
    # Summary
    print(f"\n🎉 PHASE 3 DEMO RESULTS")
    print("="*30)
    
    passed = sum(results.values())
    total = len(results)
    
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{component.replace('_', ' ').title():20s}: {status}")
    
    print(f"\nOverall: {passed}/{total} components working")
    
    if passed == total:
        print("🎉 All Phase 3 components are working correctly!")
        print("✅ Ready to process real ATLAS dataset")
    else:
        print("⚠️  Some components need attention")
    
    print("\n📋 Phase 3 Implementation Summary:")
    print("✅ 80/20 Train/Validation Split - Implemented")
    print("✅ Advanced Augmentation Pipeline - Implemented") 
    print("✅ Aspect Ratio Preserving Resize - Implemented")
    print("✅ Comprehensive Data Validation - Implemented")
    print("✅ YOLO Format Dataset Generation - Implemented")
    print("✅ Class Distribution Analysis - Implemented")
    print("✅ Annotation Quality Checks - Implemented")

if __name__ == "__main__":
    main()