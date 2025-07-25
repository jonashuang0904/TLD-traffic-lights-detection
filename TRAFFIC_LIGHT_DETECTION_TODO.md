# 🚦 Traffic Light Detection - Detailed TODO Plan

## Project Overview
**Task**: Create a 2D object detector for traffic lights with 25 classes  
**Target**: F1 Score > 40 for 3 bonus points  
**Evaluation**: F1 = (2 × TP) / (2 × TP + FP + FN) with IoU > 0.5  
**Dataset**: ATLAS dataset (YOLO format) + 425 test images  

---

## 🔍 **PHASE 1: Project Setup & Dataset Analysis**
*Priority: HIGH | Timeline: Week 1*

### Core Tasks
- [ ] **Download and extract ATLAS dataset** from https://bwsyncandshare.kit.edu/s/KcigioLJHmaKD8n
- [ ] **Explore dataset structure**: understand YOLO format (.txt annotations with class_id, x_center, y_center, width, height)
- [ ] **Analyze class distribution** across 25 traffic light categories (0-24) to identify potential class imbalance
- [ ] **Examine image characteristics**: sizes, aspect ratios, lighting conditions, camera angles, image quality
- [ ] **Validate annotation quality**: spot-check bounding boxes, verify class labels match visual inspection
- [ ] **Create data visualization script** to display sample images with annotations for each of the 25 classes

### Critical Success Factors
- **Class imbalance analysis** will determine if you need weighted sampling or focal loss
- **Image quality assessment** determines augmentation strategy
- **Annotation quality check** prevents garbage in, garbage out scenarios

### Expected Findings
- Severe class imbalance (circle_red/green common, complex arrows rare)
- Lighting conditions vary dramatically (day/night/sunset/overcast)
- Scale variation - distant vs close traffic lights
- Potential occlusion issues - partially blocked by vehicles/poles

---

## 🏗️ **PHASE 2: Model Architecture Selection & Environment Setup**
*Priority: HIGH | Timeline: Week 1*

### Core Tasks
- [ ] **Research and compare** YOLO variants (YOLOv8, YOLOv11) vs RT-DETR for traffic light detection performance
- [ ] **Set up development environment**: install Ultralytics, PyTorch, required dependencies
- [ ] **Choose pre-trained model weights**: COCO-pretrained YOLO model as starting point for transfer learning
- [ ] **Configure GPU environment** (local GPU or Google Colab) for efficient training
- [ ] **Create YOLO-compatible dataset configuration file** (data.yaml) with class names and paths

### Strategic Decision: YOLO vs RT-DETR
- **YOLOv8/v11**: Proven for real-time detection, excellent Ultralytics ecosystem
- **RT-DETR**: Newer transformer approach, potentially better accuracy but slower
- **Recommendation**: Start with **YOLOv8n/s** for speed, upgrade to **YOLOv8m/l** if needed

### Environment Setup Checklist
```bash
pip install ultralytics torch torchvision matplotlib pandas opencv-python
# For Google Colab users
from google.colab import drive
drive.mount('/content/drive')
```

---

## 📊 **PHASE 3: Data Preprocessing & Augmentation Strategy**
*Priority: HIGH | Timeline: Week 1-2*

### Core Tasks
- [ ] **Implement train/validation split strategy**: 80/20 or 85/15, ensuring balanced class representation
- [ ] **Design augmentation pipeline**: consider brightness/contrast (traffic lights vary in lighting), horizontal flip, rotation, noise
- [ ] **Handle image resizing strategy**: maintain aspect ratio, pad vs crop, optimal input resolution for model
- [ ] **Validate data pipeline**: ensure annotations remain correct after augmentations and resizing

### Critical Insight: Small Object Detection
Traffic lights are **small objects** (typically <5% of image area)
- **Augmentation balance**: Too aggressive → lose small object details
- **Resolution strategy**: Higher input size (640→1280) may be crucial for distant lights
- **Train/Val split**: Ensure similar **lighting conditions** and **traffic light sizes** in both sets

### Recommended Augmentation Strategy
```yaml
# Conservative augmentation for small objects
hsv_h: 0.015    # Hue augmentation
hsv_s: 0.7      # Saturation augmentation  
hsv_v: 0.4      # Value augmentation
degrees: 10     # Rotation degrees (conservative)
translate: 0.1  # Translation fraction
scale: 0.5      # Scale augmentation
flipud: 0.0     # Vertical flip (not recommended for traffic lights)
fliplr: 0.5     # Horizontal flip
mosaic: 1.0     # Mosaic augmentation
```

---

## 🎯 **PHASE 4: Training Pipeline Implementation**
*Priority: HIGH | Timeline: Week 2*

### Core Tasks
- [ ] **Configure training hyperparameters**: learning rate (start with 0.01), batch size (based on GPU memory), epochs (100-200)
- [ ] **Set up optimizer** (AdamW recommended) and learning rate scheduler (cosine annealing or step decay)
- [ ] **Implement training monitoring**: track loss curves, mAP, precision, recall metrics during training
- [ ] **Configure model checkpointing**: save best model based on validation F1 score, not just mAP
- [ ] **Implement early stopping mechanism** to prevent overfitting
- [ ] **Start initial training run**: baseline model with default hyperparameters to establish performance floor

### Recommended Training Configuration
```python
# Training parameters
model = YOLO('yolov8s.pt')  # Start with small model
results = model.train(
    data='traffic_lights.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    lr0=0.01,
    optimizer='AdamW',
    patience=50,
    save_period=10
)
```

### Success Metrics to Track
- **Loss convergence**: Box, cls, dfl losses should decrease steadily
- **mAP@0.5**: Should reach >0.4 for good F1 performance
- **Precision/Recall balance**: Critical for F1 optimization

---

## 🔧 **PHASE 5: Model Optimization & Hyperparameter Tuning**
*Priority: MEDIUM | Timeline: Week 2-3*

### Core Tasks
- [ ] **Analyze initial training results**: identify overfitting, underfitting, class-specific performance issues
- [ ] **Tune confidence threshold**: find optimal balance between precision and recall for maximum F1 score
- [ ] **Optimize NMS parameters**: IoU threshold, max detections per image (traffic lights rarely overlap)
- [ ] **Experiment with different input resolutions**: balance between accuracy and inference speed
- [ ] **Fine-tune learning rate and batch size** based on convergence patterns
- [ ] **Consider advanced techniques**: focal loss for class imbalance, multi-scale training, test-time augmentation

### F1 Optimization Strategy
1. **Start conservative**: High confidence (0.7+) → measure precision
2. **Lower gradually**: Find precision/recall sweet spot
3. **NMS tuning**: Aggressive filtering since traffic lights don't overlap
4. **Per-class analysis**: Some classes may need different thresholds

### Advanced Optimization Techniques
```python
# Confidence threshold sweep
conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
best_f1 = 0
best_conf = 0.5

for conf in conf_thresholds:
    results = model.val(conf=conf, iou=0.5)
    f1_score = calculate_f1_from_results(results)
    if f1_score > best_f1:
        best_f1 = f1_score
        best_conf = conf
```

---

## 📈 **PHASE 6: Validation & Performance Analysis**
*Priority: HIGH | Timeline: Week 3*

### Core Tasks
- [ ] **Implement custom F1 score calculation** with IoU > 0.5 threshold (matching assignment evaluation criteria)
- [ ] **Create per-class performance analysis**: identify which traffic light types are hardest to detect
- [ ] **Analyze failure cases**: false positives, false negatives, misclassifications with visual examples
- [ ] **Validate model on held-out validation set**: ensure F1 score > 40 target for 3 bonus points
- [ ] **Cross-validate results**: multiple training runs to ensure consistent performance

### Critical Implementation: Assignment-Specific F1
```python
def calculate_assignment_f1(predictions, ground_truth, iou_threshold=0.5):
    """
    Calculate F1 score exactly as specified in assignment:
    - IoU threshold = 0.5 (not mAP@0.5:0.95)
    - Class prediction must be exact match
    - Coordinate format: center-based (x_center, y_center, width, height)
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Implementation details here...
    
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return f1, tp, fp, fn
```

### Performance Analysis Checklist
- **Per-class metrics**: Identify weak classes needing attention
- **Size-based analysis**: Performance on small vs large traffic lights
- **Lighting condition analysis**: Day vs night performance
- **Distance analysis**: Near vs far traffic light detection

---

## 🎯 **PHASE 7: Test Set Inference & Output Generation**
*Priority: HIGH | Timeline: Week 4*

### Core Tasks
- [ ] **Load 425 test images** from student_dataset and verify image paths/accessibility
- [ ] **Run batch inference** on test set using best trained model with optimized confidence/NMS thresholds
- [ ] **Convert model predictions to required CSV format**: ImageName, [x_center, y_center, width, height], confidence, class_id
- [ ] **Validate output format**: ensure coordinates are absolute pixels, confidence values are reasonable, class IDs are 0-24
- [ ] **Handle edge cases**: images with no detections (no CSV rows), multiple detections per image (multiple rows)
- [ ] **Create prediction.csv file** with exact format specified in assignment

### Critical Output Format
```csv
ImageName,xywh,Conf,Classification
1708418258713137499_front_medium.jpg,"[260.25, 265.0, 37.0, 104.0]",0.93408203125,1
1708418258713137499_front_medium.jpg,"[1313.0, 289.75, 34.0, 98.5]",0.93359375,1
```

### Coordinate Conversion
```python
def convert_predictions_to_csv(predictions, image_names):
    """
    Convert YOLO predictions to assignment CSV format
    - Input: normalized coordinates (0-1)
    - Output: absolute pixel coordinates
    - Format: [x_center, y_center, width, height]
    """
    csv_rows = []
    
    for i, pred in enumerate(predictions):
        img_name = image_names[i]
        img_height, img_width = get_image_dimensions(img_name)
        
        for detection in pred:
            x_center = detection[0] * img_width
            y_center = detection[1] * img_height  
            width = detection[2] * img_width
            height = detection[3] * img_height
            conf = detection[4]
            class_id = int(detection[5])
            
            csv_rows.append([
                img_name,
                f"[{x_center}, {y_center}, {width}, {height}]",
                conf,
                class_id
            ])
    
    return csv_rows
```

---

## 🧪 **PHASE 8: Testing & Validation of Outputs**
*Priority: HIGH | Timeline: Week 4*

### Core Tasks
- [ ] **Test prediction.csv on evaluation website** https://kitml2.streamlit.app/ to get preliminary F1 score
- [ ] **Debug any format issues**: coordinate ranges, data types, CSV structure problems
- [ ] **Visualize sample predictions**: overlay bounding boxes on test images to manually verify correctness
- [ ] **Iterate on confidence threshold** if F1 score is suboptimal: balance precision vs recall

### Common Failure Points & Solutions
| Issue | Symptoms | Solution |
|-------|----------|----------|
| Wrong coordinates | F1 = 0 despite visible detections | Check absolute vs normalized conversion |
| Format errors | Website upload fails | Validate CSV structure, quotes, delimiters |
| Class mapping | Low F1 despite good detections | Verify model class IDs map to 0-24 range |
| Threshold issues | Very high/low precision or recall | Systematic confidence threshold sweep |

### Debugging Visualization
```python
def visualize_predictions(image, predictions, save_path):
    """
    Overlay bounding boxes on images for manual verification
    """
    import cv2
    img = cv2.imread(image)
    
    for pred in predictions:
        x_center, y_center, width, height, conf, class_id = pred
        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2) 
        y2 = int(y_center + height/2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{class_id}:{conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(save_path, img)
```

---

## 📝 **PHASE 9: Documentation & Code Organization**
*Priority: MEDIUM | Timeline: Week 5*

### Core Tasks
- [ ] **Write explanation.txt**: chosen task, YOLOv8/v11 architecture, transfer learning approach, training details
- [ ] **Organize code files**: training script, inference script, data processing utilities, configuration files
- [ ] **Add code comments** explaining key decisions: model choice, hyperparameters, postprocessing steps
- [ ] **Create requirements.txt** with all dependencies and versions used

### Explanation.txt Template
```
Traffic Light Detection Project - ML2 Assignment

CHOSEN TASK: Traffic Light Detection (Option 2)

ML ARCHITECTURE:
- Model: YOLOv8s (Small variant for balance of speed and accuracy)
- Pre-training: COCO dataset weights as starting point
- Transfer Learning: Fine-tuned on ATLAS traffic light dataset
- Input Resolution: 640x640 pixels
- Output: 25-class traffic light detection

TRAINING APPROACH:
- Dataset: ATLAS dataset with 25 traffic light classes
- Train/Val Split: 80/20 with stratified sampling
- Augmentation: Conservative approach preserving small object details
- Optimizer: AdamW with cosine annealing learning rate schedule
- Training Duration: 150 epochs with early stopping (patience=50)
- Batch Size: 16 (limited by GPU memory)

OPTIMIZATION STRATEGY:
- Confidence Threshold Tuning: Systematic sweep from 0.1 to 0.8
- NMS IoU Threshold: 0.4 (aggressive since traffic lights rarely overlap)
- F1 Score Optimization: Custom implementation matching assignment criteria
- Class Imbalance: Handled through weighted sampling and focal loss

FINAL PERFORMANCE:
- Validation F1 Score: [TO BE FILLED]
- Optimal Confidence Threshold: [TO BE FILLED]
- Best Performing Classes: [TO BE FILLED]
- Challenging Classes: [TO BE FILLED]
```

### Code Organization Structure
```
traffic_light_detection/
├── train.py              # Training script
├── inference.py          # Test set inference
├── data_analysis.py      # Dataset exploration
├── evaluation.py         # F1 score calculation
├── visualization.py      # Prediction visualization
├── config.yaml           # Training configuration
├── requirements.txt      # Dependencies
└── utils/
    ├── data_utils.py     # Data processing utilities
    ├── model_utils.py    # Model helper functions
    └── post_process.py   # Prediction postprocessing
```

---

## 🚀 **PHASE 10: Final Submission Preparation**
*Priority: HIGH | Timeline: Week 6*

### Core Tasks
- [ ] **Final test on evaluation website**: confirm F1 score meets target (>40 for 3 points, >30 for 2 points, >20 for 1 point)
- [ ] **Prepare final submission files**: explanation.txt, prediction.csv, all code files (no zip compression)
- [ ] **Double-check submission format requirements**: file naming, content structure, coordinate format
- [ ] **Submit to Ilias before deadline** (July 30, 2025, 23:55) and verify successful upload

### Pre-Submission Checklist
- [ ] **Prediction.csv format validated**: Correct columns, coordinate format, class IDs 0-24
- [ ] **F1 score confirmed**: Website shows expected performance level
- [ ] **All code files included**: Training, inference, utilities, configurations
- [ ] **Explanation.txt complete**: All required sections filled out
- [ ] **No zip files**: Individual files uploaded separately to Ilias
- [ ] **Team registration**: Ensure all team members are registered in Ilias
- [ ] **Backup submissions**: Keep local copies of all submission files

### Success Probability Assessment

**For 3 Bonus Points (F1 > 40)**:
- **High chance**: YOLOv8m + proper hyperparameter tuning + systematic optimization
- **Medium chance**: YOLOv8s + excellent optimization + perfect threshold tuning
- **Low chance**: Without systematic confidence threshold tuning

**Critical Success Dependencies**:
1. **Quality dataset analysis** → Informed augmentation strategy
2. **Proper train/validation split** → Reliable F1 estimates  
3. **Assignment-specific F1 implementation** → Accurate optimization target
4. **Confidence threshold tuning** → F1 score maximization

---

## ⚠️ **High-Risk Areas Requiring Extra Attention**

### 1. Coordinate Format Issues
- **Risk**: Assignment uses `[x_center, y_center, width, height]` in **absolute pixels**
- **Mitigation**: Implement robust coordinate conversion with validation
- **Testing**: Manual verification with bounding box visualization

### 2. Class Mapping Errors  
- **Risk**: Model's internal class IDs may not match assignment 0-24 range
- **Mitigation**: Create explicit mapping dictionary and validate
- **Testing**: Check class predictions against ground truth labels

### 3. F1 Calculation Mismatch
- **Risk**: Using standard mAP instead of assignment-specific F1 with IoU > 0.5
- **Mitigation**: Implement custom F1 calculation matching exact assignment criteria
- **Testing**: Validate against known ground truth examples

### 4. Confidence Threshold Optimization
- **Risk**: Suboptimal threshold leading to poor precision/recall balance
- **Mitigation**: Systematic threshold sweep with validation on assignment metric
- **Testing**: Use evaluation website for immediate feedback

---

## 📊 **Timeline & Milestone Management**

### Week 1: Foundation
- Complete Phases 1-2 (Dataset Analysis + Model Setup)
- **Milestone**: Dataset understood, environment configured, baseline model selected

### Week 2-3: Core Development  
- Complete Phases 3-4 (Data Pipeline + Training)
- **Milestone**: First trained model with reasonable performance

### Week 4: Optimization
- Complete Phases 5-6 (Hyperparameter Tuning + Validation)
- **Milestone**: F1 score optimization completed, target performance achieved

### Week 5: Output Generation
- Complete Phases 7-8 (Inference + Testing)
- **Milestone**: Prediction.csv generated and validated on evaluation website

### Week 6: Final Preparation
- Complete Phases 9-10 (Documentation + Submission)
- **Milestone**: All submission files ready, uploaded to Ilias

### Buffer Time
- **Week 7-8**: Reserved for unexpected issues, final optimizations, or backup approaches

---

## 🎯 **Success Metrics & Targets**

### Primary Target: **F1 Score > 40** (3 Bonus Points)
- Requires systematic optimization and proper implementation
- Focus on confidence threshold tuning and NMS parameter optimization
- Expected with YOLOv8m + dedicated optimization effort

### Secondary Target: **F1 Score > 30** (2 Bonus Points)  
- Achievable with YOLOv8s + basic optimization
- Good fallback if primary target proves challenging
- Minimum acceptable performance level

### Fallback Target: **F1 Score > 20** (1 Bonus Point)
- Should be achievable with any reasonable implementation  
- Represents successful completion of basic requirements
- Last resort performance level

Following this comprehensive plan systematically, with particular attention to the **F1 optimization loop** (Phases 5-8), maximizes your chances of achieving the 3 bonus points target while providing clear fallback strategies for different performance levels.