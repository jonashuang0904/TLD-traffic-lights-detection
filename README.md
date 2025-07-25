# ML2 Assignment - Summer Semester 2025

## 📋 Project Overview

**Course**: Maschinelles Lernen II: Fortgeschrittene Verfahren  
**Institution**: KIT - Institut für Angewandte Informatik und Formale Beschreibungsverfahren  
**Timeline**: June 3, 2025 - July 30, 2025 (23:55)  
**Group Size**: 1-3 people maximum  
**Bonus Points**: Up to 3 points (0.3/0.4 grade improvement)

## 🎯 Learning Objectives

- Why is machine learning suitable for these tasks?
- Which ML methods are appropriate?
- How to apply ML methods to novel datasets?
- How to implement ML algorithms?
- Which frameworks facilitate programming?
- Which techniques facilitate and accelerate learning?

## 🔄 Project Choice

Choose **ONE** of the following two projects:

---

## 🎨 Option 1: Image Colorization

### Task Description
Develop a machine learning model that converts grayscale images to colored RGB images.

### 📊 Dataset & Input
- **Training Data**: RGB images (convert to grayscale using `skimage.color.rgb2gray`)
- **Test Data**: 50 grayscale images from `/student_dataset/test_color/images/`
- **Challenge**: The grayscale formula `grey = 0.2125r + 0.7154g + 0.0721b` is underdetermined

### 🏗️ Technical Approach

#### Recommended Architectures:
- **U-Net**: Encoder-decoder with skip connections
- **CNNs**: With upsampling layers (`nn.Upsample`, `ConvTranspose2d`)
- **Transformers**: With attention mechanisms
- **Generative Models**: GANs or Diffusion models

#### Key Recommendations:
- **Use LAB Color Space**: Only predict 2 channels (a, b) instead of 3 RGB channels
- **Self-Supervised Learning**: No explicit labels needed
- **Color Space Conversion**: Use `rgb2lab` and `lab2rgb` for processing

### 📤 Output Requirements
- **Format**: NumPy array saved as `.npy` file
- **Shape**: `[50, 224, 224, 3]`
- **Data Type**: `uint8` values (0-255)
- **Color Space**: RGB
- **File Creation**: `np.save("prediction.npy", array)`

### 🏆 Evaluation Criteria
**Metric**: Mean Squared Error (MSE) between predictions and ground truth

```python
mse = np.square(np.subtract(student_prediction, rgb_labels)).mean()
```

**Scoring**:
- MSE < 45: **3 bonus points** ⭐⭐⭐
- MSE [45-55): **2 bonus points** ⭐⭐
- MSE [55-65]: **1 bonus point** ⭐

---

## 🚦 Option 2: Traffic Light Detection

### Task Description
Create a 2D object detector that detects and classifies traffic lights in automotive images.

### 📊 Dataset & Input
- **Training Data**: ATLAS dataset (YOLO format) from CoCar NextGen autonomous vehicle
- **Test Data**: 425 images from CoCar NextGen
- **Classes**: 25 traffic light states (see class list below)

#### Traffic Light Classes (0-24):
```
0: circle_green          13: arrow_right_green
1: circle_red            14: arrow_right_yellow
2: off                   15: arrow_straight_green
3: circle_red_yellow     16: arrow_straight_left_green
4: arrow_left_green      17: arrow_straight_red_yellow
5: circle_yellow         18: arrow_straight_left_red
6: arrow_right_red       19: arrow_straight_left_yellow
7: arrow_left_red        20: arrow_straight_left_red_yellow
8: arrow_straight_red    21: arrow_straight_right_red
9: arrow_left_red_yellow 22: arrow_straight_right_red_yellow
10: arrow_left_yellow    23: arrow_straight_right_yellow
11: arrow_straight_yellow 24: arrow_straight_right_green
12: arrow_right_red_yellow
```

### 🏗️ Technical Approach

#### Recommended Models:
- **YOLO**: Ultralytics YOLO (v8/v11)
- **RT-DETR**: Real-time detection transformer
- **Generic Object Detectors**: Pre-trained on COCO/ImageNet

#### Implementation Tips:
- Dataset already in YOLO format
- Follow Ultralytics tutorial for quick setup
- Use pre-trained models and fine-tune

### 📤 Output Requirements
**Format**: CSV file with columns: `ImageName,xywh,Conf,Classification`

```csv
ImageName,xywh,Conf,Classification
1708418258713137499_front_medium.jpg,"[260.25, 265.0, 37.0, 104.0]",0.93408203125,1
1708418258713137499_front_medium.jpg,"[1313.0, 289.75, 34.0, 98.5]",0.93359375,1
```

**Coordinate Format**:
- `x, y`: Center point of bounding box (absolute pixels)
- `w, h`: Width and height of bounding box (absolute pixels)
- `Conf`: Confidence score (use 0.8 if unavailable)
- `Classification`: Integer class ID (0-24)

### 🏆 Evaluation Criteria
**Metric**: F1 Score with IoU threshold > 0.5

```
F1 = (2 × TP) / (2 × TP + FP + FN)
```

**Scoring**:
- F1 > 40: **3 bonus points** ⭐⭐⭐
- F1 [30-40): **2 bonus points** ⭐⭐
- F1 [20-30]: **1 bonus point** ⭐

---

## 📁 Submission Requirements

Submit **3 documents minimum** (NO zip files):

### 1. 📝 Explanation File (.txt)
Include:
- Which task you chose
- ML architecture used
- Training approach overview
- Brief methodology explanation

### 2. 🔮 Predictions File
- **Colorization**: `.npy` file with predictions
- **Traffic Light Detection**: `.csv` file with detections

### 3. 💻 Code Files
- All implementation files used
- Any programming language/framework allowed
- Jupyter notebooks or .py files both acceptable
- Code documentation helpful but not required

---

## 🔗 Resources & Links

### 📊 Dataset Access
- **Main Dataset**: https://bwsyncandshare.kit.edu/s/KcigioLJHmaKD8n

### 🌐 Evaluation Website
- **Live Scoring**: https://kitml2.streamlit.app/
- Upload your prediction files to see current scores

### 📚 Tutorials & Documentation

#### PyTorch Resources:
- [Beginner Tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Custom Datasets](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Optimizers & Learning Rates](https://pytorch.org/docs/stable/optim.html)
- [Model Save/Load](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

#### YOLO Resources:
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Training Tutorial](https://www.youtube.com/watch?v=r0RspiLG260)
- [Training Arguments](https://docs.ultralytics.com/usage/cfg/#train-settings)

#### Cloud Computing:
- **Google Colab**: Free GPU access for training
- File upload: `from google.colab import files; uploaded = files.upload()`
- Drive mount: `from google.colab import drive; drive.mount('/content/drive')`

---

## ⚠️ Important Notes

### 🚫 Restrictions
- **DO NOT** publish the Ilias dataset publicly
- Maximum 3 people per group
- Only one submission per group
- Cannot add team members after deadline

### ✅ Allowed Resources
- Public code repositories
- ChatGPT/LLMs for assistance
- Stack Overflow and forums
- Pre-trained models (ImageNet, etc.)

### 💡 Technical Tips

#### For Colorization:
- Consider LAB color space over RGB
- Use U-Net architectures for pixel-level tasks
- Self-supervised learning approach
- Handle channel dimensions properly (PyTorch uses channels-first)

#### For Traffic Light Detection:
- Tune confidence thresholds for optimal F1 score
- Use aggressive NMS (traffic lights rarely overlap)
- Balance precision vs recall based on F1 metric
- Consider data augmentation for better generalization

### 🔄 File Format Handling
- `.npy` files automatically converted to `.sec` in Ilias (ignore this)
- Ensure correct array shapes and data types
- Test submissions on evaluation website before final submission

---
