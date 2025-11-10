# The Krempengs' SpongeBob Scene Classification using Neural Networks

This group project was developed as the **final project for the *Deep Learning* course** at **Faculty of Computer Science, Universitas Indonesia (UI)**. 
The goal is to build a **multi-label image classification model** capable of detecting the presence of **SpongeBob, Patrick, and Squidward** in frames from the *SpongeBob SquarePants* series.

---

## Project Overview

This project explores **transfer learning**, **color-based attention**, and **data augmentation** to handle complex multi-character scenes.  
Each image may contain one, multiple, or none of the three target characters.  
Our best model, **ResNet50 with Color Attention** outperformed both **Vision Transformer (ViT)** and **VGG16** baselines used by other team members, achieving the **highest Kaggle leaderboard score** among all submissions.

---

## Dataset and Preprocessing

Custom-labeled images indicating the presence of **SpongeBob**, **Squidward**, and **Patrick**.  
Each image can feature one, two, or all three characters at once, meaning the labels are not limited to a single class.


---

### Data Analysis and Balancing
The `analyze_data()` function examines the dataset to measure class imbalance and co-occurrence.  
Results show that **SpongeBob appears more frequently**, while **Squidward and Patrick** are underrepresented.  

| Character | Positive Samples | Approx. Ratio (1:0) |
|------------|------------------|---------------------|
| SpongeBob | ~3.2k | 1 : 1.9 |
| Squidward | ~1.1k | 1 : 7.6 |
| Patrick | ~1.1k | 1 : 7.6 |

To address this, **Focal Loss** is used with class weights (α) computed from inverse class frequency,  
and **γ = 2.5** to focus more on difficult or minority examples.

---

### Image Augmentation & Attention
- **Augmentations:** Random crop, flip, color jitter, noise, and rotation (via Albumentations).  
- **Normalization:** ImageNet mean and std.  
- **Color Attention:** Each character’s dominant color (yellow, teal, pink) generates pixel-wise masks,  
  guiding the model to focus on relevant regions.

    These preprocessing steps significantly improved class balance handling and visual focus during training.
---


## Model Architecture

### Backbone: ResNet50
- Pretrained on ImageNet with 2048-dimensional feature vectors  
- Fully connected head replaced with custom layers for 3 output nodes (multi-label)

### Color Attention Module
A lightweight attention layer designed to emphasize color regions characteristic of each character:
| Character | RGB Color | Description |
|------------|------------|-------------|
| SpongeBob  | [255, 255, 0] | Yellow |
| Patrick    | [255, 182, 193] | Pink |
| Squidward  | [64, 224, 208] | Teal |

Architecture:
This module multiplies color-based attention masks with ResNet feature maps, guiding the model to focus on relevant regions.

---

## Training Configuration

| Component | Configuration |
|------------|---------------|
| **Backbone** | ResNet50 |
| **Loss Function** | Focal Loss (`γ=2.5`, α dynamic per class) |
| **Optimizer** | AdamW (`lr=4e-4`, `weight_decay=2e-5`) |
| **Scheduler** | OneCycleLR (warm-up 10%, final div factor 100) |
| **Batch Size** | 16 |
| **Epochs** | 30 |
| **Image Size** | 320×320 |
| **Dropout** | 0.4 → 0.2 |
| **Cross-Validation** | 5-Fold Stratified |
| **Precision** | Mixed precision with GradScaler |

**Augmentation:** Extensive Albumentations pipeline (crop, rotate, jitter, noise, and normalization with ImageNet mean/std).

---

## Results & Evaluation

| Fold | Validation Accuracy | Best Epoch |
|------|----------------------|-------------|
| 1 | 97.21% | 19 |
| 2 | 97.68% | 28 |
| 3 | 97.57% | 23 |
| 4 | 97.15% | 26 |
| 5 | 97.73% | 26 |
| **Mean** | **97.47%** |  |

### Project Highlights and Takeaways
- **Best Model:** ResNet50 + Color Attention  
- **Outperformed:** ViT and VGG16 baselines  
- **Kaggle Leaderboard:** 🥇 *Highest score among all team submissions*  
- Stable performance across folds (variance < 0.3%)  
- Correctly identified SpongeBob in 36.06% of scenes, no main characters in 52.83%, and multi-character scenes in 6.91%
- **Color-guided attention** effectively focuses the model on key visual cues.  
- **Focal Loss + OneCycleLR** improved stability and generalization under class imbalance.  
- **ResNet50** provided the best trade-off between depth, speed, and accuracy compared to ViT and VGG16.  