# Automated-Lung-Cancer-Detection-from-CT-Images

## Project Overview

This project demonstrates a deep learning workflow for automated lung cancer detection and classification using CT scan images. The goal was to build a clinically relevant image classification model capable of distinguishing between normal lung tissue and three major lung cancer subtypes: adenocarcinoma, large cell carcinoma, and squamous cell carcinoma.

The project uses an EfficientNetB0 transfer learning architecture with a two-stage training strategy and 10-fold stratified cross-validation to evaluate model generalizability. The final model achieved strong diagnostic performance, including a mean cross-validation AUC of 0.9470 and a holdout test AUC of 0.9092.

## Key Results

- Built a CNN-based medical imaging classifier using EfficientNetB0
- Applied transfer learning with ImageNet-pretrained weights
- Used 10-fold stratified cross-validation for robust evaluation
- Achieved:
  - Mean cross-validation AUC: 0.9470
  - Holdout test AUC: 0.9092
  - Holdout test accuracy: 75.93%
  - Normal class recall: 1.00
- Most importantly, the model did not classify any cancerous scan as normal, making it useful as a first-pass screening support tool.

## Dataset

The dataset contains CT scan images across four classes:

- Adenocarcinoma
- Large Cell Carcinoma
- Normal lung tissue
- Squamous Cell Carcinoma

The project uses a publicly available Hugging Face lung cancer CT image dataset and evaluates performance across training, validation, and holdout test splits.

## Methodology

The workflow includes:

1. Image preprocessing  
   - grayscale conversion  
   - resizing to 224 x 224  
   - contrast enhancement  
   - channel duplication for EfficientNet compatibility  

2. Model development  
   - EfficientNetB0 backbone  
   - custom dense classification head  
   - dropout and batch normalization  
   - class-weighted training  

3. Training strategy  
   - frozen feature extraction stage  
   - fine-tuning of upper EfficientNet layers  
   - early stopping and learning-rate reduction  

4. Evaluation  
   - 10-fold stratified cross-validation  
   - holdout test evaluation  
   - accuracy, precision, recall, F1-score, AUC, specificity, and confusion matrix analysis  

## Repository Files

```text
.
├── Lung_cancer_detection_fixed_10fold.ipynb
├── Lung_Cancer_Detection_Paper.docx
└── README.md
