# Dual-Head Distillation with Entropy Regularization for Lightweight Visual Classification in Medical Imaging

## Overview
Skin-KD is a deep learning framework for skin lesion classification based on a Dual-Head Knowledge Distillation (KD) strategy. In skin cancer diagnosis, knowledge distillation enables efficient model deployment by compressing complex networks into lightweight architectures, thereby maintaining high diagnostic performance even on resource-constrained edge devices. However, conventional knowledge distillation methods often rely on a single teacher prediction head, where the teacher model tends to produce overly sharp probability distributions that suppress information from non-target categories and weaken inter-class relational knowledge. To address this issue, we propose a dual-head teacher framework with entropy regularization to generate more informative and semantically rich soft labels while preserving discriminative capability. The student model learns from both prediction heads simultaneously, enabling it to capture discriminative features and structured semantic relationships, thereby improving robustness and generalization. This project is designed for dermoscopic skin lesion classification using the ISIC 2019 dataset and BCN 20000 dataset.

## Key Contributions
- Dual-head knowledge distillation framework for sample relational knowledge maximization  
- Entropy-regularized auxiliary prediction head for richer and smoother soft labels  
- Preservation of discriminative capability while enhancing semantic diversity  
- Improved modeling of inter-class relational knowledge  
- Joint learning of discriminative and structured semantic representations  
- Improved performance on large-scale imbalanced skin lesion datasets  

## Dataset
This project uses two public dermoscopic datasets:

- ISIC 2019 Challenge Dataset  
  - ~25,000 dermoscopic images  
  - 8 classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC  
  - Dataset link: https://challenge.isic-archive.com/data/#2019  

- BCN 20000 Dataset  
  - Large-scale dermoscopic image dataset for skin lesion classification  
  - Used for additional evaluation and validation  

## Project Structure
SkinKD/  
│  
├── experiment/  
│   ├── isic/  
│   ├── my_models/  
│   ├── raug/  
│   └── kd_losses/  
│  
├── configs/  
├── requirements.txt  
└── README.md  

## Installation

git clone https://github.com/zzzzzhyyy/Skin-KD.git  
cd Skin-KD  
pip install -r requirements.txt  

## Training

Standard training:  
python experiment/isic/isic.py

Knowledge distillation training:  
python experiment/isic/our_MultiHeadWeightedLoss.py

## Evaluation

Metrics:
- Accuracy  
- Balanced Accuracy  
- F1-score  
- Confusion Matrix  

## Method Overview
The proposed framework adopts a dual-head teacher-student architecture. The primary head produces standard discriminative predictions, while the auxiliary head with entropy regularization generates smoother and more informative soft labels. The student model learns from both heads via knowledge distillation, enabling simultaneous learning of discriminative features and inter-class semantic relationships. The overall objective is to maximize sample relational knowledge and improve structured representation learning in skin cancer diagnosis.

## Requirements
- Python 
- PyTorch  
- NumPy  

pip install -r requirements.txt  

## Note

This repository contains the implementation of the paper submitted to *The Visual Computer*.


