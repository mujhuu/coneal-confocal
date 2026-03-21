# **A multi-disease** *in vivo* confocal microscopy dataset with pixel-level expert annotations for corneal disease analysis

This repository contains the official implementation code for the paper **"**A multi-disease in vivo confocal microscopy dataset with pixel-level expert annotations for corneal disease analysis". This repository provides code for model training and evaluation.

## Introduction

This project provides an implementation of lesion classification and lesion region segmentation based on corneal confocal microscopy images. To validate the effectiveness and usability of the constructed dataset, we conduct systematic evaluations using four representative classification models and three widely adopted segmentation models.

## Environment Setup

First, create and activate a Python virtual environment named `cornealconfocal`, and then install the required dependencies according to the `requirements.txt` .

```
conda create -n cornealconfocal python=3.11
conda activate cornealconfocal
pip install -r requirements.txt
```

## Model training and evaluation

### **Running the Classification Models:**

```
python jct_classification.py \
--data_root /home/u22515072/a_eye_confocal/data/cornealconfocal \
--model_type resnet18 
```

**Parameters:**

- `--data_root`: Dataset storage path
- `--model_type `: Options: “resnet18”, “resnet50”, “efficientnetb0”, and “vit”.

### **Running the Segmentation Models:**

```
python jct_segmentation.py \
--data_root /home/u22515072/a_eye_confocal/data/cornealconfocal/Lesion \
--model_type unet
```

**Parameters:**

- `--data_root`: Path to the "Lesion" folder in the dataset
- `--model_type `: Options: “unet”, “resunet”, “nestedunet”.

## Citation

If you find this work useful for your research, please consider citing:

```
@article{,
  title={A multi-disease in vivo confocal microscopy dataset with pixel-level expert annotations for corneal disease analysis},
  author={},
  journal={},
  year={}
}
```