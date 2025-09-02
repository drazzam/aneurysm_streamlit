---
title: CTA Aneurysm Detection - GLIA-Net
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
hardware: cpu-upgrade
python_version: 3.9
---

# 🧠 CTA Aneurysm Detection - GLIA-Net

AI-powered detection of intracranial aneurysms in CTA images using GLIA-Net.

## 🚀 Features

- **DICOM & NIfTI Support**: Upload .zip DICOM series or .nii/.nii.gz files
- **AI Detection**: GLIA-Net deep learning model for aneurysm segmentation  
- **Atlas Integration**: Arterial territory labeling (ACA/MCA/PCA/VB)
- **Interactive Viewer**: Slice-by-slice visualization with overlays
- **Export Results**: Download predictions and detailed CSV reports

## 📊 Usage

1. Upload your CTA scan (DICOM zip or NIfTI file)
2. Select processing options
3. Click "🚀 Run Analysis" 
4. Explore results with the interactive slice viewer
5. Download predictions and reports

## ⚙️ Model Details

- **Architecture**: GLIA-Net (Global Localization based Intracranial Aneurysm Network)
- **Input**: 3D CTA volumes
- **Output**: Aneurysm probability masks + territory labels
- **Performance**: Optimized for CPU inference

## 📋 Supported Formats

- **DICOM**: .zip archives containing DICOM series
- **NIfTI**: .nii or .nii.gz files
- **Max File Size**: 500MB per upload

## 🔬 Citation

```bibtex
@article{glianet2021,
  title={GLIA-Net: Global-to-local image analysis network for intracranial aneurysm detection},
  journal={Patterns},
  year={2021},
  doi={10.1016/j.patter.2020.100197}
}
