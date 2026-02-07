# AeroFineFusion-Det

This repository provides the official implementation of **AeroFineFusion-Det**,  
a fine-scale bidirectional multi-scale fusion framework for UAV aerial small object detection.

> 🔗 **DOI**: To be released via Zenodo upon acceptance.

---

## 📄 Paper

**Enhancing UAV Aerial Small Object Detection through Fine-Scale Bidirectional Multi-Scale Fusion**  
XuanYu Cai, Guanxun Cui  
*The Visual Computer* 

📌 *The link to the published paper will be updated once available.*

---

## 🧠 Overview

AeroFineFusion-Det is an enhanced YOLOv8-based detector designed for UAV aerial imagery,  
addressing the challenges of small object detection under complex backgrounds, scale variation,  
and dense object distributions.

**Key features:**
- Fine-scale bidirectional multi-scale feature fusion
- Dual-attention weighted fusion mechanism
- Lightweight hybrid unit embedded in the C2f module
- Cross-layer local-aware fusion detection head

---

## 🗂️ Repository Structure

```text
AeroFineFusion-Det/
├── configs/            # Model configuration files
├── models/             # Network architecture and modules
├── datasets/           # Dataset loaders and preprocessing
├── tools/              # Training and evaluation scripts
├── weights/            # Pretrained / trained model weights
├── requirements.txt
└── README.md
##⚙️ Environment Setup
Requirements
Python >= 3.9

PyTorch >= 1.13.1

CUDA >= 11.6 (recommended)

Install dependencies:

pip install -r requirements.txt
---
##📊 Datasets
This project supports the following UAV benchmarks:

VisDrone2019

UAVDT

Please download the datasets from their official websites and organize them as follows:

datasets/
├── VisDrone2019/
│   ├── images/
│   └── annotations/
└── UAVDT/
    ├── images/
    └── annotations/
⚠️ Dataset usage follows the original licenses.
This repository does not redistribute dataset files.

---
##🚀 Training
Example training command on VisDrone2019:

python tools/train.py \
  --config configs/aerofinefusion_det.yaml \
  --dataset visdrone \
  --epochs 300 \
  --batch-size 16
---
##🧪 Evaluation
Evaluate a trained model:

python tools/test.py \
  --config configs/aerofinefusion_det.yaml \
  --weights weights/aerofinefusion_det.pth \
  --dataset visdrone
---
📈 Performance
Dataset	AP$_{50:95}$	AP$_s$	Params
VisDrone2019	+4.4% ↑	+5.3% ↑	−33.9%
Compared with the YOLOv8 baseline.
---
##🔁 Reproducibility
All experiments are conducted with fixed random seeds.

Training and evaluation scripts are fully provided.

Detailed hyperparameters can be found in the configuration files.
---
##📜 Citation
If you find this work useful, please cite:

@article{CaiAeroFineFusionDet,
  title   = {Enhancing UAV Aerial Small Object Detection through Fine-Scale Bidirectional Multi-Scale Fusion},
  author  = {Cai, XuanYu and Cui, Guanxun},
  journal = {The Visual Computer},
  note    = {under review}
}
