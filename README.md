# NeuroSight AI: Retinal Microvascular & Biomarker Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble%20Model-eb3434.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroSight AI** is a clinical-grade machine learning diagnostic platform that analyzes fundus photography to segment retinal microvasculature, extract biophysical morphological biomarkers, and predict neuro-ophthalmic and systemic risk profiles using a dual-stage deep learning and gradient boosted ensemble pipeline.

---

## Key Features

* **Universal Clinical Ingestion Engine (`custom.py`)**: Automatic aspect-ratio correction, color space stabilization, background artifact removal, and adaptive normalization across clinical fundus cameras and web sources.
* **Deep Microvascular Segmentation (`models/unet_vessel.pth`)**: Custom PyTorch U-Net architecture engineered for capillary-level vessel segmentation under low-contrast illumination.
* **High-Precision Quantitative Biomarker Extraction**:
  * Retinal Vessel Caliber & Density Index
  * Arteriolar-to-Venular Ratio (AVR) approximations
  * Branching Angle Variance & Vascular Tortuosity Index
* **Hybrid Ensemble Diagnostics (`models/xgboost_model.json`)**: Multi-modal classification engine fusing segmented topological metrics with spatial feature encodings for clinical risk tiering.
* **Interactive Medical Interface (`app.py`)**: Multi-page dashboard with real-time curing side-by-side previews, diagnostic threshold adjustments, metric visualization, and dynamic dossier export.
* **Automated Clinical Reporting (`generate_pdf_dossier.py`)**: Generates structured, tamper-evident PDF dossiers containing segmentation heatmaps, metric distributions, and risk confidence scores.

---

## Architectural Pipeline

                   [ Input Fundus Image ]
                             │
                             ▼
                 [ Universal Image Curing ]
              (custom.py: CLAHE + Normalization)
                             │
     ┌───────────────────────┴───────────────────────┐
     ▼                                               ▼
[ Deep Vessel U-Net ]                       [ Morphological Feature Engine ](unet_vessel.pth)                           - Tortuosity Calculation│                                   - Vessel Caliber Index▼                                   - Branching Geometry[ Segmentation Mask ]                                    ││                                               │└───────────────────────┬───────────────────────┘▼[ XGBoost Ensemble Model ](xgboost_model.json)│▼[ Diagnostic Stratification ](Visual Dashboards + Clinical PDF Dossier)
---

## Repository Structure

retinal-neuro-ai/├── models/│   ├── unet_vessel.pth                   # Trained deep U-Net vessel segmentation weights│   └── xgboost_model.json                # Optimized gradient boosted risk classification model├── app.py                                # Streamlit multi-page clinical diagnostic interface├── custom.py                             # Advanced image curing and preprocessing pipeline├── generate_pdf_dossier.py               # Comprehensive clinical PDF generation engine├── NeuroSight_AI_Comprehensive_Master_Dossier.pdf # Reference technical dossier├── requirements.txt                      # Project dependencies and version pins├── To_Run_CMD                            # Quick terminal execution references└── Full_Steps                            # Step-by-step reproduction documentation
---

## Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Clone the Repository
```bash
git clone [https://github.com/aaryanm1208-ship-it/NeuroSight-AI.git](https://github.com/aaryanm1208-ship-it/NeuroSight-AI.git)
cd NeuroSight-AI/retinal-neuro-ai
3. Create a Virtual EnvironmentBash# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
4. Install DependenciesBashpip install --upgrade pip
pip install -r requirements.txt
Running the ApplicationLaunch the interactive clinical dashboard:Bashstreamlit run app.py
To compile a fresh copy of the master technical dossier:Bashpython generate_pdf_dossier.py
Diagnostic MethodologyProcessing PhaseEngine / ModuleOutput Metric / RepresentationIngestioncustom.pyStabilized 512x512 CLAHE green-channel normalized tensorSegmentationunet_vessel.pthBinary vessel tree mask + continuous confidence probability mapBiomarker MathNative NumPy / OpenCVTortuosity score, vessel caliber ratio, branch fractal densityInferencexgboost_model.jsonMulti-class stratification with calibrated probabilitiesDocumentationgenerate_pdf_dossier.pyAutomated diagnostic summary PDFSecurity & Clinical Use NoticeDisclaimer: NeuroSight AI is engineered as an auxiliary screening decision-support platform for clinical research. It is intended to assist medical professionals by extracting quantitative microvascular indicators and should be evaluated alongside comprehensive ophthalmic assessments and certified medical protocols.LicenseThis project is open-source and distributed under the MIT License.
---

### **To push this README directly to your GitHub repository:**

Run these commands in your PowerShell terminal:

```powershell
git add README.md
git commit -m "docs: add comprehensive clinical and technical architecture README"
git push origin main
