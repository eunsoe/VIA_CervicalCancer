# Artificial Intelligence in Global Health - Visual Inspection with Acetic Acid (VIA) in Cervical Cancer

## Overview
Cervical cancer is a major global health burden, causing more than 340,000 deaths annually, with approximately **90% occurring in low- and middle-income countries (LMICs)** where screening resources are limited. Visual Inspection with Acetic Acid (VIA) is widely used in these settings due to its low cost, rapid results, and compatibility with single-visit screen-and-treat workflows.

Despite its operational advantages, VIA interpretation is **highly operator-dependent**. Accurate screening requires reliable identification of key anatomical structures—the **cervix**, **squamocolumnar junction (SCJ)**, and **transformation zone (TZ)**—and evaluation of **acetowhite lesions** after acetic acid application. Variability in provider training, environmental conditions, and the presence of confounding factors such as inflammation often leads to inconsistent or inaccurate VIA outcomes.

This project investigates the use of artificial intelligence to improve VIA reliability and support frontline health workers in resource-limited settings. The repository includes:

- **Image Classification Models**
  - EfficientNet-B3 (binary and three-class classification)
  - PaliGemma vision encoder + custom MLP classifier
  - MedGemma Hierarchical two-gate system for improved cancer detection

- **Segmentation Models**
  - YOLOv8-Seg
  - Standard U-Net
  - nn-UNet for cervix, SCJ, and acetowhite lesion segmentation

- **Datasets and Annotation Pipelines**
  - Jhpiego VIA flashcard set (digitized and manually re-annotated)
  - IARC Colposcopy ImageBank (rule-based VIA label mapping)
  - Clinician-reviewed segmentation masks

The goal of this work is to build a robust, scalable AI-assisted VIA interpretation system capable of enhancing diagnostic consistency and reducing disparities in cervical cancer screening across low-resource environments.

---
## Installation
### Cloning the repository:
```
git clone https://github.com/eunsoe/VIA_CervicalCancer.git
cd JHPIEGO
```

### Running the notebook:
- *Google Colab*
  - Open Colab
  - Click File > Upload Notebook, and upload the VIA_BinaryClassifier.ipynb, VIA_3Class.ipynb, VIA_medgemma.ipynb, and nnunet_new.ipynb files
  - Upload the datasets in the JHPIEGO folder
  - Mount to Colab
    - from google.colab import drive
    - drive.mount('/content/drive')
  - Click 'Run All'

---
## Project Objective
This project aims to build an **AI-assisted VIA interpretation system** capable of:

- Classifying VIA images into clinically meaningful categories  
- Segmenting the **cervix**, **SCJ/TZ**, and **acetowhite lesions**  
- Reducing variability in VIA interpretation across providers  
- Supporting **screen-and-treat workflows** in low-resource settings  

---

## Real-World Significance
VIA interpretation is limited by:

- Inconsistent visualization of the **SCJ** and **transformation zone**
- Subtle, variable **acetowhite patterns**
- **Inflammation** that mimics lesions
- Variation in **lighting, angle, and device quality**
- Differences in **provider training and experience**

An effective AI-assisted VIA tool could:

- Improve detection of **precancerous lesions**
- Support **cryotherapy eligibility decisions** vs referral
- Reduce misclassification and unnecessary follow-up
- Strengthen screening programs where specialists are scarce

This work contributes to global cervical cancer elimination efforts by making VIA interpretation more consistent and objective.

---

## Data Sources

### 1. Jhpiego VIA Flashcards
- 116 clinically labeled VIA cases  
- Digitized and cropped using a standardized pipeline  
- Labels mapped to:  
  - `VIA_negative`  
  - `VIA_positive`  
  - `Suspicious_for_cancer`  
- Anatomical outlines were manually re-traced in Roboflow to generate clean segmentation masks for:
  - Cervix  
  - SCJ  
  - Lesions  

---

### 2. IARC VIA ImageBank
- 186 cases with pre- and post–acetic acid images  
- **Only post–acetic acid images** used for training  
- Labels mapped to unified three-class VIA schema  
- SCJ and lesion masks manually annotated in Roboflow using metadata  
- All annotations reviewed by clinicians due to absence of ground-truth masks  

---

### 3. IARC Colposcopy ImageBank
- ~200 colposcopy cases [3]  
- “Provisional diagnosis” fields converted into VIA-relevant labels using rule-based mapping:
  - **Included:** carcinoma, invasive, suspicious, HSIL, CIN2/3  
  - **Excluded:** infections, polyps, HPV-only findings, atrophy, inadequate images  
- Only **“After acetic acid”** images retained  
- Final dataset includes normal, precancerous, and suspicious/cancer cases  

---

### 4. AnnoCerv Dataset
- 100 cases with color-coded masks for:
  - Cervix  
  - SCJ / Transformation Zone  
  - Lesion boundaries  
- Used exclusively as **external segmentation test data**

Additional datasets listed in Appendix II may strengthen performance but require academic partnerships for access.

---

## Methods (High-Level)

## Classification

### Core Architecture: EfficientNet-B3
Models trained:
- **Binary classifier:** normal vs abnormal  
- **Three-class classifier:** VIA-negative, VIA-positive, suspicious/cancer  

**Binary classifier training setup:**
- Input size: 300 × 300  
- Optimizer: **AdamW** (lr = 1e-4)  
- Weight decay: 1e-4  
- Label smoothing: 0.1  
- Mixup: α = 0.2  
- Loss: **Focal Loss**  
- 3-fold cross-validation  

### MedGemma / PaliGemma Model
- Model: `google/paligemma-3b-mix-224`  
- Vision encoder **frozen**  
- Custom MLP head trained for 10 epochs  
  - Batch size: 4  
  - Learning rate: 1e-3  
- **Hierarchical two-gate system:**
  - **Gate 1:** Normal vs Abnormal  
  - **Gate 2:** VIA-Positive vs Suspicious/Cancer  

---

## Segmentation

### Baseline Models
- **YOLOv8-Seg**  
- **Custom U-Net** using multiple loss terms:  
  - Tversky  
  - Dice  
  - Focal  
  - Boundary Loss  
  - Anti-collapse penalties  

### Final Model: nn-UNet  
A fully self-configuring biomedical segmentation framework that:

- Automatically adjusts preprocessing, architecture, and training schedule  
- Uses dataset **“fingerprints”** to choose hyperparameters [5]  

Three nn-UNet models trained for:

- **Cervix segmentation**  
- **SCJ/TZ segmentation**  
- **Lesion segmentation**  

**Standard preprocessing performed by nn-UNet:**
- Approximate isotropic spacing  
- Intensity normalization  
- Patch-based tiling  
- 2D configuration selected (planar VIA images)  

**Architecture:**
- Residual U-Net  
- Instance normalization  
- Deep supervision  

---

## 📊 Results Overview

---

## **Classification Performance**

### **Summary Table**

| Model                          | Accuracy | F1 Score (Macro / Class)                       | Notes                                                   |
|-------------------------------|----------|------------------------------------------------|---------------------------------------------------------|
| Three-class EfficientNet-B3   | 0.50     | 0.51 (macro)                                   | Limited by class imbalance & low cancer case count      |
| Binary EfficientNet-B3        | 0.79     | 0.80                                           | Sensitivity 81.2%, Specificity 73.3%                    |
| MedGemma Two-Gate – Gate 1    | 0.80     | Normal F1 = 0.78, Abnormal F1 = 0.83          | Strong separation of normal vs abnormal                 |
| MedGemma Two-Gate – Gate 2    | 0.88     | VIA+ F1 = 0.93, Suspicious/Cancer F1 = 0.50    | Cancer detection limited by small sample size           |

### **Key Findings**
- The **three-class classifier is not reliable** with the current dataset.
- **Binary classification improves performance**, but false negatives remain clinically concerning.
- **MedGemma hierarchical two-gate model shows the most clinically meaningful behavior**:
  - **Gate 1** reliably flags abnormal images.
  - **Gate 2** identifies VIA-positive cases well but struggles with suspicious/cancer due to few examples.

---

## **Segmentation Performance**

### **Dice Scores by Model**

| Landmark  | YOLOv8 | U-Net | nn-UNet (Best) |
|-----------|--------|-------|----------------|
| Cervix    | 0.96   | 0.90  | 0.94           |
| SCJ / TZ  | 0.73   | 0.24  | 0.78           |
| Lesions   | 0.39   | 0.25  | 0.65           |

### **Key Findings**
- **Cervix segmentation** is consistently strong due to its large, well-defined structure.
- **SCJ/TZ segmentation** is harder due to small size, irregular borders, and partial visibility.
- **Lesion segmentation** is the most challenging:
  - Small and low contrast  
  - Often confounded by inflammation  
  - nn-UNet performs best but still limited (Dice = 0.65)

Segmentation results highlight the need for **more data, higher-quality annotations, and potentially more advanced architectures** for SCJ and lesion tasks.

---

## 🚀 Future Directions

### **1. Dataset Expansion**
- Increase the number and diversity of VIA images across sites, cameras, and clinical presentations.
- Add **explicit inflammation labels** and improved SCJ/lesion annotations.

### **2. Integrated “Super-Learner” Model**
- Combine segmentation + classification + VIA decision rules into a **single end-to-end system**.
- Provide **interpretable outputs**, such as:
  - Highlighted lesions  
  - SCJ/TZ visibility warnings  
  - Suggested VIA category  

### **3. Low-Cost Deployment**
- Build a **mobile Android app** for clinics in low-resource settings.
- Support **offline inference** for areas with limited connectivity.
- Conduct **usability testing** with frontline VIA providers.

---
## Acknowledgments
This project is a collaboration with **Jhpiego** as part of the **Artificial Intelligence in Global Health** class.

### Contributors
The development of this project is a group effort by this team:
- Eun-Soe (Colette) Lee
- Nishtaa Modi
- Runyu Wan
- Kaicheng Jin
- Karthik Raj Guota

With the help of:
- Luis Soenksen, MSE, PhD
- Soumyadipta Acharya, MD, PhD
- Harshad Sanghvi, MD
- Ricky Lu, MD