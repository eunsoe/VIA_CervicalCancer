# Artificial Intelligence in Global Health - Visual Inspection with Acetic Acid (VIA) in Cervical Cancer

## Overview
Cervical cancer is a major global health burden, causing more than 340,000 deaths annually, with approximately **90% occurring in low- and middle-income countries (LMICs)** where screening resources are limited. Visual Inspection with Acetic Acid (VIA) is widely used in these settings due to its low cost, rapid results, and compatibility with single-visit screen-and-treat workflows.

Despite its operational advantages, VIA interpretation is **highly operator-dependent**. Accurate screening requires reliable identification of key anatomical structures—the **cervix**, **squamocolumnar junction (SCJ)**, and **transformation zone (TZ)**—and evaluation of **acetowhite lesions** after acetic acid application. Variability in provider training, environmental conditions, and the presence of confounding factors such as inflammation often leads to inconsistent or inaccurate VIA outcomes.

This project investigates the use of artificial intelligence to improve VIA reliability and support frontline health workers in resource-limited settings. The repository includes:

- **Image Classification Models**
  - EfficientNet-B3 (binary and three-class classification)
  - PaliGemma vision encoder + custom MLP classifier
  - Hierarchical two-gate system for improved cancer detection

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
  - Click File > Upload Notebook, and upload the {ADD} VIA_nnunet_segmentation.ipynb files
  - Upload the dataset from AJL
  - Mount to Colab
    - from google.colab import drive
    - drive.mount('/content/drive')
  - Click 'Run All'

---
## Project Objective
Our team’s objective is to improve cervical cancer screening in low-resource settings by developing machine learning models that can support and enhance **Visual Inspection with Acetic Acid (VIA)**. VIA is widely used in low- and middle-income countries (LMICs) due to its low cost and ability to support same-day screen-and-treat workflows, but it is highly operator-dependent and prone to inconsistent interpretation.

This project aims to build an AI-assisted VIA decision-support system capable of:
- Classifying VIA images into clinically meaningful categories (VIA-negative, VIA-positive, suspicious for cancer)
- Accurately segmenting key anatomical structures (cervix, SCJ, acetowhite lesions)
- Ultimately improving screening quality and reducing global disparities in cervical cancer outcomes

## Real-World Significance and Impact 🌍

Cervical cancer is preventable, yet it remains a major cause of death for women in LMICs. Over **90% of global cervical cancer deaths** occur in these settings, largely due to limited access to screening and early detection. VIA is currently one of the only scalable screening tools available, but its effectiveness depends heavily on provider experience and the ability to correctly identify:

- The **cervix** and its orientation  
- The **squamocolumnar junction (SCJ)** and transformation zone  
- **Acetowhite lesions**, which indicate dysplasia after acetic acid application  

Incorrect interpretation can result in missed precancerous lesions, unnecessary referrals, delayed treatment, or misclassification of cancer risk.

AI has the potential to support frontline clinicians by providing more consistent, objective interpretations of VIA images. An effective AI-supported VIA tool could:

- Reduce diagnostic variability across providers  
- Enable earlier detection of precancerous lesions  
- Improve treatment eligibility decisions (e.g., cryotherapy vs. referral)  
- Strengthen screening programs in resource-constrained environments  

By focusing on clinically relevant anatomical segmentation and classification, this project contributes to global cervical cancer elimination goals and supports more equitable access to high-quality screening.

## Project Goals

- **Develop classification models** capable of predicting VIA-negative, VIA-positive, and suspicious-for-cancer categories.  
- **Implement segmentation models** (YOLOv8-Seg, U-Net, nn-UNet) to identify the cervix, SCJ, and acetowhite lesions.  
- **Standardize VIA image preprocessing** through cropping, normalization, and augmentation pipelines.  
- **Address dataset limitations** through manual annotation, expert-reviewed segmentation, and rule-based label mapping.  
- **Evaluate model performance** using accuracy, F1-score, Dice coefficient, and qualitative visual inspection.  
- **Support equitable screening** by improving reliability in settings where variability in clinical expertise affects outcomes.  

--- 
## Data Exploration
<ins> **Dataset Used** </ins> <br />
The dataset is a subset of the **Fitzpatrick17k** dataset, a labeled collection of about **17,000** images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the Fitzpatrick skin tone scale (FST). About **4500** images are in the subset we used, representing **21 skin conditions** out of the 100+ in the complete Fitzpatrick set.


<ins> **Data Dictionary** </ins> <br />
| Column | Data Type | Description
| --- | --- | --- |
| `md5hash` | Object | An alphanumeric hash serving as a unique identifier; file name of an image without .jpg |
| `fitzpatrick_scale` | int64 | Integer in the range [-1, 0) and [1, 6] indicating self-described FST |
| `fitzpatrick_centaur` | int64 | Integer in the range [-1, 0) and [1, 6] indicating FST assigned by Centaur Labs, a medical data annotation firm |
| `label` | Object | String indicating medical diagnosis; the target for this competition |
| `nine_partition_label` | Object | String indicating one of nine diagnostic categories |
| `three_partition_label` | Object | String indicating one of three diagnostic categories |
| `qc` | Object | Quality control check by a Board-certified dermatologist. See note. |
| `ddi_scale` | int64 | A column used to reconcile this dataset with another dataset (may not be relevant) |


<ins> **Original Dataset Files** </ins> <br />
- `images.zip` - An archive file containing the images
  - The directory is further divided into a `train` and a `test` directory.
  - `Train` is further divided into directories according to the image's label.
  - `Test` is unlabeled and the source of images for making the submission.
- `train.csv` - Full metadata about the images
- `test.csv` - The images against which you will make predictions; contains metadata but no 'label' column
- `sample_submission.csv` - A sample submission file in the correct format.
  - Note that a correct submission only has two named columns: `md5hash` and `label`

<ins> **Other Sources** </ins> <br />
"Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset", *Matthew Groh and Caleb Harris and Luis Soenksen and Felix Lau and Rachel Han and Aerin Kim and Arash Koochek and Omar Badri*; https://arxiv.org/abs/2104.09957

---
## Methodology
### 1. Dataset & Sources
- We sourced dermatology datasets from Kaggle, containing images of 21 different skin conditions across various skin tones. The dataset includes:
  - Training Set (Labeled images with conditions)
  - Test Set (Unlabeled images for evaluation)
  - Metadata (CSV files containing image IDs and condition labels)

### 2. Data Augmentation for Better Generalization
- Data augmentation increases dataset diversity and prevents overfitting:
  - Rotation (+/- 40°)
  - Width & Height Shifts (20%)
  - Shearing & Zooming (20%)
  - Horizontal Flipping

### 3. Model Architecture & Training
- Choosing the model:
  - We used **Transfer Learning** with **EfficientNetB3**, a pre-trained deep learning model that extracts image features while reducing computational costs.
- Optimizing the Model:
  - Adam Optimizer
  - Categorical Crossentropy Loss
  - Accuracy as the main performance metric
### 4. Training & Evaluation
- Model Training:
  - Trained for **7 epochs**, with **160 steps** per epoch
  - Used **early stopping** to prevent overfitting
  - Tracked **validation accuracy**
- Model Evaluation:
  - Accuracy: Measures the overall correctness of the model
  - F1 Score: Combines precision and recall into a single metric, especially useful when the classes are imbalanced
  - Prediction Confidence Histogram: Visualizes how confident the model is about its predictions.

---
## Data Exploration and Preprocessing
### 1️⃣ Data Exploration
Before training the model, we conducted a thorough exploratory data analysis (EDA) to better understand the dataset's structure, identify potential issues, and inform our preprocessing steps. Key aspects of our data exploration included:

- **Class Distribution**: We examined the distribution of images across the 21 skin conditions in the dataset to identify any class imbalances. We found significant underrepresentation of certain skin conditions, which could introduce bias into the model. This informed our decision to apply data balancing techniques to ensure fair model performance.
- **Skin Tone Representation**: We reviewed the range of skin tones represented in the dataset using the Fitzpatrick skin tone scale (FST). We noticed that certain skin tones, particularly darker tones, were underrepresented, which could lead to the model performing poorly for those individuals.
- **Data Quality**: We also performed a quality control check for missing or corrupted data. This was done by reviewing the metadata and verifying image quality to ensure reliable inputs for training.

 ### *Class Distributions*
> Displays the distribution of images across the different classes in the dataset
<img width="745" alt="Class Distribution" src="https://github.com/user-attachments/assets/d42b87b2-d3c9-4443-a300-e4c48f8678c8" />

 ### *Image Size Distributions*
> Visualizes the distribution of image sizes in the dataset, indicating how the images are sized before resizing or processing
<img width="700" alt="Image Size Distribution" src="https://github.com/user-attachments/assets/c6b9008a-dd24-4912-8f58-046d502525dd" />

 ### *Encoded Label Distributions*
> Shows the distribution of encoded labels for each class, visualizing the number of samples per class after label encoding
<img width="739" alt="Encoded Label Distribution" src="https://github.com/user-attachments/assets/220eaf68-28b6-4060-a2fb-18eae2352da2" />

 ### *Training vs. Validation Set Distributions*
> Compares the distribution of samples between the training and validation datasets, indicating how the dataset is split for training and validation purposes
<img width="739" alt="Training vs  Validation Set Distribution" src="https://github.com/user-attachments/assets/4fdc4b75-25df-4bbe-859e-de152fd1b19d" />

### 2️⃣ Preprocessing Approaches
Data preprocessing is a crucial step in preparing the dataset for model training. Our preprocessing pipeline included:

- **Data Augmentation**: Given the imbalance in the dataset and to improve model generalization, we applied several data augmentation techniques to artificially expand the training set and introduce more variation. This included:
  - Rotation (45°)
  - Flipping horizontally
  - Increasing brightness 50%
  - Adding Gaussian Blur
  - Adding extreme color
  - Adding noise
 
 ### *Image Augmentation Example #1*
> Demonstrates an example of image augmentation, highlighting a transformed image generated from an original image through random modifications
<img width="739" alt="Image Augmentation Example 1" src="https://github.com/user-attachments/assets/6cc2a78e-9413-4018-9a38-a49024c80be0" />

 ### *Image Augmentation Example #2*
> Another example showing the effect of a different augmentation transformation, showcasing the variability introduced to images for training
<img width="737" alt="Image Augmentation Example 2" src="https://github.com/user-attachments/assets/5ca2b6d9-90d4-4796-9096-8b1c9322277e" />

 ### *Image Distribution BEFORE Augmentation*
> Shows the initial distribution of images before any augmentation techniques were applied
<img width="740" alt="Image Distribution BEFORE Augmentation" src="https://github.com/user-attachments/assets/643c64d8-e362-4d86-9626-3fb547aa7b05" />

 ### *Image Distribution AFTER Augmentation*
> Displays the updated distribution of images after applying data augmentation, highlighting the increase in image diversity
<img width="738" alt="Image Distribution AFTER Augmentation" src="https://github.com/user-attachments/assets/81cb652b-d18c-49d0-bd6e-0abdd0e57d84" />

 ### *Fitzpatrick Scale Distribution*
> Illustrates the distribution of Fitzpatrick skin type ratings across the dataset, showing how skin types are distributed
<img width="741" alt="Fitzpatrick Scale Distribution" src="https://github.com/user-attachments/assets/f3e006d2-52bb-46f4-89ae-0942344f2f46" />
    
- **Class Balancing**: To address the class imbalance, we used oversampling and undersampling techniques to ensure a more even distribution of images across the classes. This was critical to mitigate the risk of the model being biased toward more frequent classes.
- **Normalization**: We normalized the pixel values of the images to a range of [0, 1] to ensure consistent input for the model and speed up convergence during training.
- **Splitting the Data**: We split the dataset into training and validation sets, ensuring that images from each skin condition were properly represented in both sets. Additionally, we used a test set with unlabeled images to evaluate model performance after training.

These preprocessing and exploration steps were vital in ensuring that the data fed into the model was clean, balanced, and diverse enough to promote fairness and avoid overfitting.

### *Learning Rate Reduction Over Time*
> Tracks how the learning rate changes over time during training, based on the learning rate reduction callback
<img width="724" alt="Learning Rate Reducation" src="https://github.com/user-attachments/assets/889b19a2-1d1a-4778-ab24-28df54bf0cc1" />

### *Training and Validation Loss & Accuracy*
> Plots the training and validation loss and accuracy over epochs, providing a comparison of model performance during training
<img width="741" alt="Training   Validation Loss:Accuracy" src="https://github.com/user-attachments/assets/6b234be1-1a51-473e-a545-4274760ca412" />

### *Prediction Confidence Histogram*
> Shows the confidence levels of the model's predictions, illustrating how certain or uncertain the model is about its predictions
<img width="690" alt="Prediction Confidence Histogram" src="https://github.com/user-attachments/assets/4ce1f01e-162d-4055-a427-a34d06f2ad78" />

---
## Results and Key Findings
📊 **Accuracy:** 0.86 <br />
📊 **F1-Score:** 0.63 <br />

## Future Improvements & Next Steps  
- **Improve Dataset Diversity**: Source additional dermatology images with diverse skin tones to reduce bias.  
- **Enhance Fairness Strategies**: Refine class-balancing techniques and explore advanced data augmentation methods.  
- **Experiment with Alternative Architectures**: Fine-Tuning: Additional fine-tuning of the EfficientNetB3 model may improve performance.
- **Deploy Model & Conduct Bias Audits**: Evaluate real-world performance across different demographics and ensure fairness in predictions.  

---
## Acknowledgments
This project is a collaboration between **Algorithmic Justice League** and **Cornell Tech** as part of the **Spring 2025 Kaggle Competition** program.

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

## Acknowledgments
- The project utilizes a dataset from Kaggle's Dermatology Dataset and the EfficientNetB3 model pre-trained on ImageNet.
- Special thanks to the *Break Through Tech AI* and *Algorithmic Justice League* for providing the challenge resources.
