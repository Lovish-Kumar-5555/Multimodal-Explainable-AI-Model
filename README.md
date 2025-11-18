# 🫀 Multimodal Explainable AI for Cardiovascular Disease Prediction

This repository contains the implementation of a **multimodal machine learning & deep learning framework** that integrates **clinical tabular data** and **medical imaging** to predict **cardiovascular disease (CVD)**. The model incorporates **Explainable AI (XAI)** techniques such as **SHAP**, **LIME**, and **Grad-CAM** to ensure transparency and clinical interpretability.

This work is based on the research article:

> **“Explainable AI (XAI) in Healthcare Predictions using Tabular + Image Data Fusion”**
> *Lovish Kumar, Hasanpreet, Esha Bhardwaj, Aaskaran Bishnoi, Dayal Chandra Sati, Mankaranveer Singh*

---

## 📌 Overview

Early detection of cardiovascular disease requires a combination of **clinical knowledge**, **imaging insights**, and **transparent AI models**. This project:

* Builds **unimodal** tabular and image models.
* Combines them using **multimodal late fusion** to enhance accuracy.
* Integrates **XAI** to show *why* the AI made a prediction.
* Compares fusion strategies: **early fusion**, **late fusion**, and **joint fusion**.
* Evaluates fairness, interpretability, and reliability for potential clinical use.

---

## 🧠 Key Features

### 🔹 Multimodal Learning

* **Tabular data models** (XGBoost, LightGBM, Random Forest, MLP)
* **Imaging models** (ResNet, EfficientNet, VGG, Vision Transformers)
* **Fusion strategies**:

  * Early Fusion (feature concatenation)
  * Late Fusion (ensemble)
  * Joint Fusion (shared latent space)

### 🔹 Explainable AI (XAI)

* **SHAP** – global + local feature attribution for tabular data
* **LIME** – instance-level decision explanations
* **Grad-CAM** – heatmaps highlighting important regions in medical images

### 🔹 Evaluation Metrics

* Accuracy, Precision, Recall, F1-score
* ROC-AUC
* Sensitivity & Specificity
* Confusion Matrix
* MCC (Matthews Correlation Coefficient) for imbalanced datasets

---

## 📂 Project Structure

```
├── data/
│   ├── clinical.csv
│   ├── images/
│   │   ├── patient_001/
│   │   ├── patient_002/
│   └── ...
│
├── src/
│   ├── preprocess_tabular.py
│   ├── preprocess_images.py
│   ├── train_tabular.py
│   ├── train_imaging.py
│   ├── fusion_late.py
│   ├── fusion_early.py
│   ├── fusion_joint.py
│   ├── explain_shap.py
│   ├── explain_lime.py
│   ├── explain_gradcam.py
│   └── utils/
│
├── models/
│   ├── tabular/
│   ├── imaging/
│   ├── fusion/
│
├── results/
│   ├── metrics/
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── gradcam_heatmaps/
│
├── requirements.txt
├── README.md
└── config.yaml
```

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Description

### **Clinical Tabular Data**

Features include:

* Age, Sex
* Blood Pressure
* Cholesterol Levels
* Serum Creatinine
* Hemoglobin
* Diabetes, Hypertension
* Smoking, Lifestyle Factors

### **Medical Imaging Data**

* Echocardiography images
* Cardiac ultrasound / MRI slices
* Standardized to 224×224 resolution
* Augmented for robustness

---

## 🚀 Usage

### **1. Preprocess Tabular Data**

```bash
python src/preprocess_tabular.py
```

### **2. Preprocess Image Data**

```bash
python src/preprocess_images.py
```

### **3. Train Tabular Model**

```bash
python src/train_tabular.py
```

### **4. Train Imaging Model**

```bash
python src/train_imaging.py
```

### **5. Train Multimodal Fusion Model**

Late Fusion:

```bash
python src/fusion_late.py
```

Early Fusion:

```bash
python src/fusion_early.py
```

Joint Fusion:

```bash
python src/fusion_joint.py
```

### **6. Run Explainability Scripts**

```bash
python src/explain_shap.py
python src/explain_lime.py
python src/explain_gradcam.py
```

---


## 7. Train different models.

## With & Without clincal data.

From previous examples, we used clinical + CXR image to train the model. In this phase, we want to compare **CXR + clincal data** model and **CXR only** model. The training result are shown below. We find that clinical data can slightly improve the performance on training and validation dataset.

![image](https://user-images.githubusercontent.com/37566901/155869533-982ad3ae-f44a-42f8-8986-156311ca905a.png)


One of our reason to include clinical data is that we assume the clinical data can promote explainability. However, as we ploted the GradCAM++ for both models, we found the **CXR + clinical data** model has a difficulty to point out abnormalities and has strange rectangles around corners. This can be cuased by the elementwise sum operation we conduct in the fusion layer. To further investigate this problem, we train another model with alternative fusion strategy, concatenation. And, the result are shown below.

![image](https://user-images.githubusercontent.com/37566901/155870609-747c0465-9357-4e09-8e85-8d521fa0aa27.png)

![image](https://user-images.githubusercontent.com/37566901/155870613-4cdb1b9b-7170-469a-a8dd-33e3015df86f.png)

The model with concatenation operation for fusion has better performance, Also, in the GradCAM, it doesn't show strange rectangle around the coners of radiographs. Therefore, we decided to use concatenation operation for rest of the experiments to obtain better GradCAM++ images (heatmaps). 

The GradCAM++ is using the last convolutional layer to calculat the gradient to output and obtain the activation map. However, in the **CXR + clinical data** model, the CXR images is not the only contributing to the output. When we'er using the GradCAM++ to generate the sailency map, the GradCAN can't measure the effect of clinical data, which may affect the explainability.

### Add *without CXR*. 

Also, we add an experiement to know how's the perfromance when the model can only use clinical data. The result shows the model with *only clinical data* can't be trainable.

![image](https://user-images.githubusercontent.com/37566901/155871601-f9ecdefe-b24d-4b0c-b8fb-b2f5a3b1b218.png)

## 📈 Results

### **Model Performance**

| Model                        | AUC      | F1       | Sensitivity | Specificity |
| ---------------------------- | -------- | -------- | ----------- | ----------- |
| Tabular-Only                 | 0.88     | 0.74     | 0.79        | 0.72        |
| Image-Only                   | 0.86     | 0.71     | 0.74        | 0.70        |
| **Multimodal Fusion (Ours)** | **0.93** | **0.81** | **0.84**    | **0.85**    |

### **Explainability Outputs**

* **SHAP** identifies top tabular predictors:

  * Serum Creatinine
  * Diabetes Status
  * Hemoglobin Levels
  * Blood Pressure
* **Grad-CAM** identifies key cardiac image regions:

  * Ventricular wall thickness
  * Atrial enlargement
  * Texture abnormalities

---

## 🏥 Clinical Impact

This system:

* enhances clinician trust through transparent decisions
* improves prediction accuracy using multimodal evidence
* provides interpretable visualizations for diagnostic support
* follows SPIRIT-AI & CONSORT-AI standards

> ⚠️ **Disclaimer:**
> This model is NOT intended for real-time clinical decision-making.
> Research use only.

---

## 🤝 Contributors

* **Lovish Kumar**
* Hasanpreet
* Aaskaran Bishnoi
* Dayal Chandra Sati
* Mankaranveer Singh

---


## ⭐ Acknowledgements

We thank all collaborators and institutions supporting explainable AI research in healthcare.

---


