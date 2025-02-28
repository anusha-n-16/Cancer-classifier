
# Breast Cancer Classification

## Overview
This project focuses on classifying breast cancer tumors as **Benign (B)** or **Malignant (M)** using machine learning models. The dataset used is the **Wisconsin Breast Cancer Dataset (WBCD)**, which contains 569 instances and 32 features describing various characteristics of cell nuclei present in breast cancer biopsies.

## Dataset Description
The dataset consists of **30 numerical features**, an **ID column (removed during preprocessing)**, and a **diagnosis label (B/M)**.

### Features:
- **Mean Features**: Mean values of tumor characteristics.
- **Standard Error Features**: Measures variability in tumor characteristics.
- **Worst Features**: Maximum values observed for each characteristic.

### Target Variable:
- **Diagnosis**: The outcome variable, where:
  - `B` (Benign) = **0**
  - `M` (Malignant) = **1**

### Class Distribution:
- **Benign (B)**: 357 cases (~62.7%)
- **Malignant (M)**: 212 cases (~37.3%)
- The dataset is slightly imbalanced, which is addressed using **SMOTE (Synthetic Minority Over-sampling Technique)**.

## Data Preprocessing
- **Dropped 'ID' column** (not useful for classification).
- **Checked for missing values** (none found).
- **Encoded the target variable** (`B` → 0, `M` → 1).
- **Standardized features** using **StandardScaler**.
- **Handled class imbalance** using **SMOTE**.

## Models Used
I implemented and evaluated four machine learning models:
1. **Random Forest**
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**
4. **XGBoost**

### Model Evaluation Metrics:
- **Accuracy**
- **Precision & Recall**
- **F1-score**
- **ROC-AUC Score**

## Results
| Model                 | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-----------------------|----------|------------|--------|----------|----------|
| Random Forest        | xx.xx% | xx.xx% | xx.xx% | xx.xx% | xx.xx% |
| Logistic Regression  | xx.xx% | xx.xx% | xx.xx% | xx.xx% | xx.xx% |
| SVM                  | xx.xx% | xx.xx% | xx.xx% | xx.xx% | xx.xx% |
| XGBoost              | xx.xx% | xx.xx% | xx.xx% | xx.xx% | xx.xx% |

(*Replace xx.xx% with actual results after model evaluation*)

## How to Run the Project

### Prerequisites:
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### Steps to Run:
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
   ```
2. Run the Python script:
   ```bash
   python breast_cancer_classification.py
   ```
3. View results and model performance metrics.

## Conclusion
This project demonstrates the effectiveness of different machine learning models in diagnosing breast cancer. **XGBoost and Random Forest** provided the best performance in terms of accuracy and recall, making them ideal choices for this task. Further improvements can be made by hyperparameter tuning and incorporating deep learning models.

## Acknowledgments
- **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Libraries Used**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Imbalanced-learn, XGBoost

---
### 🚀 Future Enhancements
- **Hyperparameter tuning** to improve model performance.
- **Deep Learning approach** using Neural Networks.
- **Feature selection techniques** to reduce dimensionality.

Feel free to contribute or suggest improvements! 😊

## Author
Anusha N - Data Analyst Enthusiast

📌 **GitHub Repository**: [Your GitHub Link]

