
# Breast Cancer Classification

📌 Overview

This project focuses on classifying breast cancer tumors as Benign (B) or Malignant (M) using machine learning models. The dataset used is the Wisconsin Breast Cancer Dataset (WBCD), which contains 569 instances and 32 features describing various characteristics of cell nuclei present in breast cancer biopsies.

📊 Dataset Description

The dataset consists of 30 numerical features, an ID column (removed during preprocessing), and a diagnosis label (B/M).

🔹 Features:

Mean Features: Mean values of tumor characteristics.

Standard Error Features: Measures variability in tumor characteristics.

Worst Features: Maximum values observed for each characteristic.

🔹 Target Variable:

Diagnosis: The outcome variable, where:

B (Benign) = 0

M (Malignant) = 1

🔹 Class Distribution:

Benign (B): 357 cases (~62.7%)

Malignant (M): 212 cases (~37.3%)

The dataset is slightly imbalanced, which is addressed using SMOTE (Synthetic Minority Over-sampling Technique).

🛠 Data Preprocessing

Dropped 'ID' column (not useful for classification).

Checked for missing values (none found).

Encoded the target variable (B → 0, M → 1).

Standardized features using StandardScaler.

Handled class imbalance using SMOTE.

🤖 Models Used

The following machine learning models were implemented and evaluated:

Random Forest

Logistic Regression

Support Vector Machine (SVM)

XGBoost

📈 Model Evaluation Metrics:

Accuracy

Precision & Recall

F1-score

ROC-AUC Score

📊 Results

Here’s the model performance in a clean table format:  

| Model               | Accuracy | Precision | Recall  | F1-Score | AUC-ROC |
|---------------------|----------|-----------|-------- |----------|---------|
| Random Forest       | 97.37%   | 96.00%    | 100.00% | 98.00%   | 99.75%  |
| Logistic Regression | 97.37%   | 97.56%    | 95.24%  | 96.39%   | 99.31%  |
| SVM                 | 97.37%   | 97.56%    | 95.24%  | 96.39%   | 99.54%  |
| XGBoost             | 97.37%   | 100.00%   | 92.86%  | 96.30%   | 99.44%  |

🚀 How to Run the Project

🔧 Prerequisites:

Ensure you have Python installed along with the required libraries:

pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn

🔹 Steps to Run:

1. Clone the Repository

git clone https://github.com/anusha-n-16/Cancer-classifier.git
cd cancer-classifier

2. Open the Jupyter Notebook

jupyter notebook

Then, navigate to breast_cancer_classification.ipynb and run the cells sequentially.

3. View Results

Model performance metrics such as accuracy, precision, recall, F1-score, and AUC-ROC will be displayed.

Confusion matrices and ROC curves will be generated for better visualization of model effectiveness.

🏆 Conclusion

This project demonstrates the effectiveness of different machine learning models in diagnosing breast cancer. XGBoost and Random Forest provided the best performance in terms of accuracy and recall, making them ideal choices for this task.

🔹 Key Takeaways:

Random Forest and XGBoost emerged as the best-performing models.

High recall models are preferable in medical diagnosis to minimize false negatives.

Hyperparameter tuning and feature selection could further enhance model performance.

🔍 Future Work:

Optimization through hyperparameter tuning to improve accuracy and generalization.

Explainability techniques such as SHAP or LIME to understand feature importance.

Integration with a web-based or mobile application to make predictions accessible to users.

This project highlights the potential of machine learning in healthcare and how predictive models can support early and accurate breast cancer detection, ultimately aiding in better patient outcomes.

📢 Acknowledgments

**Dataset Source**:

UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set

Scikit-Learn Built-in Dataset: Breast Cancer Dataset

Libraries Used: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Imbalanced-learn, XGBoost 

## Author
Anusha N - Data Analyst Enthusiast

📌 **GitHub Repository**: [Breast Cancer Classification](https://github.com/anusha-n-16/Cancer-classifier.git)


