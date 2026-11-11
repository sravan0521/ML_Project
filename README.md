# Loan Approval Prediction Using Machine Learning

## Project Overview
This project uses supervised machine learning classification algorithms to predict whether a loan application will be **Approved** or **Rejected** based on applicant and financial information.

The project follows an end-to-end machine learning workflow, from data understanding and preprocessing to model evaluation, saving the trained model, and deploying an interactive Gradio application.

> **Disclaimer:** This project is an educational demonstration. Its predictions should not be used as the sole basis for real-world lending decisions.

## Problem Statement
Financial institutions receive many loan applications. Manually reviewing every application can be time-consuming. This project explores how historical loan application data can be used to build a classification model that predicts loan approval status.

## Project Objective
To train and evaluate classification models that predict the `loan_status` of a loan application.

- **Machine learning type:** Supervised Learning
- **Task:** Binary Classification
- **Target column:** `loan_status`
- **Target classes:** `Approved`, `Rejected`

## Dataset
The dataset contains **4,269 records and 13 columns**.

### Features
| Column | Description |
|---|---|
| `loan_id` | Loan application identifier |
| `no_of_dependents` | Number of dependents |
| `education` | Applicant's education level |
| `self_employed` | Whether the applicant is self-employed |
| `income_annum` | Applicant's annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term |
| `cibil_score` | Applicant's CIBIL score |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | Target: Approved or Rejected |

The model uses 11 selected input features; `loan_id` is an identifier rather than a predictive feature.

## Workflow
1. Business understanding and problem definition
2. Dataset loading and initial inspection
3. Data cleaning
4. Exploratory Data Analysis (EDA)
5. Feature and target selection
6. Train-test split (80% training and 20% testing, with stratification)
7. Numerical and categorical preprocessing
8. Feature selection
9. Training six classification algorithms
10. Comparing models using evaluation metrics
11. Selecting the best model
12. Evaluating the selected model
13. Saving the trained pipeline
14. Testing predictions on a sample input
15. Deploying an interactive Gradio interface

## Machine Learning Algorithms
The following algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting

## Evaluation Metrics
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC score

## Results
The best-performing model was selected based on its F1-score.

| Metric | Gradient Boosting |
|---|---:|
| Accuracy | 98.24% |
| Precision | 0.9831 |
| Recall | 0.9887 |
| F1-score | 0.9859 |
| ROC-AUC | 0.9976 |

These are the results recorded in the project notebook's test-set evaluation. Performance on other data may differ.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Gradio
- Joblib

## Project Files
Keep the following files in the project folder:

```text
Loan-Approval-Prediction/
├── loan_Approval.ipynb
├── loan_approval_dataset.csv
├── loan_approval_model.pkl
└── README.md
```

- `loan_Approval.ipynb`: notebook containing the project workflow, analysis, model training and evaluation.
- `loan_approval_dataset.csv`: dataset used in the project.
- `loan_approval_model.pkl`: saved trained pipeline, created when the notebook's model-saving cell is executed.
- `README.md`: project documentation.

## How to Run the Project

### 1. Clone or download the project
Download the project files or clone the repository:

```bash
git clone <your-github-repository-url>
cd Loan-Approval-Prediction
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn gradio joblib jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open `loan_Approval.ipynb` and run the cells in order. Make sure `loan_approval_dataset.csv` is in the same folder as the notebook.

### 4. Run the Gradio application
Execute the Gradio deployment cells in the notebook. The interface allows a user to enter applicant details and view the predicted loan status and estimated class probabilities.

## Key Learnings
- Understanding a real-world binary classification problem
- Inspecting and cleaning a dataset
- Performing exploratory data analysis
- Preprocessing numerical and categorical features
- Selecting relevant features
- Training and comparing multiple classification algorithms
- Evaluating models with classification metrics
- Saving a complete machine learning pipeline
- Building an interactive prediction interface using Gradio

## Future Improvements
- Test the model on additional, independent data.
- Review feature importance and model behaviour.
- Add validation and clearer input constraints to the user interface.
- Evaluate fairness and potential bias before any real-world use.

## Author
**Chintala Sravan Kumar Goud**

Machine Learning Project | Loan Approval Prediction
