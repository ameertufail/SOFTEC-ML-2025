# SOFTEC'25 – Machine Learning Competition: Hospital Readmission Prediction Challenge

## Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Objective](#objective)
4. [Evaluation Criteria](#evaluation-criteria)
5. [Submission Format](#submission-format)
6. [Project Structure](#project-structure)
7. [Installation and Setup](#installation-and-setup)
8. [Usage](#usage)
9. [EDA and Insights](#exploratory-data-analysis-and-insights)
10. [Model Development](#model-development)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Contributing](#contributing)
13. [Acknowledgments](#acknowledgments)

---

## Overview
The **SOFTEC'25 Machine Learning Competition** challenges participants to build a predictive model for identifying patients at risk of hospital readmission within 30 days of discharge. Leveraging electronic health records (EHR) and clinical data, the goal is to develop a robust binary classification pipeline that can improve patient outcomes and reduce unnecessary hospitalizations.

This project aims to provide a complete end-to-end solution, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, optimization, and evaluation.

---

## Dataset Description
The dataset provided consists of anonymized clinical records for hospital stays. It includes features such as diagnosis codes, procedure codes, admission details, and discharge statuses. The target variable `Readmitted_30` indicates whether a patient was readmitted within 30 days (`1`) or not (`0`).

### Files Provided:
- **train.csv**: Training dataset with features and the target column `Readmitted_30`.
- **test.csv**: Test dataset (without the target column) for predictions.
- **sample_submission.csv**: Template for submission format.
- **metaData.csv**: Supplementary information about the dataset's features.

### Key Features:
- **ID**: Unique identifier for each hospital stay.
- **STAY_DRG_CD**: Diagnosis-Related Group code.
- **STAY_FROM_DT** and **STAY_THRU_DT**: Admission and discharge dates.
- **STUS_CD**: Patient discharge status code.
- **TYPE_ADM** and **SRC_ADMS**: Admission type and source.
- **AD_DGNS**: Admitting diagnosis code.
- **DGNSCD01 to DGNSCD25**: Primary and secondary diagnosis codes.
- **PRCDRCD01 to PRCDRCD25**: Procedure codes.

For more details, refer to the `metaData.csv` file.

---

## Objective
The objective is to predict the likelihood of a patient being readmitted within 30 days of discharge (`Readmitted_30`). This is a binary classification problem where:
- `1`: Patient readmitted within 30 days.
- `0`: Patient not readmitted within 30 days.

Participants are expected to preprocess the data, perform exploratory data analysis, engineer relevant features, train robust models, and evaluate their performance using specified metrics.

---

## Evaluation Criteria
Submissions will be evaluated based on the following criteria:
1. **Data Preparation**:
   - Handling missing values.
   - Feature engineering and transformations.
   - Encoding categorical variables and scaling numerical features.
2. **Exploratory Data Analysis (EDA)**:
   - Quality of EDA.
   - Identification of patterns, trends, and anomalies.
   - Clear visualizations and insights.
3. **Model Selection**:
   - Justification for model choices (e.g., logistic regression, XGBoost, etc.).
   - Use of baseline vs. advanced models.
4. **Model Optimization**:
   - Hyperparameter tuning strategies.
   - Cross-validation and training pipelines.
5. **Evaluation Metrics**:
   - F1 Score.
   - Area Under the Curve (AUC-ROC).
   - Precision.
   - Recall.

---

## Submission Format
Your final submission should include:
1. A **notebook or script** containing your full pipeline:
   - Data preprocessing.
   - EDA and feature engineering.
   - Model training and evaluation.
2. **Clear visualizations** and explanations of results.
3. A **separate report** or markdown section summarizing:
   - Key findings from EDA.
   - Model performance metrics.
   - Insights and recommendations.

---

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ameertufail/SOFTEC-ML-2025.git
   cd SOFTEC-ML-2025
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset files (`train.csv`, `test.csv`, `sample_submission.csv`, `metaData.csv`) and place them in the `data/` directory.

---

## Model Development
### Baseline Models
- Logistic Regression.
- Random Forest Classifier.

### Advanced Models
- Gradient Boosting (XGBoost, LightGBM).
- Neural Networks (optional).

### Hyperparameter Tuning
- Grid Search or Randomized Search for hyperparameter optimization.
- Cross-validation to ensure model generalization.

---

## Evaluation Metrics
The following metrics will be used to evaluate model performance:
- **F1 Score**: Balances precision and recall.
- **AUC-ROC**: Measures the ability to distinguish between classes.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

---

## Acknowledgments
- Special thanks to the organizers of the SOFTEC'25 Machine Learning Competition for providing this dataset and challenge.
- Inspiration and guidance from Kaggle competitions and open-source projects.
