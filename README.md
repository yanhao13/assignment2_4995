# assignment2_4995_UNI_yb2630

## 📌 Project Overview
This repository contains a comprehensive comparative analysis of two powerful machine learning paradigms: **Gradient Boosted Decision Trees (GBDT)** and **Multi-Layer Perceptrons (MLP)**. The objective of this project is to predict loan repayment behavior using the **Home Credit Default Risk dataset**, exploring how ensemble methods and neural networks improve generalization through fundamentally different learning mechanisms on tabular financial data.

A primary focus of this project is handling severe class imbalance (~8% default rate) and constructing rigorous Scikit-Learn pipelines to strictly prevent data leakage during preprocessing and feature engineering.

## 📂 Dataset
The data is sourced from the [Home Credit Default Risk Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk). It features mixed data types, non-linear relationships, and moderate dimensionality. 

To capture a holistic view of applicant behavior, historical data from secondary tables was aggregated and merged into the primary application dataset:
* `application_train.csv` (Main applicant data)
* `bureau.csv` (Historical credit data from other institutions)
* `previous_application.csv` (Previous loan applications at Home Credit)
* `installments_payments.csv` (Historical repayment data)

## 🛠️ Methodology

### 1. Data Preparation & Feature Engineering
Strict adherence to preventing data leakage was maintained by encapsulating all transformations within Scikit-Learn `ColumnTransformer` pipelines applied strictly *after* the train/validation/test split.
* **Feature Engineering:** Domain-specific ratios were created to capture financial stability, including `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, and `DAYS_EMPLOYED_PERC`.
* **Imputation & Encoding:** Missing numerical values were handled via median imputation, while categorical variables were one-hot encoded.
* **Scaling:** A `StandardScaler` was applied to all numerical features. While tree-based models are scale-invariant, standardization is mathematically required for the MLP to ensure stable gradient descent and prevent exploding/vanishing gradients.

### 2. Model Training & Tuning
* **GBDT (XGBoost):** Trained utilizing GPU acceleration (`tree_method='hist'`). Hyperparameters such as `learning_rate`, `max_depth`, and `subsample` were explored. Early stopping was implemented by monitoring `logloss` on a dedicated validation set to prevent overfitting.
* **MLP (Neural Network):** Constructed using Scikit-Learn's `MLPClassifier`. Various network topologies were evaluated (e.g., shallow vs. deep architectures). The final architecture utilized a `(64, 32)` hidden layer structure with ReLU activation and early stopping based on a validation fraction.

## 📊 Results & Comparison

Because the dataset is heavily imbalanced, overall Accuracy is a deceptive metric. Models were primarily evaluated on **Area Under the Precision-Recall Curve (AUC-PR)** and **Recall**, utilizing a customized prediction threshold of **0.10** to reflect real-world banking logic (prioritizing the identification of defaults over raw accuracy).

### Final Model Comparison on Test Set (Threshold: 0.10)

| Metric | GBDT (XGBoost) | MLP (Neural Network) |
| :--- | :--- | :--- |
| **Accuracy** | 0.7639 | 0.6759 |
| **Precision** | 0.1936 | 0.1562 |
| **Recall** | 0.6079 | 0.6847 |
| **F1-Score** | 0.2936 | 0.2544 |
| **AUC-PR** | 0.2560 | 0.2112 |
| **Training Time** | 9.47 sec | 29.25 sec |

### Key Insights
1. **Predictive Power:** The GBDT comprehensively outperformed the MLP in underlying model intelligence (AUC-PR of 0.2560 vs. 0.2112). The GBDT successfully balanced catching true defaults (~61% Recall) while maintaining a healthy overall accuracy.
2. **Computational Efficiency:** The XGBoost model trained nearly three times faster on this tabular dataset compared to the neural network. 
3. **Interpretability:** GBDTs offer high interpretability. Feature importance analysis revealed that external credit scores and the engineered `CREDIT_INCOME_RATIO` were the strongest predictors of default risk. In contrast, the MLP acts as a "black box" where specific decision weights cannot be easily translated into business insights.

## 💻 Repository Structure

```text
├── data/                   # Directory for dataset files (ignored in git)
├── notebooks/              # Jupyter notebooks containing the EDA, modeling, and evaluation
│   └── GBDT_vs_MLP.ipynb   # Main execution notebook
├── src/                    # Custom Python scripts for data processing/plotting
├── README.md               # Project documentation
└── requirements.txt        # Required Python packages
```

## 🚀 How to Run
1. Clone this repository.
2. Download the required dataset files from Kaggle and place them in the `/data` directory.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Jupyter Notebook in Google Colab (ensure GPU runtime is enabled for XGBoost acceleration).
