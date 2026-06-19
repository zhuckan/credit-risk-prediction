# Credit Risk Prediction Model

## Project Overview
This project aims to build a robust machine learning model for credit risk management in the banking sector. The primary goal is to predict the probability of a borrower defaulting on a loan (defined as failing to make a payment for more than 90 days). 

By accurately identifying high-risk clients, financial institutions can minimize potential financial losses, streamline the loan approval process, and make more informed lending decisions. The final solution processes large-scale data iteratively, engineers relevant financial features, and implements a high-performance classification pipeline achieving a ROC-AUC of **0.75** on the test dataset.

## Business Context
Credit risk models are essential for banks to evaluate the trustworthiness of loan applicants. Automated systems evaluate applicants based on their credit history, repayment behavior, and current financial status. The implementation provided in this project allows for the automated aggregation of historical loan data to predict future defaults, reducing manual labor and accelerating the confirmation process for legitimate borrowers.

## Technical Stack
- **Programming Language**: Python 3.9+
- **Data Manipulation**: pandas, PyArrow (for Parquet files)
- **Machine Learning**: scikit-learn, LightGBM, CatBoost
- **Model Management**: sklearn.pipeline, pickle
- **Data Visualization**: Matplotlib

## Dataset Description
The dataset consists of extensive historical records of loan products, provided in Parquet format. Each row represents an individual credit product issued to a specific borrower. 

Due to the large volume of data (approximately 3 million records after aggregation), the data is processed in chunks to handle memory constraints. The primary data attributes include:

- **Loan Temporal Features**: Days since credit opened, confirmed, planned close, and actual close.
- **Financial Indicators**: Credit limits, outstanding balance, next payment amounts, overdue amounts, and credit cost rate.
- **Delinquency History**: Number of past due events grouped by duration (5, 5-30, 30-60, 60-90, and >90 days).
- **Encoded Categorical Features**: Payment statuses for the last 25 months, account holder type, credit status, and credit type.
- **Target Variable (`flag`)**: Binary indicator where `1` signifies a default event. The dataset is highly imbalanced, with positive default cases accounting for approximately **3.55%** of the data.

## Methodology

### 1. Data Preprocessing and Aggregation
Due to the size of the source data, files are processed iteratively. For each chunk:
- Data is sorted by client ID (`id`) and loan sequence (`rn`).
- Aggregation features are calculated across all loans belonging to each client, generating summary statistics (sum, mean, max, std, last).
- The final output is a dataset of 3,000,000 unique clients, merged with the target variable.

### 2. Feature Engineering
A set of new financial and behavioral features were engineered to improve the predictive power of the model:
- **Weighted Delinquency**: A score calculated by applying higher penalties to longer-term delinquencies (5 days = 1, >90 days = 5).
- **Utilization and Overdue Ratios**: Calculated as the ratio of outstanding loans and overdue amounts to the credit limit, providing a normalized view of a client's financial burden.
- **Term Differences**: The difference between the planned and actual term of the loan, indicating early or late closure.
- **Payment Trend Indicators**: 
  - Good/bad ratios for the 25-month payment history.
  - Mean status of the last 3 months versus the oldest 3 months.
  - Trend calculations to detect if a client's payment discipline is improving or worsening.
  - Standard deviation of payment statuses to assess consistency.

### 3. Model Training and Selection
- The model pipeline was designed using `sklearn.pipeline`.
- Two gradient boosting algorithms were evaluated: **LightGBM** and **CatBoost**.
- To handle the severe class imbalance, `scale_pos_weight` (LightGBM) and `auto_class_weights` (CatBoost) were utilized.
- **Hyperparameter Tuning**: `GridSearchCV` with 3-fold Stratified K-Fold cross-validation was conducted on a 50,000-observation subsample of the training data to optimize model performance.

### 4. Final Model and Results
After extensive experimentation, **CatBoost** was selected as the final model. The optimized pipeline was trained on the full training set and evaluated on an isolated test set (30% holdout).

| Metric | Value |
| :--- | :--- |
| **Cross-Validation Score** | 0.6711 |
| **Test Set ROC-AUC** | **0.75** |

This result aligns with the technical requirements of the project, meeting the minimum acceptable baseline of 0.75 ROC-AUC.

## Project Structure
```text
credit-risk-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   └── credit_risk_pipeline.pkl
├── notebooks/
│   └── good_job.ipynb
├── data/
│   ├── train_data_*.pq
│   └── train_target.csv
└── predictions.csv
```

## Installation and Usage

### Prerequisites
- Python 3.9 or higher.
- PyArrow (required for reading Parquet files).

### Installation Steps
1. Clone the repository to your local machine.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
3. Place the training data files (train_data_*.pq and train_target.csv) into the data/ directory.

## Running the Pipeline
Open and execute the good_job.ipynb notebook to perform:

- Data loading and aggregation.
- Feature engineering.
- Model training and evaluation.

The trained pipeline will be exported as credit_risk_pipeline.pkl in the models/ directory. The predictions on the test set will be saved as predictions.csv in the project root.

## Performance Requirements
This project fulfills the following criteria outlined in the technical assignment:

- Metric Quality: ROC-AUC exceeds 0.75 on the test set.
- Data Handling: Implements iterative processing to handle large Parquet datasets.
- Pipeline Implementation: Uses sklearn.pipeline for a streamlined fit and predict workflow.
- Experiments: Conducted multiple experiments for feature selection and algorithm comparison.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
