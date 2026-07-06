# Job Satisfaction Prediction — Stack Overflow Developer Survey

A machine learning project that predicts developer job satisfaction using responses from the Stack Overflow Developer Survey. The project covers the full pipeline from raw survey data to a trained predictive model, including extensive data cleaning, feature engineering, and feature selection.

## Team & Role

This project was built collaboratively as a team project.
1. Tasneem Ma'mon: Primary Cleaning.
2. Gehad Ebrahim: Missing Values & Outliers.
3. Tasneem Osama: Encoding & Multi-label features.
4. Lina Mohamed: Feature selection & scaling.
5. Basmala Ahmed: Regression Models & Evaluation.

## Project Structure

```
JobSatisfaction/
├── data/                      # Raw/processed survey data (not included, see Data section)
├── notebooks/
│   └── preprocessing.ipynb    # Exploratory preprocessing notebook
├── src/
│   ├── preprocessing.py       # Core preprocessing functions (missing values, outliers, encoding, feature selection)
│   ├── preprocessing_assets.pkl  # Saved fitted encoders/scalers/imputers
│   └── training.ipynb         # Model training & evaluation
├── requirements.txt
└── README.md
```

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GehadEbrahim/Job_Satisfaction_Predection.git
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data

This project uses the [Stack Overflow Developer Survey](https://survey.stackoverflow.co/) public dataset. The raw data is **not included** in this repository (to keep it lightweight and avoid redistributing third-party data).

To reproduce the pipeline:
1. Download the survey results CSV from the link above.
2. Place it inside the `data/` folder.
3. Update the file path at the top of `notebooks/preprocessing.ipynb` if needed.

## Preprocessing Pipeline (`src/preprocessing.py`)

Key steps implemented:
- **Column triage:** columns with ≥70% missing values are dropped; columns with low missingness (≤5%) have their rows dropped; the rest are imputed.
- **Missing value imputation:**
  - Mode / grouped-mode imputation for categorical features (grouping by a related column when available)
  - `IterativeImputer` for `WorkExp` and `YearsCode`, combined with domain-logic rules (e.g., students assumed to have 0 work experience)
  - Country-based median imputation for salary
- **Outlier handling:** IQR-based clipping plus manual capping on known long-tailed features (e.g., `YearsCode`, `WorkExp`, tool counts)
- **Encoding:** ordinal encoding (Age, EdLevel), one-hot encoding (low-cardinality categoricals), multi-label binarization (multi-select survey questions), and target/Bayesian encoding (high-cardinality columns)
- **Feature engineering:** derived ratios (e.g., tools "worked with" vs. "want to work with")
- **Feature selection:** variance thresholding followed by Random Forest importance-based selection

All fitting statistics (imputation values, scalers, encoders) are computed on the training split only and then applied to validation/test to prevent data leakage.

## Tech Stack

Python, pandas, NumPy, scikit-learn, SciPy, category_encoders

## License

This project is for educational purposes.
