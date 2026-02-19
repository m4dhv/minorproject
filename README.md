# Credit Risk Analysis using Machine Learning

## Overview
Built an interpretable Probability of Default (PD) model using feature engineering and multiple machine learning models, including XGBoost and Random Forest. This project focuses on predicting the likelihood of loan default and assess financial risk. The goal is to support better decision‑making in credit approval using data‑driven models.

Created as an academic project part of my **7th Semester Minor Project** in November of 2025.

------------------------------------------------------------------------

## Objectives

-   Predict credit default risk using machine learning models
-   Improve financial risk assessment accuracy
-   Apply advanced ML techniques including:
    -   Reinforcement Learning 
    -   XGBoost
    -   Random Forest Algorithm
-   Build preprocessing pipelines and visualization dashboards

------------------------------------------------------------------------

## Technologies Used

-   Python
-   XGBoost
-   Reinforcement Learning approaches
-   Random Forest
-   Pandas / NumPy --- Data preprocessing
-   Jupyter Notebook --- Model development
-   Matplotlib / Visualization tools
-   Streamlit dashboard (`dashboard.py`) for insights

------------------------------------------------------------------------

## Repository Structure

    minorproject/
    │
    ├── archive/              # Raw dataset
        └─ credit_risk_dataset.csv
    ├── tables/               # Processed data table
        └─ pd_prediction.xlsx
    ├── data preprocess.ipynb # Data cleaning & feature engineering
    ├── dashboard.py          # Streamlit visualization dashboard
    ├── archive.zip           # Dataset backup
    └── README.md

------------------------------------------------------------------------

## How to Run

### 1. Clone the repo

``` bash
git clone https://github.com/m4dhv/minorproject.git
cd minorproject
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run dashboard

``` bash
streamlit run dashboard.py
http://localhost:8501/
```
------------------------------------------------------------------------


