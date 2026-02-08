# FK Sentiment Analysis

A sentiment analysis system built on Flipkart product reviews, combining classical machine learning with experiment tracking and an interactive UI.

The project focuses on clean training pipelines, reproducibility, and practical usability rather than just model accuracy.

---

## Overview

This project analyzes customer reviews and predicts sentiment using multiple machine learning models.
It includes a complete training workflow, experiment tracking with MLflow, and a Streamlit-based interface for running predictions.

The system is designed to be modular, reproducible, and easy to extend with additional models or datasets.

---

## Features

- **Text Preprocessing**
  - Cleaning and normalization
  - Tokenization and lemmatization
  - TF-IDF vectorization with n-grams

- **Model Training & Selection**
  - Logistic Regression
  - Linear SVM
  - Naive Bayes
  - Cross-validated hyperparameter tuning
  - Automatic best-model selection

- **Experiment Tracking**
  - Parameters, metrics, and artifacts logged using MLflow
  - Confusion matrices and classification reports saved per run
  - Centralized experiment comparison via MLflow UI

- **Workflow Orchestration**
  - Training pipeline implemented using Prefect flows

- **Interactive UI**
  - Streamlit app for live sentiment prediction
  - Displays prediction confidence when available
  - Shows cleaned text used by the model

---

## Project Structure

```
FK Sentimental analysis/
│
├── app/
│   └── utils/
│       └── preprocessing.py
│
├── training/
│   ├── __init__.py
│   ├── train_flow.py
│   ├── data_loader.py
│   ├── cleaning.py
│   └── model_selection.py
│
├── streamlit_app.py
├── data/
│   └── data.csv
├── models/
│   └── sentiment_model.joblib
├── mlruns/
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements
- Python 3.11+
- Virtual environment (recommended)

### Installation

```bash
git clone https://github.com/your-username/fk-sentimental-analysis.git
cd "FK Sentimental analysis"

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

---

## Training the Model

The training pipeline must be executed as a Python module to ensure imports resolve correctly.

From the project root:

```bash
python -m training.train_flow
```

This process:
- Loads and cleans the dataset
- Trains and evaluates multiple models
- Logs metrics and artifacts to MLflow
- Saves the best-performing model locally

---

## Experiment Tracking (MLflow)

Start the MLflow tracking UI in a separate terminal:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open in your browser:

```
http://127.0.0.1:5000
```

MLflow is used to:
- Compare model performance
- Inspect hyperparameters
- View evaluation artifacts
- Manage experiment history

---

## Running the Streamlit App

After training completes and a model is saved:

```bash
streamlit run streamlit_app.py
```

The UI allows you to enter custom review text and receive real-time sentiment predictions.

---

## Tech Stack

- **Python**
- **Scikit-learn**
- **MLflow**
- **Prefect**
- **Streamlit**
- **Pandas, NumPy**

---

## License

MIT License