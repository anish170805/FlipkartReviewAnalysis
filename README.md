# FK Sentiment Analysis

A sentiment analysis system built on Flipkart product reviews, combining classical machine learning with experiment tracking and an interactive UI.

The project focuses on clean training pipelines, reproducibility, and practical usability rather than just model accuracy.

---

## Overview

This project analyzes customer reviews and predicts sentiment using multiple machine learning models.  
It includes a complete training workflow, experiment tracking with MLflow, and a Streamlit-based interface for running predictions.

Key goals:
- Train and compare multiple models reliably
- Track experiments and models over time
- Provide a simple UI for inference without exposing an API

---

## Features

- **Text Preprocessing**
  - Cleaning and normalization
  - TF-IDF vectorization with n-grams

- **Model Training & Selection**
  - Logistic Regression
  - Linear SVM
  - Naive Bayes
  - Cross-validated model selection

- **Experiment Tracking**
  - Parameters, metrics, and artifacts logged using MLflow
  - Local tracking server for inspection and comparison

- **Workflow Orchestration**
  - Training pipeline defined using Prefect flows

- **Interactive UI**
  - Streamlit app for live sentiment prediction
  - Loads trained model artifacts directly

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

The training pipeline is executed as a Python module to ensure imports work correctly.

From the project root:

```bash
python -m training.train_flow
```

---

## Experiment Tracking (MLflow)

```bash
mlflow ui
```

Open:
```
http://127.0.0.1:5000
```

---

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## Tech Stack

- Python
- Scikit-learn
- MLflow
- Prefect
- Streamlit
- Pandas, NumPy

---

## License

MIT License