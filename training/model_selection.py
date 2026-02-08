from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os


def train_and_select_model(df, text_col="Review text", label_col="label"):
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=5,
        max_features=30000
    )

    models = {
        "logreg": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__class_weight": [None, "balanced"]
            }
        },
        "svm": {
            "model": LinearSVC(),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__class_weight": [None, "balanced"]
            }
        },
        "naive_bayes": {
            "model": MultinomialNB(),
            "params": {
                "clf__alpha": [0.1, 0.5, 1.0]
            }
        }
    }

    best_overall_model = None
    best_overall_score = -1

    for name, cfg in models.items():
        with mlflow.start_run(run_name=f"{name}_model"):

            pipeline = Pipeline([
                ("tfidf", tfidf),
                ("clf", cfg["model"])
            ])

            grid = GridSearchCV(
                pipeline,
                cfg["params"],
                cv=5,
                scoring="f1",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)

            # ---- Log metrics ----
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", acc)

            # ---- Log params ----
            mlflow.log_params(grid.best_params_)

            # ---- Confusion Matrix ----
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            os.makedirs("artifacts", exist_ok=True)
            cm_path = f"artifacts/{name}_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(cm_path)

            # ---- Classification Report ----
            report = classification_report(y_test, y_pred)
            report_path = f"artifacts/{name}_classification_report.txt"

            with open(report_path, "w") as f:
                f.write(report)

            mlflow.log_artifact(report_path)

            # ---- Log model ----
            mlflow.sklearn.log_model(best_model, "model")

            if f1 > best_overall_score:
                best_overall_score = f1
                best_overall_model = best_model

    return best_overall_model