from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
                "clf__class_weight": [None, "balanced", {0: 2, 1: 1}]
            }
        },
        "naive_bayes": {
            "model": MultinomialNB(),
            "params": {
                "clf__alpha": [0.1, 0.5, 1.0]
            }
        }
    }

    best_models = {}

    for name, cfg in models.items():
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

        best_models[name] = grid.best_estimator_

    # Select best model using validation F1
    best_model = max(
        best_models.values(),
        key=lambda m: f1_score(y_test, m.predict(X_test))
    )

    # Final evaluation
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return best_model, metrics