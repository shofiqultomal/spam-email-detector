
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Load Dataset
df = pd.read_csv("data/enron_cleaned.csv")


# Separate Feature and Target
x = df["text"]
y = df["label"]


# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Models
results = []

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(max_iter=10000)
}


# Train and Evaluate Models
for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training Model:", name)
    print("=" * 50)

    # Pipeline
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("model", model)
    ])

    # Train Model
    pipeline.fit(x_train, y_train)

    # Prediction
    predictions = pipeline.predict(x_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam"
    )

    # Store Results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    ))


# Model Comparison (Outside Loop)
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# Save Results
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)
