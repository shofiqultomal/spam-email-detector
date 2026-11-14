import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



df = pd.read_csv("data/enron_cleaned.csv")
print("\nDataset loaded:")
print(df.head())

# feature and label preparing
x = df["text"]
y = df["label"]

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print("\nTrain Data:",len(x_train))
print("\nTest Data:",len(x_test))

# convert text into numbers
vectorizer = CountVectorizer(
    stop_words="english",
)
x_train_vector =vectorizer.fit_transform(x_train)
x_test_vector =vectorizer.transform(x_test)

#Training model > Naive_bayes
model = MultinomialNB()
model.fit(x_train_vector,y_train)
print("\n Model Train is Complete")

# predriction
y_pred = model.predict((x_test_vector))

# 7. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save Model and Vectorizer
joblib.dump(
    model,
    "models/enron_spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/enron_vectorizer.pkl"
)

print("\nModel and Vectorizer Saved Successfully!")

