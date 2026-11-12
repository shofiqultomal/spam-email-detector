import pandas as  pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/processed_spam.csv")

print("\nEmail Category Distribution:")
print(df["label"].value_counts())


x = df["clean_message"]
y = df["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size= 0.3,random_state=42,stratify=y)
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train, y_train)
print("Model Training Completed!")

print("Training Samples:", x_train.shape[0])
print("Testing Samples:", x_test.shape[0])

# 7. Save Model
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model Saved Successfully!")

# Make Predictions
y_pred = model.predict(x_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))