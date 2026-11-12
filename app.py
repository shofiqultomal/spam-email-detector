
import streamlit as st
import joblib
import re

# Load saved model
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text

# Dashboard
st.title("Spam Email Detector")
st.write("Enter an email to detect whether it is Spam or Not Spam.")

email = st.text_area("Enter Email Message")

if st.button("Check Email"):

    if email.strip():

        cleaned_email = clean_text(email)

        numerical_email = vectorizer.transform([cleaned_email])

        prediction = model.predict(numerical_email)

        if prediction[0] == "spam":
            st.error("Warning! This is a Spam Email.")

        else:
            st.success("This is a Not Spam Email.")

    else:
        st.warning("Please enter an email.")
