import streamlit as st
import pickle
import re
import nltk
import json
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources (only first time)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# Load model and vectorizer
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Label map
label_map = {0: "ham", 1: "spam"}

# Streamlit UI
st.title("📧 Spam vs Ham Email Classifier")
st.write("This app classifies your input email as **Spam** or **Ham** and explains why.")

# User input
email_text = st.text_area("✍️ Enter email content:")

# Predict button
if st.button("🔍 Predict"):
    if not email_text.strip():
        st.warning("Please enter some email text.")
    else:
        # Preprocess
        cleaned_text = preprocess(email_text)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)

        # Map and create JSON output
        predicted_label = int(prediction[0])
        email_type = label_map.get(predicted_label, "unknown")

        # Heuristic reason (you can improve this with SHAP/LIME later)
        reason = (
            "contains suspicious or promotional keywords" if email_type == "spam"
            else "does not contain spam-related patterns"
        )

        result = {
            "email_type": email_type,
            "reason": reason
        }

        # Show result
        st.subheader("📬 Classification Result (JSON):")
        st.json(result)

        # Optional: display simple feedback
        if email_type == "spam":
            st.error("🚨 This is classified as SPAM.")
        else:
            st.success("✅ This is classified as HAM.")
