import streamlit as st
import pickle
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# App title and instructions
st.title("🧠 Sentiment Analysis App")
st.write("This app predicts whether the input text has a Positive or Negative sentiment.")

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as file:
    senti_model = pickle.load(file)

with open('sentiment_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# User input
user_input = st.text_input("✍️ Enter your text here:")

# Predict button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        # Preprocess and predict
        preprocessed_text = preprocess(user_input)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = senti_model.predict(vectorized_text)
        confidence = max(senti_model.predict_proba(vectorized_text)[0]) * 100

        # Show only Positive or Negative
        if prediction[0] == 'positive':
            st.success(f"😊 Positive — Confidence: {confidence:.2f}%")
        else:
            st.error(f"😞 Negative — Confidence: {confidence:.2f}%")
