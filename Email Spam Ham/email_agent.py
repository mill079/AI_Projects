import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI
st.title("📧 AI Email Classifier Agent")
st.write("Classify emails as Spam or Ham using LLaMA 3 (Groq API)")

# Input box
email_input = st.text_area("✉️ Paste the email content here:", height=200)

# Classification logic
if st.button("🚀 Classify Email"):
    if not email_input.strip():
        st.warning("Please paste some email content.")
    else:
        # Constructing prompt for LLM
        prompt = [
            {
                "role": "user",
                "content": f"""
You are a smart AI email classifier. Classify the following email as either "spam" or "ham" based on its content.

- Label as **spam** if the email contains: irrelevant offers, phishing, scams, advertisements, or nonsense content.
- Label as **ham** if it's useful, relevant, or legitimate communication.

### Email Content:
{email_input}

### Return only a JSON object in this format:
{{
  "email_type": "spam" or "ham",
  "reason": "Short reason why this is spam or ham"
}}
"""
            }
        ]

        # Query Groq's LLaMA3 model
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=prompt,
                max_tokens=500,
                temperature=0.3,
            )
            content = response.choices[0].message.content

            # Try to parse JSON output
            try:
                result = json.loads(content)
                st.json(result)
            except json.JSONDecodeError:
                st.error("The response is not a valid JSON. Output received:")
                st.write(content)

        except Exception as e:
            st.error(f"Failed to get classification: {e}")
