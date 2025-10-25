# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# Load IMDB word index
word_index = imdb.get_word_index()

# Load model
model = load_model('simple_rnn_model.keras', compile=False)

# Helper functions
def clean_text(text):
    """Lowercase and remove non-alphabet characters"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

max_features = 10000
max_len = 500

def preprocess_input(text):
    text = clean_text(text)
    words = text.split()
    encoded = []
    for w in words:
        idx = word_index.get(w)
        if idx is not None and idx < max_features:
            encoded.append(idx + 3)
        else:
            encoded.append(2)  # <UNK>
    if not encoded:
        encoded = [2]
    return sequence.pad_sequences([encoded], maxlen=max_len)


def predict_sentiment(review):
    """Safe prediction: handles unknown/random input"""
    try:
        x = preprocess_input(review)
        p = float(model.predict(x, verbose =0)[0][0])
        return ("Positive" if p>0.5 else "Negative"),p
    except Exception:
        return "Error", 0.0

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to check if it is Negative or Positive:")

# User Input
user_input = st.text_area("Movie Review", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        sentiment, score = predict_sentiment(user_input)

        # Color-coded display
        if sentiment == "Positive":
            color = "green"
        elif sentiment == "Negative":
            color = "red"
        else:
            color = "gray"

        st.markdown(f"**Sentiment:** <span style='color:{color}; font-size:24px'>{sentiment}</span>", unsafe_allow_html=True)

        # Progress bar
        st.progress(score)

        st.write(f"Prediction Score: {score:.4f}")
else:
    st.info("Please provide a movie review to classify.")
