# Import libraries
import numpy as np
import tensorflow as tf
import streamlit as st
import re

# ✅ FIX: use Keras 3 direct imports (instead of tensorflow.keras)
from keras.datasets import imdb          # was: from tensorflow.keras.datasets import imdb
from keras.preprocessing import sequence # was: from tensorflow.keras.preprocessing import sequence
from keras.models import load_model      # was: from tensorflow.keras.models import load_model

# ✅ FIX: set page config early (avoids rerun glitches)
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.caption(f"✅ TensorFlow: {tf.__version__}")

# Load IMDB word index
word_index = imdb.get_word_index()

# ✅ FIX: cache the model so it loads only once (Streamlit reruns the script on every interaction)
@st.cache_resource
def get_model():
    try:
        # You can also switch to a SavedModel dir if you saved one:
        # return tf.keras.models.load_model("simple_rnn_saved", compile=False)
        return load_model("simple_rnn_model.keras", compile=False)  # Keras 3 loader
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        return None

model = get_model()

# Helper functions
def clean_text(text):
    """Lowercase and remove non-alphabet characters"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
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
    # ✅ FIX: handle empty inputs safely
    if not encoded:
        encoded = [2]
    return sequence.pad_sequences([encoded], maxlen=max_len)

def predict_sentiment(review):
    """Safe prediction: handles unknown/random input"""
    try:
        if model is None:
            return "Model not loaded", 0.0
        x = preprocess_input(review)
        p = float(model.predict(x, verbose=0)[0][0])
        return ("Positive" if p > 0.5 else "Negative"), p
    except Exception as e:
        # ✅ FIX: never crash — return safe output
        print("⚠️ Prediction error:", e)
        return "Error", 0.0

# UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to check if it is Negative or Positive:")

user_input = st.text_area("Movie Review", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        sentiment, score = predict_sentiment(user_input)
        color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"
        st.markdown(
            f"**Sentiment:** <span style='color:{color}; font-size:24px'>{sentiment}</span>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(score, 0.0), 1.0))  # ✅ FIX: keep score in [0,1]
        st.write(f"Prediction Score: {score:.4f}")
else:
    st.info("Please provide a movie review to classify.")
