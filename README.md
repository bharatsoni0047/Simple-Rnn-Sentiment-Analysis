# 🎬 IMDB Sentiment Analysis with Simple RNN

This project performs **sentiment analysis** on IMDB movie reviews using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras.  
The model predicts whether a given review is **Positive 😊** or **Negative 😞**.

here is streamlit app link:- [https://simple-rnn-by-bharat.streamlit.app/]
---

## 🧠 Model Overview
- **Architecture:** Embedding → SimpleRNN → Dense  
- **Dataset:** IMDB (50,000 labeled reviews)  
- **Accuracy:** ~84% (10 epochs)  
- **Framework:** TensorFlow 2.20 / Keras 3  
- **Model File:** `simple_rnn_model.keras`

---

## ⚙️ Setup & Installation
bash
git clone https://github.com/bharatsoni0047/simple-rnn-sentiment-analysis.git
cd simple-rnn-sentiment-analysis
python -m venv venv
venv\Scripts\activate     # On Windows
pip install -r requirements.txt

## Example
| Review                    | Sentiment  |
| ------------------------- | ---------- |
| "The movie was awesome!"  | ✅ Positive |
| "It was boring and slow." | ❌ Negative |
| "bad"                     | ❌ Negative |
