import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Data ---
data = {
    'text': [
        "I am so happy today!",
        "This is the worst day ever.",
        "I love my dog.",
        "Feeling very sad and lonely.",
        "What a beautiful view!",
        "I'm scared of the dark.",
        "Yay! I passed my exam!",
        "He betrayed me.",
        "Let's go party tonight!",
        "I miss you so much."
    ],
    'emoji': [
        "😊", "😞", "🐶", "😢", "😍",
        "😱", "🎉", "💔", "🥳", "😭"
    ]
}

df = pd.DataFrame(data)
X = df['text']
y = df['emoji']

# --- ML ---
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vectorized, y)

# --- Streamlit UI ---
st.set_page_config(page_title="Emoji Predictor", page_icon="🧠")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .emoji {
        font-size: 80px;
        text-align: center;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Emoji Predictor")
st.subheader("Type a sentence and get an emoji that matches your emotion!")

user_input = st.text_input("👉 Enter your sentence here:")

if user_input:
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    st.markdown(f"<div class='emoji'>{prediction}</div>", unsafe_allow_html=True)

