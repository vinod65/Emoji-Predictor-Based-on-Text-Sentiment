import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
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

# DataFrame
df = pd.DataFrame(data)

# Vectorization
X = df['text']
y = df['emoji']
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict emoji from new sentence
def predict_emoji(text):
    text_vector = vectorizer.transform([text])
    predicted = model.predict(text_vector)[0]
    return predicted

# Test predictions from user input
while True:
    user_input = input("Type a sentence (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    emoji = predict_emoji(user_input)
    print(f"Predicted Emoji: {emoji}")



