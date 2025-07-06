import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def load_kaggle_data(balance=True, sample_size=None):
    if not os.path.exists("True.csv") or not os.path.exists("Fake.csv"):
        print("❌ Missing True.csv or Fake.csv. Download from Kaggle:")
        print("👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        exit()

    print("📂 Loading Kaggle dataset...")
    true = pd.read_csv("True.csv")
    fake = pd.read_csv("Fake.csv")

    true["label"] = "REAL"
    fake["label"] = "FAKE"

    if balance:
        min_len = min(len(true), len(fake))
        true = true.sample(min_len)
        fake = fake.sample(min_len)

    data = pd.concat([true, fake], axis=0).sample(frac=1).reset_index(drop=True)

    if sample_size:
        data = data.head(sample_size)

    print(f"✅ Loaded {len(data)} articles.\n")
    return data

def train_model(df):
    print("📊 Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=100)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Accuracy: {round(acc * 100, 2)}%")
    print(classification_report(y_test, y_pred))

    return model, tfidf

def predict_news(text, model, vectorizer):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return prediction

def main():
    print("🧠 Fake News Detection - CLI Version")
    df = load_kaggle_data(balance=True, sample_size=10000)  # Small sample for faster training
    model, vectorizer = train_model(df)

    print("\n🧪 Ready to detect! Type a news headline or full story (type 'exit' to quit):\n")

    while True:
        news = input("📝 Your news: ")
        if news.strip().lower() == 'exit':
            print("👋 Exiting. Stay smart. Stay informed.")
            break
        if len(news.strip()) < 20:
            print("⚠️ Please enter more detailed text.\n")
            continue

        prediction = predict_news(news, model, vectorizer)
        icon = "📰" if prediction == "REAL" else "🚨"
        print(f"🔎 Prediction: {prediction} {icon}\n")

if __name__ == "__main__":
    main()
