import os
import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text_preprocessing import clean_text

# ------------------------------------------------------------
# Download NLTK Resources
# ------------------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ------------------------------------------------------------
# Initialize NLP Tools
# ------------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------
# Load Saved Model & Vectorizer
# ------------------------------------------------------------

current_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(current_folder)

models_folder = os.path.join(project_folder, "models")

model = joblib.load(
    os.path.join(models_folder, "best_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(models_folder, "tfidf_vectorizer.pkl")
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

print("=" * 60)
print("FAKE NEWS DETECTOR")
print("=" * 60)

# ------------------------------------------------------------
# User Input
# ------------------------------------------------------------

news = input("\nEnter News Article:\n\n")

# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------

cleaned_news = clean_text(news)

news_vector = vectorizer.transform([cleaned_news])

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

prediction = model.predict(news_vector)[0]

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(news_vector)[0]

    fake_confidence = probabilities[0] * 100
    real_confidence = probabilities[1] * 100

else:
    fake_confidence = 0
    real_confidence = 0

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

print("\n" + "=" * 60)

if prediction == 0:
    print("Prediction : FAKE NEWS")
    print(f"Confidence : {fake_confidence:.2f}%")
else:
    print("Prediction : REAL NEWS")
    print(f"Confidence : {real_confidence:.2f}%")

print("=" * 60)

# ------------------------------------------------------------
# Show Both Probabilities
# ------------------------------------------------------------

if hasattr(model, "predict_proba"):
    print("\nProbability Distribution")
    print(f"Fake : {fake_confidence:.2f}%")
    print(f"Real : {real_confidence:.2f}%")