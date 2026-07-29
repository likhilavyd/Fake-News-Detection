import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

current_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(current_folder)

dataset_folder = os.path.join(project_folder, "dataset")
models_folder = os.path.join(project_folder, "models")
outputs_folder = os.path.join(project_folder, "outputs")

os.makedirs(models_folder, exist_ok=True)
os.makedirs(outputs_folder, exist_ok=True)

processed_file = os.path.join(dataset_folder, "processed_news.csv")

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

news_data = pd.read_csv(processed_file)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", news_data.shape)

# ------------------------------------------------------------
# Features & Labels
# ------------------------------------------------------------

X = news_data["content"]
y = news_data["label"]

# ------------------------------------------------------------
# Split Dataset FIRST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------

print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("TF-IDF Completed")
print("Training Matrix :", X_train.shape)
print("Testing Matrix  :", X_test.shape)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Save Logistic Regression for deployment
    if name == "Logistic Regression":
        best_model = model
        best_model_name = name
        best_accuracy = accuracy

# ------------------------------------------------------------
# Save Results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

results_file = os.path.join(outputs_folder, "model_results.csv")
results_df.to_csv(results_file, index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df)

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------

joblib.dump(
    best_model,
    os.path.join(models_folder, "best_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(models_folder, "tfidf_vectorizer.pkl")
)

print("\nSaved Model :", best_model_name)
print("Accuracy    :", round(best_accuracy, 4))

print("\nBest model saved.")
print("TF-IDF Vectorizer saved.")
print("Results saved.")
print("\nTraining Complete.")