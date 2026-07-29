import os
import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text_preprocessing import clean_text

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

print("Libraries Imported Successfully")
print("NLTK Resources Downloaded Successfully")

# ---------------------------------------------------
# Locate dataset folder
# ---------------------------------------------------

current_folder = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(current_folder, "..", "dataset")

fake_path = os.path.join(dataset_folder, "Fake.csv")
true_path = os.path.join(dataset_folder, "True.csv")

print("\nDataset Folder :", dataset_folder)
print("Fake Dataset   :", fake_path)
print("True Dataset   :", true_path)

# ---------------------------------------------------
# Read datasets
# ---------------------------------------------------

fake_news = pd.read_csv(fake_path)
true_news = pd.read_csv(true_path)

print("\nDatasets Loaded Successfully!")

print("\nFake Dataset Shape :", fake_news.shape)
print("True Dataset Shape :", true_news.shape)

# ---------------------------------------------------
# Add Labels
# ---------------------------------------------------

fake_news["label"] = 0
true_news["label"] = 1

print("\nLabels Added Successfully!")

# ---------------------------------------------------
# Merge Datasets
# ---------------------------------------------------

news_data = pd.concat([fake_news, true_news], ignore_index=True)

print("Datasets Merged Successfully!")

print("\nCombined Dataset Shape :", news_data.shape)

print("\nFirst 5 Rows:")
print(news_data.head())

# ---------------------------------------------------
# Combine Title and Text
# ---------------------------------------------------

news_data["content"] = news_data["title"] + " " + news_data["text"]

print("\nContent Column Created Successfully!")

print("\nSample Content:")
print(news_data[["content", "label"]].head())

# ---------------------------------------------------
# Check for Missing Values
# ---------------------------------------------------

print("\nMissing Values in Dataset:")
print(news_data.isnull().sum())

# ---------------------------------------------------
# Initialize NLP Tools
# ---------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------
# Apply Text Cleaning
# ---------------------------------------------------

print("\nCleaning text... Please wait.")

news_data["content"] = news_data["content"].apply(clean_text)

print("Text Cleaning Completed Successfully!")

print("\nSample Cleaned Content:")
print(news_data[["content", "label"]].head())

