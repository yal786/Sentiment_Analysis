from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
from preprocess import clean_text

# Load dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]

texts = [clean_text(x["text"]) for x in train_data]
labels = [x["label"] for x in train_data]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Save
os.makedirs("../models", exist_ok=True)

with open("../models/tfidf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training complete and model saved.")