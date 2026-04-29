# 🎬 Sentiment Analysis — IMDB Movie Reviews

An end-to-end sentiment analysis system comparing **classical ML, deep learning, and transformer-based models** on the IMDB dataset.

---

## 🚀 Overview

This project implements and compares three different approaches:

- TF-IDF + Logistic Regression (baseline)
- Bidirectional LSTM (custom deep learning model)
- DistilBERT (pretrained transformer)

The goal is to understand performance differences between traditional and modern NLP techniques.

---

## 📊 Results

| Model                  | Accuracy | Notes                          |
|-----------------------|----------|--------------------------------|
| TF-IDF + Logistic Reg | ~85–89%  | Fast, interpretable baseline   |
| Bidirectional LSTM    | ~83–84%  | Learns sequence & context      |
| DistilBERT            | ~90–94%  | Best contextual understanding  |

---

## 🔥 Sample Output
Text : This movie was absolutely fantastic, I loved every second!
TF-IDF : POSITIVE
LSTM : POSITIVE (99.69%)
BERT : POSITIVE (99.99%)

Text : Complete waste of time. Terrible acting and boring plot.
TF-IDF : NEGATIVE
LSTM : NEGATIVE (0.11%)
BERT : NEGATIVE (99.98%)

Text : It was okay, nothing special but not terrible either.
TF-IDF : NEGATIVE
LSTM : NEGATIVE (2.40%)
BERT : POSITIVE (91.59%)


---

## 🧠 Key Learnings

- TF-IDF performs strongly as a baseline despite its simplicity  
- LSTM captures word order but requires tuning and more data  
- BERT handles context better (e.g., "not terrible") due to pretraining  
- Classical ML can compete with deep learning in smaller setups  

---

## ⚙️ Tech Stack

- Python  
- Scikit-learn  
- PyTorch  
- HuggingFace Transformers  
- NLTK  

---

## 📂 Project Structure
sentiment_analysis/
├── data/
├── models/
├── notebooks/
│ └── 01_explore.ipynb
├── src/
│ ├── preprocess.py
│ ├── train.py
│ └── predict.py
├── requirements.txt
└── README.md


---
Dataset is loaded dynamically using HuggingFace datasets library, so no raw data files are stored in the repository.
## ▶️ How to Run

### 1. Install dependencies

pip install -r requirements.txt


### 2. Run predictions

cd src
python predict.py


---

## 📦 Dataset

- IMDB Movie Reviews dataset (50,000 samples)  
- Loaded using HuggingFace `datasets` library  

---

## 💡 Future Improvements

- Hyperparameter tuning for LSTM  
- Fine-tuning BERT instead of using pretrained inference  
- Add web UI using Streamlit  
- Deploy as an API  

---

## 📌 Author

Built as part of AIML project practice and portfolio development.