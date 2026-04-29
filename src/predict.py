import os
import pickle
import torch
from transformers import pipeline
from preprocess import clean_text

# ========= PATH SETUP =========
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ========= LOAD TF-IDF =========
with open(os.path.join(BASE_DIR, "models", "tfidf_model.pkl"), "rb") as f:
    tfidf_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vec = pickle.load(f)

# ========= LOAD BERT =========
bert_pipe = pipeline("sentiment-analysis")

# ========= LOAD LSTM =========
# Load vocab
with open(os.path.join(BASE_DIR, "models", "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

MAX_LEN = 256

def encode(text, vocab):
    tokens = text.split()[:MAX_LEN]
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0] * (MAX_LEN - len(ids))
    return ids

# Define same architecture (must match training)
class SentimentLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(20000, 128, padding_idx=0)
        self.lstm  = torch.nn.LSTM(128, 64, batch_first=True,
                                  num_layers=2, dropout=0.3, bidirectional=True)
        self.drop  = torch.nn.Dropout(0.4)
        self.fc    = torch.nn.Linear(64 * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)
        out = self.drop(out)
        return self.fc(out).squeeze(1)

# Load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lstm = SentimentLSTM().to(device)
model_lstm.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "lstm_model.pth"), map_location=device))
model_lstm.eval()

def predict_lstm(text):
    encoded = encode(clean_text(text), vocab)
    tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model_lstm(tensor)
        prob = torch.sigmoid(output).item()

    label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
    return label, prob

# ========= MAIN FUNCTION =========
def predict_all(text):
    cleaned = clean_text(text)

    # TF-IDF
    tfidf_pred = tfidf_model.predict(tfidf_vec.transform([cleaned]))[0]
    tfidf_label = "POSITIVE" if tfidf_pred == 1 else "NEGATIVE"

    # LSTM
    lstm_label, lstm_prob = predict_lstm(text)

    # BERT
    bert_result = bert_pipe(text)[0]

    print("\nText   :", text)
    print("TF-IDF :", tfidf_label)
    print("LSTM   :", lstm_label, f"({lstm_prob:.2%})")
    print("BERT   :", bert_result["label"], f"({bert_result['score']:.2%})")
    print("-" * 60)


# ========= RUN =========
if __name__ == "__main__":
    predict_all("This movie was absolutely fantastic, I loved every second!")
    predict_all("Complete waste of time. Terrible acting and boring plot.")
    predict_all("It was okay, nothing special but not terrible either.")