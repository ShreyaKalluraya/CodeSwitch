import pandas as pd
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from modules.feature_extraction import vectorizer

# Load dataset
data = pd.read_csv("dataset/words_dataset.csv")

words = data["word"].astype(str)
labels = data["label"]

# Vectorize
X = vectorizer.fit_transform(words)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# SAVE BOTH
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"✅ Model Accuracy: {acc*100:.2f}%")
print("✅ Model and Vectorizer saved")
