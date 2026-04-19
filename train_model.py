import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load data
fake = pd.read_csv(r"C:\Users\UJJWAL\Downloads\fake news detection\dataset\fake.csv")
true = pd.read_csv(r"C:\Users\UJJWAL\Downloads\fake news detection\dataset\true.csv")

fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true])

print(f"Total records: {len(data)}")
print(f"Fake news: {(data['label'] == 0).sum()}")
print(f"True news: {(data['label'] == 1).sum()}")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data["text"] = data["text"].apply(clean_text)

# FIXED: Proper train-test split (should return 4 values)
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# Training Accuracy
train_pred = model.predict(X_train_vec)
train_accuracy = accuracy_score(y_train, train_pred)
train_precision = precision_score(y_train, train_pred)
train_recall = recall_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)

print(f"\n{'='*50}")
print(f"TRAINING METRICS:")
print(f"{'='*50}")
print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1-Score:  {train_f1:.4f}")

# TEST ACCURACY (ADDED)
test_pred = model.predict(X_test_vec)
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)

print(f"\n{'='*50}")
print(f"TEST METRICS:")
print(f"{'='*50}")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Save model and vectorizer
pickle.dump(model, open(r"C:\Users\UJJWAL\Downloads\fake news detection\dataset\model.pkl", "wb"))
pickle.dump(vectorizer, open(r"C:\Users\UJJWAL\Downloads\fake news detection\dataset\vectorizer.pkl", "wb"))

print(f"\n✅ Model and vectorizer saved successfully!")
