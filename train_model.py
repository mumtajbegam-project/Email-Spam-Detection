import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

# Select required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Clean dataset
df = df.dropna()
df["label"] = df["label"].str.lower()

# Input and output
X = df["message"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, "spam_model.pkl")

print("Model saved successfully!")
