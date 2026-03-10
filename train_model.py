import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (run once)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("fake_job_postings.csv").head(5000)

# Combine important text columns
data['text'] = data['title'].fillna('') + " " + \
               data['company_profile'].fillna('') + " " + \
               data['description'].fillna('') + " " + \
               data['requirements'].fillna('')

# Target column (0 = Real, 1 = Fake)
y = data['fraudulent']
X = data['text']

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)

X = X.apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")
