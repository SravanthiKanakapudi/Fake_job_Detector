import pickle
import re
import nltk
from nltk.corpus import stopwords

# load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# user input
job_text = input("Enter job description: ")

cleaned = clean_text(job_text)

vectorized = vectorizer.transform([cleaned])

prediction = model.predict(vectorized)

if prediction[0] == 1:
    print("⚠️ This Job Posting is FAKE")
else:
    print("✅ This Job Posting is REAL")
