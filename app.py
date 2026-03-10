from flask import Flask, request, render_template_string
import pickle
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

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

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Fake Job Detector</title>
</head>
<body style="font-family:Arial;text-align:center;margin-top:50px;">
<h2>Fake Job Detection System</h2>

<form method="post">
<textarea name="job_text" rows="6" cols="60" placeholder="Paste job description here"></textarea><br><br>
<button type="submit">Check Job</button>
</form>

{% if result %}
<h3>{{ result }}</h3>
{% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["job_text"]
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            result = "⚠ This Job Posting is FAKE"
        else:
            result = "✅ This Job Posting is REAL"

    return render_template_string(HTML_PAGE, result=result)

if __name__ == "__main__":
    app.run(debug=True)
