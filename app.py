from flask import Flask, request, render_template
import pickle
import os
import re

app = Flask(__name__)

base_dir = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_dir, "dataset/model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(base_dir, "dataset/vectorizer.pkl"), "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        result = "REAL NEWS ✅" if pred == 1 else "FAKE NEWS ❌"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)