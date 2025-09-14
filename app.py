from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
import os
from urllib.parse import unquote

app = Flask(__name__)

data = {
    "text": [
    "I love this movie, it’s fantastic!",
    "This was the worst film I’ve ever seen.",
    "Absolutely amazing experience, highly recommend!",
    "Terrible acting and boring storyline.",
    "Best movie of the year!",
    "I hated it so much.",
    "It was okay, not great but not bad.",
    "I enjoyed the film, very entertaining.",
    "Waste of time, completely disappointing.",
    "Such a wonderful and heartwarming story!",
    "Not good, very boring",
    "Absolutely loved it!",
    "Could have been better"
    ],
    # 1 => positive, 0 => negative/neutral
    "label": [1,0,1,0,1,0,0,1,0,1,0,1,0]
}
df = pd.DataFrame(data)


vectorizer = CountVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = MultinomialNB()
model.fit(X, y)

MOVIES = [
    "Inception",
    "The Godfather",
    "Avengers: Endgame",
    "Titanic",
    "Parasite"
]

SUGGESTIONS = {
    "Inception": [
        "Mind-bending and brilliant!",
        "Confusing and overrated",
        "Great visuals and story",
        "Too complicated for me",
        "Absolutely loved the soundtrack"
    ],
    "The Godfather": [
        "A masterpiece of cinema",
        "Slow and boring",
        "Perfect acting and direction",
        "Not my kind of movie",
        "Classic gangster storytelling"
    ],
    "Avengers: Endgame": [
        "Epic and emotional",
        "Too long and messy",
        "Best superhero movie",
        "Not as good as previous ones",
        "Amazing fan-service"
    ],
    "Titanic": [
        "Heartbreaking and beautiful",
        "Romantic but slow",
        "A timeless love story",
        "Too sentimental",
        "Great performances"
    ],
    "Parasite": [
        "Sharp, dark and brilliant",
        "Weird and uncomfortable",
        "Perfectly crafted thriller",
        "Not for everyone",
        "Masterful social commentary"
    ]
}

VOTES_FILE = "votes.json"

def load_votes():
    if not os.path.exists(VOTES_FILE):
        init = {m: [] for m in MOVIES}
        with open(VOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(init, f, indent=2)
        return init
    try:
        with open(VOTES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {m: [] for m in MOVIES}
    
    for m in MOVIES:
        if m not in data:
            data[m] = []
            
    return data
        
def save_votes(votes):
    with open(VOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(votes, f, indent=2)


def movie_rating_info(votes_list):
    total = len(votes_list)
    if total == 0:
        return {"stars": 0.0, "percent_positive": 0.0, "votes": 0}
    positives = sum(votes_list)
    ratio = positives / total
    stars = round(ratio * 5, 2) # scale to 0-5
    return {"stars": stars, "percent_positive": round(ratio * 100, 1), "votes": total}


@app.route("/")
def index():
    votes = load_votes()
    movies_info = []
    for m in MOVIES:
        info = movie_rating_info(votes.get(m, []))
        movies_info.append({"title": m, **info})
    return render_template("index.html", movies=movies_info)


@app.route("/movie/<title>", methods=["GET", "POST"])
def movie_page(title):
    title = unquote(title)
    if title not in MOVIES:
        return redirect(url_for('index'))
    
    
    votes = load_votes()
    message = None
    prediction_text = None
    prediction_label = None
    if request.method == "POST":
        custom_review = request.form.get("custom_review", "").strip()
        selected = request.form.get("suggestion") # may be None
        review_text = custom_review if custom_review else selected
        if not review_text:
            message = "Please select or type a review."
        else:
            vec = vectorizer.transform([review_text])
            pred = int(model.predict(vec)[0]) # 1 or 0
            prediction_label = pred
            prediction_text = "Positive 😀" if pred == 1 else "Negative 😡"
            message = f"Model prediction: {prediction_text}"
            # store vote
            votes.setdefault(title, []).append(pred)
            save_votes(votes)
    
    votes = load_votes()
    info = movie_rating_info(votes.get(title, []))
    suggestions = SUGGESTIONS.get(title, [])
    return render_template("movie.html",
                            title=title,
                            suggestions=suggestions,
                            info=info,
                            message=message,
                            prediction_text=prediction_text,
                            prediction_label=prediction_label)
    
if __name__ == "__main__":
    load_votes()
    app.run(debug=True)


