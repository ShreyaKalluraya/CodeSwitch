from flask import Flask, render_template, request
import pickle

from modules.preprocessing import tokenize
from modules.feature_extraction import extract_features
from modules.translation import translate_word



app = Flask(__name__)

# Load trained ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        sentence = request.form["sentence"]
        return render_template("result.html", sentence=sentence)
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    sentence = request.form["sentence"]
    words = tokenize(sentence)

    results = []

    for word in words:
        X = extract_features([word.lower()])
        lang = model.predict(X)[0]
        results.append((word, lang))

    return render_template(
        "detect.html",
        sentence=sentence,
        results=results
    )


'''@app.route("/translate", methods=["POST"])
def translate():
    sentence = request.form["sentence"]

    # word-level detection (for display)
    words = tokenize(sentence)
    results = []

    for word in words:
        X = extract_features([word.lower()])
        lang = model.predict(X)[0]
        results.append((word, lang))

    # TEMP FIX: avoid undefined function error
    final_translation = sentence

    return render_template(
        "translate.html",
        sentence=sentence,
        results=results,
        translation=final_translation
    )'''

'''@app.route("/translate", methods=["POST"])
def translate():
    sentence = request.form.get("sentence", "").strip()

    if not sentence:
        return render_template(
            "translate.html",
            sentence="",
            results=[],
            kannada_percent=0,
            english_percent=0,
            dominance="No input",
            pattern="No pattern",
            intent="No intent"
        )

    # -------- WORD LEVEL DETECTION --------
    words = tokenize(sentence)
    results = []

    for word in words:
        X = extract_features([word.lower()])
        lang = model.predict(X)[0]
        results.append((word, lang))

    # -------- LANGUAGE DOMINANCE --------
    kannada_count = 0
    english_count = 0

    for word, lang in results:
        if lang == "kn":
            kannada_count += 1
        elif lang == "en":
            english_count += 1

    total_words = kannada_count + english_count

    if total_words > 0:
        kannada_percent = round((kannada_count / total_words) * 100, 2)
        english_percent = round((english_count / total_words) * 100, 2)
    else:
        kannada_percent = 0
        english_percent = 0

    if kannada_percent >= 70:
        dominance = "Kannada-dominant"
    elif english_percent >= 70:
        dominance = "English-dominant"
    else:
        dominance = "Highly Code-Mixed"

    # -------- CODE-MIXING PATTERN ANALYSIS --------
    if kannada_percent >= 90 or english_percent >= 90:
        pattern = "Low Code-Mixing"
    elif 40 <= kannada_percent <= 60:
        pattern = "High Code-Mixing"
    else:
        pattern = "Moderate Code-Mixing"

    # -------- INTENT DETECTION --------
    sentence_lower = sentence.lower()
    intent = "Statement / Casual"   # ✅ DEFAULT (IMPORTANT)

    if "?" in sentence or any(
        q in sentence_lower
        for q in ["what", "why", "how", "when", "where", "yen", "yaake", "hege"]
    ):
        intent = "Question"
    elif any(
    r in sentence_lower
    for r in ["please", "plz", "help", "beku", "kodi", "madsi"]
    ):
        intent = "Request"
    elif any(
    g in sentence_lower
    for g in ["hi", "hello", "hey", "hai", "namaskara"]
    ):
        intent = "Greeting"

    # -------- SOCIAL MEDIA TEXT CLASSIFICATION --------
    if intent == "Greeting":
        social_type = "Conversational Chat"
    elif intent == "Question":
        social_type = "Question Post"
    elif intent == "Request":
        social_type = "Help / Request Post"
    elif dominance == "English-dominant":
        social_type = "Informative / Announcement"
    else:
        social_type = "Casual Social Media Post"


    return render_template(
    "translate.html",
    sentence=sentence,
    results=results,
    dominance=dominance,
    kannada_percent=kannada_percent,
    english_percent=english_percent,
    intent=intent,
    pattern=pattern,          # 🔥 ADD THIS
    social_type=social_type
)'''

@app.route("/translate", methods=["POST"])
def translate():
    sentence = request.form.get("sentence", "").strip()

    if not sentence:
        return render_template(
            "translate.html",
            sentence="",
            results=[],
            dominance="No input",
            kannada_percent=0,
            english_percent=0,
            intent="No intent",
            social_type="Unknown"
        )

    # ---------------- WORD-LEVEL DETECTION ----------------
    words = tokenize(sentence)
    results = []

    for word in words:
        X = extract_features([word.lower()])
        lang = model.predict(X)[0]
        results.append((word, lang))

    # ---------------- LANGUAGE DOMINANCE ----------------
    kn = sum(1 for w, l in results if l == "kn")
    en = sum(1 for w, l in results if l == "en")
    total = kn + en

    kannada_percent = round((kn / total) * 100, 2) if total else 0
    english_percent = round((en / total) * 100, 2) if total else 0

    if kannada_percent >= 70:
        dominance = "Kannada-dominant"
    elif english_percent >= 70:
        dominance = "English-dominant"
    else:
        dominance = "Highly Code-Mixed"

    # -----------CODE-MIXING PATTERN ANALYSIS--------
    if english_percent ==0:
        pattern = "Pure Kannada (Monolingual)"
    elif kannada_percent ==0:
        pattern = "Pure English (Monolingual)"
    elif kannada_percent < 30 or english_percent < 30:
        pattern = "Low Code-Mixing"
    elif 30 <= kannada_percent <=60 :
        pattern = "Moderate Code-Mixing"
    else:
        pattern = "Heavy Code-Mixing"

    # ---------------- INTENT DETECTION ----------------
    s = sentence.lower()

    if "?" in sentence or any(q in s for q in ["what", "why", "how", "when", "where", "yenu","yen", "yaake", "hege","yellige"]):
        intent = "Question"

    elif any(r in s for r in ["please", "plz", "help", "beku", "kodi", "madsi"]):
        intent = "Request"

    elif any(g in s for g in ["hi", "hello", "hey", "hai", "namaskara","namaste"]):
        intent = "Greeting"

    elif any(o in s for o in ["think", "feel", "ansutte", "nanage ansutte", "in my opinion"]):
        intent = "Opinion"

    elif any(c in s for c in ["bad", "worst", "problem", "issue", "bekagilla", "sari illa"]):
        intent = "Complaint"

    elif any(a in s for a in ["good", "great", "awesome", "super", "thanks", "thank you", "tumba chennagide"]):
        intent = "Appreciation"

    elif any(sg in s for sg in ["should", "better", "try", "recommend", "suggest"]):
        intent = "Suggestion"

    elif any(e in s for e in ["happy", "sad", "angry", "excited", "tired"]):
        intent = "Emotion"

    elif any(cmd in s for cmd in ["go", "come", "stop", "listen", "look"]):
        intent = "Command"

    else:
        intent = "Informative / Statement"

    
    # -------- SOCIAL MEDIA TEXT CLASSIFICATION (STYLE-BASED) --------
    word_count = len(words)
    exclamation_count = sentence.count("!")
    question_count = sentence.count("?")
    emoji_present = any(char in sentence for char in ["😂", "😅", "😊", "🔥", "❤️"])
    social_marker_present = "#" in sentence or "@" in sentence

    if word_count <= 3:
        social_type = "Short Chat / Reply"
    elif emoji_present or exclamation_count >= 2:
        social_type = "Expressive / Emotional Post"
    elif social_marker_present:
        social_type = "Social Media Engagement Post"
    elif word_count >= 12:
        social_type = "Descriptive / Informative Post"
    else:
        social_type = "Casual Conversational Post"


    return render_template(
        "translate.html",
        sentence=sentence,
        results=results,
        dominance=dominance,
        kannada_percent=kannada_percent,
        english_percent=english_percent,
        intent=intent,
        pattern=pattern,
        social_type=social_type
    )


if __name__ == "__main__":
    app.run(debug=True)
