# ==========================
# 🔥 CRASH FIX
# ==========================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ==========================
# IMPORTS
# ==========================
import torch, pickle, json, random, base64
from flask import Flask, render_template, request, jsonify, redirect
from transformers import BertTokenizer, BertForSequenceClassification
from deep_translator import GoogleTranslator
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from gtts import gTTS
from io import BytesIO

# ==========================
# APP CONFIG
# ==========================
app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ==========================
# DATABASE MODELS
# ==========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    title = db.Column(db.String(200), default="New Chat")

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer)
    message = db.Column(db.Text)
    response = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ==========================
# LOAD MODEL
# ==========================
device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained("bert_chatbot_model")
tokenizer = BertTokenizer.from_pretrained("bert_chatbot_model")
model.to(device)
model.eval()

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("DataSet/intents.json", encoding="utf-8") as f:
    data = json.load(f)

LANG_MAP = {"ta-IN": "ta", "hi-IN": "hi", "en-US": "en"}

# ==========================
# TRANSLATION
# ==========================
def translate_to_english(text, lang):
    try:
        if lang in ["ta", "hi"]:
            text = GoogleTranslator(source='auto', target='en').translate(text)
    except:
        pass
    return str(text).lower().strip()

def translate_to_user_lang(text, lang):
    try:
        if lang == "ta":
            return GoogleTranslator(source='en', target='ta').translate(text)
        elif lang == "hi":
            return GoogleTranslator(source='en', target='hi').translate(text)
    except:
        pass
    return text

# ==========================
# INTENT
# ==========================
def predict_intent(text):
    text = text.lower().strip()

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in text:
                return intent["tag"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    _, predicted = torch.max(probs, dim=1)

    return le.inverse_transform([predicted.item()])[0]

def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand."

# ==========================
# VOICE
# ==========================
def generate_voice(text, lang):
    try:
        tts = gTTS(text=text[:200], lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode()
    except:
        return None

# ==========================
# AUTH ROUTES
# ==========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = User(
            username=request.form['username'],
            password=generate_password_hash(request.form['password'])
        )
        db.session.add(user)
        db.session.commit()
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect("/")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# ==========================
# MAIN
# ==========================
@app.route("/")
@login_required
def home():
    return render_template("Chat.html")

# ==========================
# ADMIN
# ==========================
@app.route("/admin")
@login_required
def admin():
    return render_template(
        "admin.html",
        users=User.query.count(),
        sessions=ChatSession.query.count(),
        messages=Message.query.count()
    )

# ==========================
# SESSION ROUTES (🔥 FIX)
# ==========================
@app.route("/sessions")
@login_required
def sessions():
    s = ChatSession.query.filter_by(user_id=current_user.id).all()
    return jsonify([{"id": i.id, "title": i.title} for i in s])

@app.route("/new_chat")
@login_required
def new_chat():
    s = ChatSession(user_id=current_user.id)
    db.session.add(s)
    db.session.commit()
    return jsonify({"id": s.id})

@app.route("/messages/<int:id>")
@login_required
def messages(id):
    msgs = Message.query.filter_by(session_id=id).all()
    return jsonify([{"m": m.message, "r": m.response} for m in msgs])

@app.route("/delete/<int:id>")
@login_required
def delete(id):
    Message.query.filter_by(session_id=id).delete()
    ChatSession.query.filter_by(id=id).delete()
    db.session.commit()
    return "ok"

# ==========================
# CHAT
# ==========================
@app.route("/ask", methods=["POST"])
@login_required
def ask():
    msg = request.form['messageText']
    session_id = request.form.get("session_id")

    lang_code = request.form.get('lang', 'en-US')
    lang = LANG_MAP.get(lang_code, "en")

    en = translate_to_english(msg, lang)
    tag = predict_intent(en)
    res = get_response(tag)
    final = translate_to_user_lang(res, lang)

    # 🔥 SAVE CHAT
    if session_id:
        db.session.add(Message(
            session_id=session_id,
            message=msg,
            response=final
        ))
        db.session.commit()

    audio = generate_voice(final, lang) if lang in ["ta", "hi"] else None

    return jsonify({"answer": final, "audio": audio})

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)