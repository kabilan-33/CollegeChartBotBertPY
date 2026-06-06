import torch
import pickle
import json
import random
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# ==========================
# LOAD MODEL
# ==========================
model = BertForSequenceClassification.from_pretrained("bert_chatbot_model")
tokenizer = BertTokenizer.from_pretrained("bert_chatbot_model")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("DataSet/intents.json", encoding="utf-8") as f:
    intents = json.load(f)

model.eval()

# ==========================
# RESPONSE FUNCTION
# ==========================
def get_response(user_input):

    user_input_lower = user_input.lower().strip()

    tag = None
    confidence = 0.0

    # ==========================
    # 🔥 1. EXACT PATTERN MATCH (100% GUARANTEED)
    # ==========================
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input_lower:
                tag = intent["tag"]
                confidence = 1.0
                break
        if tag is not None:
            break

    # ==========================
    # 🔥 2. KEYWORD RULES (BACKUP)
    # ==========================
    if tag is None:

        # ==========================
        # 🔥 IMPROVED KEYWORD RULES
        # ==========================

        if any(w in user_input_lower for w in ["hi", "hello", "hey"]):
            tag = "greeting";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["bye", "goodbye", "exit", "quit"]):
            tag = "goodbye";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["creator", "developer"]):
            tag = "creator";
            confidence = 1.0

        # ✅ FIXED (important)
        elif "name" in user_input_lower or "who are you" in user_input_lower:
            tag = "name";
            confidence = 1.0

        elif "about" in user_input_lower:
            tag = "about";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["course", "courses"]):
            tag = "courses";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["fee", "fees"]):
            tag = "fees_overview";
            confidence = 1.0

        # ✅ FIXED (important)
        elif any(w in user_input_lower for w in ["scholarship", "scholorship", "schlorsship"]):
            tag = "scholarships";
            confidence = 1.0

        elif "hostel" in user_input_lower:
            tag = "hostel";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["syllabus", "subjects"]):
            tag = "syllabus";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["canteen", "food"]):
            tag = "canteen";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["placement", "job", "salary"]):
            tag = "placement";
            confidence = 1.0

        elif "principal" in user_input_lower:
            tag = "principal";
            confidence = 1.0

        elif "cutoff" in user_input_lower:
            tag = "cutoff";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["admission", "apply"]):
            tag = "admission_process";
            confidence = 1.0

        # ✅ FIXED (typo support)
        elif any(w in user_input_lower for w in ["eligibility", "elibility"]):
            tag = "eligibility_criteria";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["transport", "bus"]):
            tag = "transport";
            confidence = 1.0

        elif "hod" in user_input_lower:
            tag = "hod";
            confidence = 1.0

        # ✅ FIXED (typo support)
        elif any(w in user_input_lower for w in ["faculty", "facult", "staff"]):
            tag = "faculty";
            confidence = 1.0

        elif any(w in user_input_lower for w in ["contact", "phone", "email", "address"]):
            tag = "contact";
            confidence = 1.0

    # ==========================
    # 🤖 3. BERT FALLBACK
    # ==========================
    if tag is None:
        inputs = tokenizer(user_input_lower, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

        confidence = confidence.item()
        tag = le.inverse_transform(predicted.cpu().numpy())[0]

    # ==========================
    # DEBUG
    # ==========================
    print(f"[DEBUG] Tag: {tag} | Confidence: {confidence:.2f}")

    # ==========================
    # RESPONSE
    # ==========================
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand."


# ==========================
# CHAT LOOP
# ==========================
print("🤖 Chatbot Ready! Type 'exit' to stop.\n")

while True:
    msg = input("You: ")

    if msg.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    response = get_response(msg)
    print("Bot:", response)