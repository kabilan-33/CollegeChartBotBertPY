import json
import torch
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# DEVICE
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# LOAD DATASET
# ==========================
with open("DataSet/intents.json", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.strip().lower()

        # ❌ remove junk / empty
        if pattern and len(pattern) > 1:
            sentences.append(pattern)
            labels.append(intent['tag'])

# ==========================
# LABEL ENCODING
# ==========================
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# save encoder (important for prediction)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

num_labels = len(le.classes_)

print("Total classes:", num_labels)

# ==========================
# TRAIN / VALIDATION SPLIT
# ==========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    sentences,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

# ==========================
# TOKENIZER
# ==========================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=32)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=32)

# ==========================
# DATASET CLASS
# ==========================
class ChatDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ChatDataset(train_encodings, train_labels)
val_dataset = ChatDataset(val_encodings, val_labels)

# ==========================
# MODEL
# ==========================
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels
)

model.to(device)

# ==========================
# TRAINING ARGUMENTS
# ==========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,   # 🔥 slightly increased
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=2e-5,
    weight_decay=0.01,

    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",

    report_to=[]
)

# ==========================
# TRAINER
# ==========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ==========================
# TRAIN
# ==========================
print("🚀 Training Started...\n")
trainer.train()

# ==========================
# EVALUATION
# ==========================
print("\n📊 Evaluating...\n")

predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)

print("\nClassification Report:\n")
print(classification_report(val_labels, preds, zero_division=0))

# ==========================
# CONFUSION MATRIX
# ==========================
cm = confusion_matrix(val_labels, preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", annot=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ==========================
# SAVE MODEL
# ==========================
print("\n💾 Saving model...\n")

model.save_pretrained("bert_chatbot_model")
tokenizer.save_pretrained("bert_chatbot_model")

print("✅ Model Training Completed Successfully!")