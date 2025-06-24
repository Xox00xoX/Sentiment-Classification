import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split  # ✅ 추가
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ElectraTokenizer, TFElectraForSequenceClassification

# 📦 경로
MODEL_DIR = "./models/koelectra_sentiment_model_tf"
CSV_PATH = "./data/cleaned_reviews_strong.csv"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# 📊 데이터 로딩
print("📦 데이터 로딩 중...")
df = pd.read_csv(CSV_PATH).dropna(subset=["cleaned", "label"])
train_texts, val_texts, train_labels, val_labels = train_test_split(df["cleaned"], df["label"], test_size=0.2)

# 🔑 모델 및 토크나이저 로딩
print("🔑 모델 및 토크나이저 로딩...")
model = TFElectraForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = ElectraTokenizer.from_pretrained(MODEL_DIR)

# 🔤 토크나이징
print("🔤 토크나이징...")
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, return_tensors="tf")

# 🧠 예측 수행
print("🧠 예측 수행 중...")
val_dataset = tf.data.Dataset.from_tensor_slices(dict(val_encodings)).batch(32)
y_true = val_labels.values
y_pred_logits = model.predict(val_dataset, verbose=1).logits
y_pred = np.argmax(y_pred_logits, axis=1)

# 📊 혼동 행렬 및 리포트
print("📊 혼동 행렬 및 리포트 생성...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(f"{PLOT_DIR}/confusion_matrix.png")
plt.close()

# 📈 Precision-Recall Curve
print("📈 PR 곡선 생성 중...")
y_probs = tf.nn.softmax(y_pred_logits, axis=1).numpy()
plt.figure(figsize=(8, 6))
for class_id in range(3):
    y_true_binary = (y_true == class_id).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, class_id])
    ap = average_precision_score(y_true_binary, y_probs[:, class_id])
    plt.plot(recall, precision, label=f"Class {class_id} (AP={ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(f"{PLOT_DIR}/precision_recall_curve.png")
plt.close()

# 🧾 classification report 출력
print(classification_report(y_true, y_pred, digits=4))
# 모델 저장
model.save_pretrained("./models/koelectra_sentiment_model_tf_updated")

# 토크나이저 저장 (안 바꿨더라도 함께 저장하는 게 좋아)
tokenizer.save_pretrained("./models/koelectra_sentiment_model_tf_updated")