import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split
import os

print("📦 데이터 불러오는 중...")
df = pd.read_csv("./data/cleaned_reviews_strong.csv").dropna(subset=["cleaned", "label"])
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["cleaned"], df["label"], test_size=0.2, random_state=42
)
print(f"✔️ 총 데이터 수: {len(df)}, 학습: {len(train_texts)}, 검증: {len(val_texts)}")

print("🔑 토크나이저 로딩 중...")
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(model_name)

MAX_LEN = 128
BATCH_SIZE = 24

print("🔤 Tokenizing train 데이터...")
train_encodings = tokenizer(
    list(train_texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="tf"
)
print("✅ train tokenizing 완료")

print("🔤 Tokenizing validation 데이터...")
val_encodings = tokenizer(
    list(val_texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="tf"
)
print("✅ validation tokenizing 완료")

print("📁 TF Dataset 생성 중...")
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels.values
)).shuffle(10000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels.values
)).batch(BATCH_SIZE)
print("✅ TF Dataset 생성 완료")

print("🧠 모델 로딩 (PyTorch → TensorFlow 변환)...")
model = TFElectraForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    from_pt=True
)

print("⚙️ 옵티마이저 설정 중...")
num_train_steps = len(train_dataset) * 3  # 3 epochs
optimizer, schedule = create_optimizer(init_lr=5e-5, num_train_steps=num_train_steps, num_warmup_steps=0)

print("🔧 모델 컴파일 중...")
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# ✅ GPU 메모리 동적 할당 설정
print("🧠 GPU 메모리 설정 중...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 메모리 동적 할당 설정 완료")
    except RuntimeError as e:
        print(f"❌ 메모리 설정 오류: {e}")

print("🚀 학습 시작!")
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

print("💾 모델 저장 중...")
model.save_pretrained("./models/koelectra_sentiment_model_tf")
tokenizer.save_pretrained("./models/koelectra_sentiment_model_tf")
print("✅ 저장 완료")
