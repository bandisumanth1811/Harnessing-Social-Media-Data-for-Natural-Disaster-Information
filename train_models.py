import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pickle

# Create model save directory
os.makedirs("saved_models", exist_ok=True)

print("🔄 Training Stage 1: Info/Not Info Classifier...")
df_stage1 = pd.read_csv("Title-Category.csv", encoding='latin1')  # 👈 FIXED
df_stage1.dropna(subset=['Title', 'Category'], inplace=True)

# Map 'Information' => 1, 'Not Information' => 0
df_stage1['InfoLabel'] = df_stage1['Category'].apply(lambda x: 1 if str(x).strip().lower() == 'information' else 0)

X1 = df_stage1['Title'].astype(str).tolist()
y1 = df_stage1['InfoLabel'].tolist()

X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, test_size=0.2, random_state=42)

tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
train_enc1 = tokenizer1(X1_train, truncation=True, padding=True, return_tensors='tf')
val_enc1 = tokenizer1(X1_val, truncation=True, padding=True, return_tensors='tf')

train_ds1 = tf.data.Dataset.from_tensor_slices((dict(train_enc1), y1_train)).batch(16)
val_ds1 = tf.data.Dataset.from_tensor_slices((dict(val_enc1), y1_val)).batch(16)

model1 = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model1.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

model1.fit(train_ds1, validation_data=val_ds1, epochs=3)
model1.save_pretrained("saved_models/stage1_info_model")
tokenizer1.save_pretrained("saved_models/stage1_info_model")

print("✅ Stage 1 model saved.")

# ================== STAGE 2: Multi-class Info Category ================== #
print("🔄 Training Stage 2: Information Category Classifier...")
df_stage2 = pd.read_excel("Hurricane_Reddit_Categorized.xlsx")
df_stage2.dropna(subset=['Title', 'Category'], inplace=True)

X2 = df_stage2['Title'].astype(str).tolist()
y2 = df_stage2['Category'].tolist()

label_encoder = LabelEncoder()
y2_encoded = label_encoder.fit_transform(y2)

X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2_encoded, test_size=0.2, random_state=42)

tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')
train_enc2 = tokenizer2(X2_train, truncation=True, padding=True, return_tensors='tf')
val_enc2 = tokenizer2(X2_val, truncation=True, padding=True, return_tensors='tf')

train_ds2 = tf.data.Dataset.from_tensor_slices((dict(train_enc2), y2_train)).batch(16)
val_ds2 = tf.data.Dataset.from_tensor_slices((dict(val_enc2), y2_val)).batch(16)

model2 = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model2.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

model2.fit(train_ds2, validation_data=val_ds2, epochs=3)
model2.save_pretrained("saved_models/stage2_category_model")
tokenizer2.save_pretrained("saved_models/stage2_category_model")

with open("saved_models/stage2_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Stage 2 model and label encoder saved.")
