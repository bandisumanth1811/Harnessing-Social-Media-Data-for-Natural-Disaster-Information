import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(page_title="Hurricane Tweet Classifier", layout="centered")

from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pickle

# -------- Load Models and Tokenizers -------- #
@st.cache_resource
def load_stage1_model():
    model = TFBertForSequenceClassification.from_pretrained("saved_models/stage1_info_model")
    tokenizer = BertTokenizer.from_pretrained("saved_models/stage1_info_model")
    return model, tokenizer

@st.cache_resource
def load_stage2_model():
    model = TFBertForSequenceClassification.from_pretrained("saved_models/stage2_category_model")
    tokenizer = BertTokenizer.from_pretrained("saved_models/stage2_category_model")
    return model, tokenizer

@st.cache_resource
def load_label_encoder():
    with open("saved_models/stage2_label_encoder.pkl", "rb") as f:
        return pickle.load(f)

model_info, tokenizer_info = load_stage1_model()
model_cat, tokenizer_cat = load_stage2_model()
label_encoder = load_label_encoder()

# -------- Streamlit UI -------- #
st.title("🌪️ Hurricane Tweet Classifier")
st.write("This tool first checks if a tweet is **Informational**. If it is, it classifies the **type of information**.")

tweet = st.text_area("✍️ Enter a Reddit tweet below:")

if st.button("🔍 Classify Tweet"):
    if not tweet.strip():
        st.warning("Please enter a tweet to classify.")
    else:
        try:
            # ---- Stage 1: Info vs Not ---- #
            inputs_info = tokenizer_info(tweet, return_tensors='tf', truncation=True, padding=True)
            outputs_info = model_info(inputs_info)
            pred_info = tf.argmax(outputs_info.logits, axis=1).numpy()[0]

            info_label = "Information" if pred_info == 1 else "Not Information"
            st.markdown(f"### 🧾 Informational Check: `{info_label}`")

            # ---- Stage 2: Info Category ---- #
            if pred_info == 1:
                inputs_cat = tokenizer_cat(tweet, return_tensors='tf', truncation=True, padding=True)
                outputs_cat = model_cat(inputs_cat)
                pred_cat = tf.argmax(outputs_cat.logits, axis=1).numpy()[0]
                category = label_encoder.inverse_transform([pred_cat])[0]

                st.markdown(f"### 🏷️ Information Category:\n**`{category}`**")

        except Exception as e:
            st.error(f"🚨 Error during classification:\n{e}")
