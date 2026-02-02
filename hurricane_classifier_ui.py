import streamlit as st
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

# MUST be the first Streamlit command
st.set_page_config(page_title="Hurricane Tweet Classifier", layout="centered")

# -------- Load Models and Tokenizers -------- #

@st.cache_resource
def load_stage1_model():
    model = BertForSequenceClassification.from_pretrained(
        "saved_models/stage1_info_model"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "saved_models/stage1_info_model"
    )
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_stage2_model():
    model = BertForSequenceClassification.from_pretrained(
        "saved_models/stage2_category_model"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "saved_models/stage2_category_model"
    )
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_label_encoder():
    with open("saved_models/stage2_label_encoder.pkl", "rb") as f:
        return pickle.load(f)


# -------- Load Everything -------- #
model_info, tokenizer_info = load_stage1_model()
model_cat, tokenizer_cat = load_stage2_model()
label_encoder = load_label_encoder()

# -------- Helper Function -------- #
def predict(text, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction


# -------- Streamlit UI -------- #
st.title("🌪️ Hurricane Tweet Classifier")
st.write(
    "This tool first checks if a tweet is **Informational**. "
    "If it is, it classifies the **type of information**."
)

tweet = st.text_area("✍️ Enter a Reddit tweet below:")

if st.button("🔍 Classify Tweet"):
    if not tweet.strip():
        st.warning("Please enter a tweet to classify.")
    else:
        try:
            # ---- Stage 1: Info vs Not ---- #
            pred_info = predict(tweet, model_info, tokenizer_info)
            info_label = "Information" if pred_info == 1 else "Not Information"

            st.markdown(f"### 🧾 Informational Check: `{info_label}`")

            # ---- Stage 2: Info Category ---- #
            if pred_info == 1:
                pred_cat = predict(tweet, model_cat, tokenizer_cat)
                category = label_encoder.inverse_transform([pred_cat])[0]

                st.markdown(
                    f"### 🏷️ Information Category:\n**`{category}`**"
                )

        except Exception as e:
            st.error(f"🚨 Error during classification:\n{e}")
