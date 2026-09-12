import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# IMPORTANT: change this to the repo you upload your model to on huggingface.co
# Format is "your-username/your-model-name", for example "sara123/fake-news-bert"
MODEL_REPO = "saramohamed05/fake-news-bert"

MAX_LENGTH = 256  # must match what was used during training
LABELS = {0: "Fake", 1: "Real"}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    return tokenizer, model


st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.caption("Fine-tuned bert-base-uncased classifier — 99.9% test accuracy")

tokenizer, model = load_model()

title = st.text_input("Headline / Title", placeholder="e.g. Senate passes new infrastructure bill")
text = st.text_area("Article text", height=250, placeholder="Paste the body of the article here...")

if st.button("Check", type="primary"):
    title_clean = (title or "").strip()
    text_clean = (text or "").strip()

    if not title_clean and not text_clean:
        st.warning("Please enter a headline or article text first.")
    else:
        # Same construction as make_content() in the notebook: title + " " + text
        if title_clean and text_clean:
            content = title_clean + " " + text_clean
        else:
            content = title_clean or text_clean

        inputs = tokenizer(
            content,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))

        st.subheader(f"Prediction: {LABELS[pred]}")
        col1, col2 = st.columns(2)
        col1.metric("Fake probability", f"{float(probs[0]) * 100:.1f}%")
        col2.metric("Real probability", f"{float(probs[1]) * 100:.1f}%")
