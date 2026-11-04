import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_models():
    sent = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    truncation=True, max_length=512)
    issue = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    return sent, issue

sent, issue = load_models()
ISSUES = ["management", "pay", "work-life balance", "career growth", "culture"]

st.title("Employee Review Analyzer")
review = st.text_area("Paste an employee review")

if st.button("Analyze") and review.strip():
    s = sent(review)[0]
    i = issue(review, candidate_labels=ISSUES)
    st.write(f"**Sentiment:** {s['label']} ({s['score']:.2f})")
    st.write(f"**Main issue:** {i['labels'][0]} ({i['scores'][0]:.2f})")