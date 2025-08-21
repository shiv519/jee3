import os
import re
import base64
import requests
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from datetime import datetime, timedelta

# -----------------------------
# AI Detection Functions
# -----------------------------

def gemini_detect(text):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Extract exam questions with options (A, B, C, D) from the text below:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text

def groq_detect(text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": f"Extract exam questions with options (A, B, C, D) from:\n{text}"}]
    }
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return None

def hf_detect(text):
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        return None
    url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": f"Extract exam questions with options (A, B, C, D) from:\n{text}"}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        out = r.json()
        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"]
    return None

def regex_detect(text: str):
    # Matches "1. Question text ... A) option ... B) option ... C) option ... D) option"
    pattern = r"(\d+\..*?(?:A\).+?B\).+?C\).+?D\).+?))(?=\d+\.|$)"
    questions = re.findall(pattern, text, re.S)
    return "\n\n".join(q.strip() for q in questions) if questions else None


@st.cache_data(show_spinner=False)
def detect_questions_pagewise(pages, use_ai=False):
    results = []
    for i, page in enumerate(pages, 1):
        with st.spinner(f"Processing page {i}/{len(pages)}..."):
            if use_ai:
                try:
                    if os.getenv("GOOGLE_API_KEY"):
                        res = gemini_detect(page)
                    elif os.getenv("GROQ_API_KEY"):
                        res = groq_detect(page)
                    elif os.getenv("HF_API_KEY"):
                        res = hf_detect(page)
                    else:
                        res = regex_detect(page)
                except Exception:
                    res = regex_detect(page)
            else:
                res = regex_detect(page)
            if res:
                results.append(res)
    return "\n\n".join(results)

# -----------------------------
# Parse Questions
# -----------------------------

def parse_questions(raw_text):
    questions = []
    blocks = re.split(r"\n\s*\n", raw_text)
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines: 
            continue
        q = lines[0]
        opts = [l for l in lines[1:] if re.match(r"^[A-D]\)", l)]
        if opts:
            questions.append({"q": q, "options": opts})
    return questions

# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="JEE CBT App", layout="wide")

if "exam_started" not in st.session_state:
    st.session_state.exam_started = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "current" not in st.session_state:
    st.session_state.current = 0
if "end_time" not in st.session_state:
    st.session_state.end_time = None

st.title("📝 JEE CBT Practice Platform")

# -----------------------------
# Before Exam Start
# -----------------------------
if not st.session_state.exam_started:
    uploaded_file = st.file_uploader("Upload a Question Paper (PDF)", type="pdf")
    use_ai = st.checkbox("Use AI detection (slower)")

    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)

        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]

        raw_questions = detect_questions_pagewise(pages, use_ai)
        questions = parse_questions(raw_questions)

        if questions:
            if st.button("Start Exam"):
                st.session_state.exam_started = True
                st.session_state.questions = questions
                st.session_state.answers = {i: [] for i in range(len(questions))}
                st.session_state.current = 0
                st.session_state.end_time = datetime.now() + timedelta(hours=3)
                st.rerun()
        else:
            st.warning("No questions detected.")

# -----------------------------
# Exam Mode
# -----------------------------
else:
    questions = st.session_state.questions
    current = st.session_state.current
    answers = st.session_state.answers
    end_time = st.session_state.end_time

    # Timer
    remaining = end_time - datetime.now()
    if remaining.total_seconds() <= 0:
        st.warning("⏰ Time's up! Auto-submitting...")
        st.session_state.exam_started = False
        st.rerun()
    else:
        mins, secs = divmod(int(remaining.total_seconds()), 60)
        st.sidebar.markdown(f"⏳ Time Left: **{mins}m {secs}s**")

    # Palette
    st.sidebar.subheader("Question Palette")
    for i, ans in enumerate(answers.values()):
        label = f"Q{i+1}"
        if ans: label += " ✅"
        if st.sidebar.button(label, key=f"nav{i}"):
            st.session_state.current = i
            st.rerun()

    # Question Area
    q = questions[current]
    st.subheader(f"Q{current+1}: {q['q']}")
    selected = st.session_state.answers[current]

    new_selected = st.multiselect(
        "Select answer(s):", q["options"], default=selected, key=f"multi{current}"
    )
    st.session_state.answers[current] = new_selected

    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("⬅️ Prev") and current > 0:
            st.session_state.current -= 1
            st.rerun()
    with cols[1]:
        if st.button("➡️ Next") and current < len(questions)-1:
            st.session_state.current += 1
            st.rerun()
    with cols[2]:
        if st.button("Submit Exam", type="primary"):
            st.session_state.exam_started = False
            st.rerun()

# -----------------------------
# After Submission
# -----------------------------
if not st.session_state.exam_started and st.session_state.questions:
    st.success("✅ Exam submitted!")
    st.write("Your responses:")
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}: {q['q']}**")
        st.markdown("Your answer: " + (", ".join(st.session_state.answers[i]) if st.session_state.answers[i] else "Not Answered"))
