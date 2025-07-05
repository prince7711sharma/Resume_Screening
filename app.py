import streamlit as st
import pickle
import re
import PyPDF2
import io

# -----------------------------
# Load models
# -----------------------------
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('clf.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# -----------------------------
# Category name mapping
# -----------------------------
CATEGORY_NAMES = {
    "HR": "Human Resources",
    "Advocate": "Legal / Advocacy",
    "Arts": "Arts & Design",
    "Data Science": "Data Science",
    "Engineering": "Engineering",
    "Finance": "Finance",
    "Health": "Healthcare",
    "Sales": "Sales & Marketing",
    "Web Designing": "Web Design",
    "Mechanical Engineer": "Mechanical Engineering",
    "Civil Engineer": "Civil Engineering",
    "Consultant": "Business Consulting",
    "Designer": "Creative Design",
    "Digital Media": "Digital Marketing",
    "Information Technology": "IT & Tech",
    "Public Relations": "PR & Communications",
}

# -----------------------------
# Helper Functions
# -----------------------------
def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if 1 < len(line.split()) <= 4 and re.match(r'^[A-Za-z\s\-]+$', line):
            return line
    return "Unknown"

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_skills(text):
    skill_set = [
        "python", "java", "sql", "excel", "nlp", "communication", "teamwork",
        "machine learning", "deep learning", "tensorflow", "keras", "pandas",
        "numpy", "power bi", "tableau", "leadership", "data analysis", "git",
        "linux", "c++", "r", "project management", "problem solving"
    ]
    text = text.lower()
    found = [skill.title() for skill in skill_set if skill in text]
    return list(set(found))

def generate_report(name, category, skills):
    report = f"""Resume Screening Report

Candidate Name: {name}
Predicted Category: {category}
Top Skills: {', '.join(skills) if skills else 'None Detected'}
"""
    return report.encode("utf-8")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🚀 Smart Resume Screener", layout="wide")
st.markdown("<h1 style='text-align: center; color: #3366cc;'>📄 AI-Powered Resume Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload or paste a resume to detect the job category and candidate name.</p>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📂 Upload Resume")

uploaded_file = st.sidebar.file_uploader("Choose a .pdf or .txt file", type=["pdf", "txt"])

st.sidebar.markdown("📋 Or paste resume text below:")

# Maintain session state for pasted text
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

pasted_text = st.sidebar.text_area("Paste Resume Text Here", value=st.session_state.resume_text, height=200)

col1, col2 = st.sidebar.columns([1, 1])
if col1.button("🟢 Apply"):
    st.session_state.resume_text = pasted_text
if col2.button("🔴 Remove"):
    st.session_state.resume_text = ""

# Resume processing logic
resume_text = ""
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        resume_text = uploaded_file.read().decode("utf-8")
elif st.session_state.resume_text.strip():
    resume_text = st.session_state.resume_text

# If resume exists, predict
if resume_text.strip():
    st.markdown("### 🔍 Prediction Result")
    candidate_name = extract_name(resume_text)
    vector = tfidf.transform([resume_text])
    predicted_index = knn_model.predict(vector)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    category_name = CATEGORY_NAMES.get(predicted_label, predicted_label)

    skills = extract_skills(resume_text)

    # Show results
    st.success(f"🧠 **Predicted Category:** {category_name}")
    st.info(f"👤 **Candidate Name (estimated):** {candidate_name}")
    st.markdown(f"💼 **Top Skills Found:** {', '.join(skills) if skills else 'None detected'}")

    with st.expander("📄 Show Resume Text"):
        st.text_area("Resume Text", resume_text, height=300)

    report_bytes = generate_report(candidate_name, category_name, skills)
    st.download_button("📥 Download Resume Report", report_bytes, file_name="resume_report.txt")

else:
    st.warning("📎 Please upload a resume or click 'Apply' after pasting text.")

# Footer
st.markdown("---")
st.caption("✨ Created with Streamlit | KNN | NLP | PyPDF2")

