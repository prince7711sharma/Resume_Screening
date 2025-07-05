# 🤖 AI Resume Screening App using NLP & Streamlit

This project is a smart resume screening tool that uses **Natural Language Processing (NLP)** and **Machine Learning (KNN)** to automatically classify resumes into job categories, extract top skills, and generate short downloadable reports. It features an interactive **Streamlit UI**, allowing users to upload resumes in both **PDF** and **text** formats.

---

## 🚀 Features

- 📄 Upload resumes in PDF or paste text directly
- 🔍 Predicts job category using TF-IDF + KNN
- 👤 Extracts candidate name from resume
- 💡 Shows top skills using keyword extraction
- 📥 Generates a short downloadable report
- ✨ Clean and user-friendly Streamlit interface
- ✅ Supports `.pdf` and `.txt` file formats

---

## 📂 File Structure
resume-screening-app/
│
├── app.py # Main Streamlit application
├── knn.pkl # Trained KNN model
├── tfidf.pkl # TF-IDF vectorizer
├── label_encoder.pkl # LabelEncoder for category mapping
├── requirements.txt # Python dependencies
├── sample_resumes/ # Sample resume files
└── README.md # You're reading this!

---

## 🧠 Model Details

- **Vectorizer:** `TF-IDF`
- **Classifier:** `K-Nearest Neighbors (KNN)`
- **Skill Extraction:** Custom logic using keyword matching
- **PDF Parsing:** `PyPDF2`

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/resume-screening-app.git
cd resume-screening-app

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate     # On Windows (Git Bash or PowerShell)
# OR
source venv/bin/activate         # On macOS/Linux

# Install required libraries
pip install -r requirements.txt

