# AI-Resume-Screening-

zfrom google.colab import drive
drive.mount('/content/drive')

!pip install --upgrade torch torchvision torchaudio sentence-transformers datasets PyPDF2 opencv-python pytesseract
!sudo apt install tesseract-ocr -y > /dev/null 2>&1

import os
import PyPDF2
import pandas as pd
import cv2
import pytesseract
import numpy as np
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load AI Model for Text Similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Training Data for Matching Skills Model
training_data = [
    ("I have developed deep learning models using TensorFlow.", "Machine Learning"),
    ("Worked with Python for data analysis.", "Python"),
    ("Developed web applications using React and Node.js.", "Web Development"),
    ("Built cloud solutions with AWS and Kubernetes.", "Cloud Computing"),
    ("I am an expert in SQL database optimization.", "SQL"),
]

texts = [text for text, label in training_data]
labels = [label for text, label in training_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

classifier = LogisticRegression()
classifier.fit(X, labels)

print("✅ Matching Skills Model Trained!")

# ✅ Training Data for Experience Level Model
experience_data = [
    ("I have 10 years of experience in software engineering.", "Senior"),
    ("Worked as a machine learning engineer for 5 years.", "Mid-Level"),
    ("Internship in AI research for 6 months.", "Entry-Level"),
    ("3 years of experience in deep learning and NLP.", "Junior"),
]

exp_texts = [text for text, label in experience_data]
exp_labels = [label for text, label in experience_data]

exp_vectorizer = TfidfVectorizer()
X_exp = exp_vectorizer.fit_transform(exp_texts)

exp_classifier = LogisticRegression()
exp_classifier.fit(X_exp, exp_labels)

print("✅ Experience Level Model Trained!")


# Function: Extract text from Resume (PDF/PNG)
def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return "".join([page.extract_text() or "" for page in reader.pages[:3]])  # Read up to 3 pages
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray).strip()
    return ""

# Function: Load job descriptions from CSV
def load_job_descriptions(csv_path):
    job_df = pd.read_csv(csv_path)
    return job_df.to_dict(orient="records")  # Convert CSV rows into a dictionary list

# Function: Compute Cosine Similarity
def calculate_cosine_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_text]).toarray()

    cosine_sim = util.cos_sim(vectors[0], vectors[1]).item() * 100  # Scale 0-100%
    return cosine_sim

# Function: Predict Skills using ML Model
def predict_skills(resume_text):
    sentences = resume_text.split(".")
    predicted_skills = [classifier.predict(vectorizer.transform([s]))[0] for s in sentences]
    return list(set(predicted_skills)) if predicted_skills else ["No Matching Skills"]

# Function: Predict Experience Level using ML Model
def predict_experience(resume_text):
    sentences = resume_text.split(".")
    predictions = [exp_classifier.predict(exp_vectorizer.transform([s]))[0] for s in sentences]

    experience_levels = ["Entry-Level", "Junior", "Mid-Level", "Senior"]
    for level in experience_levels[::-1]:  # Prioritize higher levels
        if level in predictions:
            return level
    return "No Experience Found"

# Function: Calculate Final Score (0-100%)
def calculate_final_score(similarity_score, matched_skills, experience_level):
    exp_weight = {"No Experience Found": 10, "Entry-Level": 15, "Junior": 20, "Mid-Level": 30, "Senior": 45}[experience_level]

    return (
        (similarity_score * 0.4) +   # ✅ Cosine Similarity
        (len(matched_skills) * 5) +  # ✅ Skill Match (5 pts per skill)
        exp_weight                  # ✅ Experience Level Weight
    )

# Function: Match Resume to Job Descriptions & Show Results
def match_resume_to_jobs(resume_file, resume_folder, job_csv, output_file):
    job_descriptions = load_job_descriptions(job_csv)

    resume_text = extract_text(os.path.join(resume_folder, resume_file))
    results = []

    for job in job_descriptions:
        job_text = job["description"]

        similarity_score = calculate_cosine_similarity(resume_text, job_text)
        matched_skills = predict_skills(resume_text)
        experience_level = predict_experience(resume_text)

        final_score = calculate_final_score(
            similarity_score, matched_skills, experience_level
        )

        results.append({
            "Job": job["title"],
            "Cosine Similarity (%)": round(similarity_score, 2),
            "Matching Skills": matched_skills,
            "Experience Level": experience_level,
            "Final Score (%)": round(final_score, 2)
        })

    df = pd.DataFrame(results).sort_values(by="Final Score (%)", ascending=False)
    df.to_csv(output_file, index=False)

    print("\n🔹 **Top 5 Matching Jobs:**")
    for i, row in df.head(5).iterrows():
        print(f"\n🏆 Rank {i+1}: {row['Job']}")
        print(f"   ✅ Final Score: {row['Final Score (%)']}%")
        print(f"   🔹 Matching Skills: {', '.join(row['Matching Skills'])}")
        print(f"   🔹 Experience Level: {row['Experience Level']}")

# RUN MATCHING
match_resume_to_jobs(
    "luxsleek-cv.pdf",
    "/content/drive/MyDrive/2nd Year/2nd Semester/AI/AI Resume",
    "/content/drive/MyDrive/2nd Year/2nd Semester/AI/AI Job description/300fake_job_postings.csv",
    "top_matches.csv"
)
