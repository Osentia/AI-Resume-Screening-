# 🤖 AI Resume Screener with Custom ML

An AI-powered resume screening tool that matches resumes with job descriptions using machine learning techniques. Built using Python, TF-IDF, Cosine Similarity, and Logistic Regression — **no external APIs or tokenizers** required. Perfect for beginners and students in AI/ML and resume parsing projects.

---

## 📌 Features
- ✅ Extracts text from PDF & image resumes (PNG, JPG)
- ✅ Loads job descriptions from CSV files
- ✅ Calculates similarity score using **TF-IDF + Cosine Similarity**
- ✅ Detects skills and experience using **Logistic Regression**
- ✅ Generates final matching score (0–100%)
- ✅ Outputs **Top 5 job matches** with breakdown
- ✅ Saves full results in CSV format

---

## 📁 Folder Structure
```
📂 AI-Resume-Screener
 ├── resume_examples/            # Folder with PDF/PNG resumes
 ├── job_descriptions.csv        # CSV file with job listings
 ├── main.py                     # Main Python code
 ├── training_data.py            # Skill & Experience training set
 ├── top_matches.csv             # Output result
 └── README.md                   # You’re here!
```

---

## 📐 How It Works (Flowchart)

```
[Load Resume] → [Extract Text] → 
[Load Jobs CSV] → [Preprocess] →
[Cosine Similarity] + [Skills Matching] + [Experience Detection] →
[Final Score Calculation] → [Top 5 Output + CSV]
```

---

## 🧠 Methodology

### 🧮 Cosine Similarity
```python
vectors = TfidfVectorizer().fit_transform([resume_text, job_text]).toarray()
similarity = util.cos_sim(vectors[0], vectors[1]).item() * 100
```

### 🏷️ Final Score Calculation
```python
final_score = (
    (similarity_score * 0.4) +
    (len(matched_skills) * 5) +
    experience_weight
)
```

### 🎯 Experience Weights
| Level          | Weight |
|----------------|--------|
| No Experience  | 10     |
| Entry-Level    | 15     |
| Junior         | 20     |
| Mid-Level      | 30     |
| Senior         | 45     |

---

## 📊 Example Output
```
🏆 Rank 1: Machine Learning Engineer
   ✅ Final Score: 92.0%
   🔹 Matching Skills: Python, TensorFlow, ML
   🔹 Experience Level: Mid-Level
```

---

## 📥 Installation
```bash
git clone https://github.com/yourusername/ai-resume-screener
cd ai-resume-screener
pip install -r requirements.txt
```

---

## 🧪 Run the Project
```bash
python main.py
```

> You can modify resume/job file paths in `main.py`.

---

## 🔎 Use Cases
- Internship screening
- University admissions filtering
- Hackathon resume matchers
- Entry-level job recommendation bots

---

## 📌 Project Limitations
- Works best with structured resumes
- Basic ML training (small dataset)
- No deep semantic understanding or context

---

## 📈 Future Improvements
- Add visualization of matched text
- Train with larger labeled datasets
- Integrate Streamlit for web UI
- Include support for Thai or multilingual resumes

---

## 📎 License
This project is licensed under the MIT License.

---

## 📬 Contact
Feel free to reach out or open an issue for any bugs, feedback, or suggestions!
