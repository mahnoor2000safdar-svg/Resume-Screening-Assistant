````markdown
# Resume/CV Screening Assistant

## Project Overview

Resume Intelligence is a web-based Resume/CV Screening Assistant developed to automate the initial screening and ranking of candidate resumes against a given Job Description.

The system analyzes resume content, identifies relevant skills, detects missing skills, and generates an explainable candidate ranking based on text similarity and skill matching.

---

## Objective

The main objective of this project is to make the initial resume screening process faster, more consistent, and easier to evaluate by automatically comparing candidate resumes with the requirements of a Job Description.

---

## Key Features

- Job Description input and editing
- PDF and TXT resume support
- Multiple resume screening
- TF-IDF based text similarity
- Skill matching
- Matched skill identification
- Missing skill detection
- Candidate ranking
- Overall Match Score
- TF-IDF Score
- Skill Match Score
- Candidate screening summary
- Explainable scoring results
- Professional Streamlit-based interface

---

## Screening Method

The system uses two main components to calculate the final candidate score:

### 1. TF-IDF Similarity – 55%

TF-IDF (Term Frequency–Inverse Document Frequency) is used to measure the textual similarity between the Job Description and each candidate's resume.

### 2. Skill Matching – 45%

The system identifies required skills from the Job Description and checks whether those skills are present in each candidate's resume.

### Final Score

```text
Final Score =
(TF-IDF Similarity × 0.55)
+
(Skill Match × 0.45)
````

A higher score indicates stronger relevance to the provided Job Description.

---

### System Workflow

```text
Job Description
       ↓
Resume Upload / Sample Resumes
       ↓
Text Extraction
       ↓
TF-IDF Similarity Analysis
       +
Skill Matching
       ↓
Final Match Score
       ↓
Candidate Ranking
       ↓
Matched & Missing Skills
```

---

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity
* Regular Expressions
* HTML/CSS
* PDF/TXT Text Processing

---

## Project Structure

```text
Resume_CV_Screening_Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── job_description.txt
│
├── data/
│
├── resumes/
│   ├── Candidate_01.pdf
│   ├── Candidate_02.pdf
│   ├── Candidate_03.pdf
│   └── ...
│
└── output/
```

---

## How to Run

### 1. Install Dependencies

Open the project folder in VS Code and run:

```bash
python -m pip install -r requirements.txt
```

### 2. Start the Application

```bash
python -m streamlit run app.py
```

### 3. Open the Application

The application will open in the browser at:

```text
http://localhost:8501
```

---

## How to Use

### Step 1 – Enter Job Description

Enter or edit the requirements of the target job.

### Step 2 – Upload Resumes

Upload candidate resumes in PDF or TXT format, or use the sample resumes available in the project folder.

### Step 3 – Screen Resumes

Click **Screen Resumes** to start the screening process.

### Step 4 – Review Results

The system generates a ranked list of candidates containing:

* Rank
* Candidate Name
* Match Score
* TF-IDF Score
* Skill Match
* Matched Skills
* Missing Skills

---

## Output

The system provides a ranked candidate table and screening summary.

The dashboard displays:

* Total Candidates
* Top Match
* Average Score
* TF-IDF / Skill weighting

This allows recruiters to quickly identify candidates whose resumes are more relevant to the provided Job Description.

---

## Privacy & Responsible Use

This system is intended to support the initial recruitment screening process.

Screening results should be treated as an assistive assessment and **not as the sole basis for hiring decisions**.

Users should avoid uploading private candidate information without proper authorization.

---

## Limitations

* Text similarity does not guarantee candidate suitability.
* Skill matching depends on the skills identified from the Job Description and resume text.
* Different wording for the same skill may not always be recognized.
* Human review is required before making final recruitment decisions.

---

## Future Enhancements

* Advanced semantic resume matching
* Improved skill extraction
* Experience and education matching
* Resume filtering and search
* CSV/Excel result export
* Candidate database integration
* Advanced analytics and visualizations

```


```

