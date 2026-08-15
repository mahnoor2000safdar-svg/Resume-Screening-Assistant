import os
import re
from pathlib import Path

import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Resume Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 90% 5%, rgba(37, 99, 235, 0.18), transparent 25%),
        radial-gradient(circle at 10% 20%, rgba(6, 182, 212, 0.10), transparent 25%),
        #06111f;
    color: #e5eefc;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09172a 0%, #07111f 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

.sidebar-title {
    font-size: 25px;
    font-weight: 800;
    color: white;
}

.sidebar-subtitle {
    color: #91a9c7;
    font-size: 13px;
    line-height: 1.7;
}

.sidebar-section {
    color: white;
    font-size: 16px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 12px;
}

.sidebar-card {
    background: linear-gradient(145deg, #102d50, #081c34);
    border: 1px solid rgba(59, 130, 246, 0.38);
    border-radius: 16px;
    padding: 20px;
}

.sidebar-card-title {
    color: #60a5fa;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 15px;
}

.sidebar-card p {
    color: #d9e6f7;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 8px;
}

.sidebar-divider {
    height: 1px;
    background: rgba(148, 163, 184, 0.10);
    margin: 25px 0;
}


/* HERO */

.hero {
    background:
        radial-gradient(circle at 95% 5%, rgba(34, 211, 238, 0.18), transparent 25%),
        linear-gradient(145deg, #10284a, #0a1a31);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 24px;
    padding: 42px;
    margin-bottom: 35px;
    box-shadow: 0 25px 70px rgba(0, 0, 0, 0.25);
}

.hero-badge {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.16);
    border: 1px solid rgba(96, 165, 250, 0.35);
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.05;
    color: white;
    margin-bottom: 15px;
}

.hero-title span {
    color: #38bdf8;
}

.hero-subtitle {
    color: #c7d6e9;
    font-size: 16px;
    line-height: 1.8;
    max-width: 850px;
}


/* SECTIONS */

.section {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 28px;
    margin-bottom: 15px;
}

.section-number {
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    font-weight: 800;
}

.section-title {
    color: white;
    font-size: 25px;
    font-weight: 800;
}


/* METRIC CARDS */

.metric-card {
    background: linear-gradient(145deg, #0d2340, #09182b);
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    color: #60a5fa;
    font-size: 30px;
    font-weight: 800;
}

.metric-label {
    color: #9fb2cc;
    font-size: 12px;
    margin-top: 5px;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 13px 20px;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
}

.stButton > button:hover {
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.30);
}


/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #0b1b30;
    border: 1px dashed rgba(96, 165, 250, 0.35);
    border-radius: 15px;
}


/* TEXT AREA */

textarea {
    background: #0b1b30 !important;
    color: #e6f0ff !important;
    border: 1px solid rgba(96, 165, 250, 0.22) !important;
    border-radius: 14px !important;
}


/* FOOTER */

.app-footer {
    margin-top: 50px;
    padding: 25px 0;
    border-top: 1px solid rgba(148, 163, 184, 0.10);
    color: #7186a3;
    text-align: center;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RESUME_DIR = BASE_DIR / "resumes"

JOB_FILE = BASE_DIR / "data" / "job_description.txt"

RESUME_DIR.mkdir(exist_ok=True)


# =========================================================
# PDF READER
# =========================================================

def extract_pdf_text(file_path):

    try:

        from pypdf import PdfReader

        reader = PdfReader(str(file_path))

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception:
        return ""


# =========================================================
# TXT READER
# =========================================================

def read_txt_file(file_path):

    try:

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


# =========================================================
# RESUME READER
# =========================================================

def read_resume(file_path):

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    if extension == ".txt":
        return read_txt_file(file_path)

    return ""


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9+#.\s-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SKILLS
# =========================================================

SKILLS = [
    "python",
    "html",
    "css",
    "javascript",
    "sql",
    "git",
    "github",
    "react",
    "apis",
    "api",
    "database",
    "databases",
    "problem solving",
    "communication",
    "software development",
    "teamwork",
    "node.js",
    "java",
    "c++",
    "machine learning",
    "pandas",
    "numpy",
    "scikit-learn",
    "flask",
    "streamlit"
]


# =========================================================
# EXTRACT SKILLS
# =========================================================

def extract_skills(text):

    text = clean_text(text)

    found = []

    for skill in SKILLS:

        skill_clean = clean_text(skill)

        if skill_clean in text:

            found.append(skill)

    return sorted(set(found))


# =========================================================
# SKILL MATCHING
# =========================================================

def calculate_skill_match(job_text, resume_text):

    job_skills = extract_skills(job_text)

    resume_skills = extract_skills(resume_text)

    if not job_skills:

        return 0, [], []

    matched = [
        skill
        for skill in job_skills
        if skill in resume_skills
    ]

    missing = [
        skill
        for skill in job_skills
        if skill not in resume_skills
    ]

    score = (
        len(matched) / len(job_skills)
    ) * 100

    return score, matched, missing


# =========================================================
# TF-IDF
# =========================================================

def calculate_tfidf(job_text, resume_text):

    job_text = clean_text(job_text)

    resume_text = clean_text(resume_text)

    if not job_text or not resume_text:

        return 0

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            [
                job_text,
                resume_text
            ]
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return float(similarity * 100)

    except Exception:

        return 0


# =========================================================
# FINAL SCORE
# =========================================================

def calculate_final_score(
    tfidf_score,
    skill_score
):

    return (
        tfidf_score * 0.55
    ) + (
        skill_score * 0.45
    )


# =========================================================
# DEFAULT JOB DESCRIPTION
# =========================================================

def load_job_description():

    if JOB_FILE.exists():

        text = read_txt_file(JOB_FILE)

        if text.strip():

            return text

    return """
Software Engineering Intern

We are looking for a motivated Software Engineering Intern.

Requirements:

- Basic knowledge of Python
- HTML and CSS
- JavaScript
- SQL
- Git and GitHub
- Problem-solving skills
- Basic understanding of software development
- Good communication skills

Preferred:

- Experience with React
- Knowledge of APIs
- Familiarity with databases
- Teamwork
- Willingness to learn
"""


# =========================================================
# LOAD RESUMES FROM FOLDER
# =========================================================

def load_sample_resumes():

    resumes = []

    for file in sorted(
        RESUME_DIR.iterdir()
    ):

        if file.suffix.lower() not in [
            ".pdf",
            ".txt"
        ]:

            continue

        text = read_resume(file)

        if text.strip():

            resumes.append(
                {
                    "name": file.stem,
                    "filename": file.name,
                    "text": text,
                    "path": str(file)
                }
            )

    return resumes


# =========================================================
# SCREEN RESUMES
# =========================================================

def screen_resumes(
    job_description,
    resumes
):

    results = []

    for resume in resumes:

        tfidf_score = calculate_tfidf(
            job_description,
            resume["text"]
        )

        skill_score, matched, missing = (
            calculate_skill_match(
                job_description,
                resume["text"]
            )
        )

        final_score = calculate_final_score(
            tfidf_score,
            skill_score
        )

        results.append(
            {
                "Candidate": resume["name"],
                "Match Score": round(
                    final_score,
                    2
                ),
                "TF-IDF Score": round(
                    tfidf_score,
                    2
                ),
                "Skill Match": round(
                    skill_score,
                    2
                ),
                "Matched Skills": ", ".join(
                    matched
                ),
                "Missing Skills": ", ".join(
                    missing
                )
            }
        )

    df = pd.DataFrame(results)

    if not df.empty:

        df = df.sort_values(
            "Match Score",
            ascending=False
        ).reset_index(
            drop=True
        )

        df.insert(
            0,
            "Rank",
            range(
                1,
                len(df) + 1
            )
        )

    return df


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown('<div class="sidebar-title">📄 Resume Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-subtitle">AI-assisted resume screening and candidate ranking.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Screening Method</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="sidebar-card">
    <div class="sidebar-card-title">SCORING MODEL</div>
    <p><b>55%</b> TF-IDF similarity</p>
    <p><b>45%</b> Skill matching</p>
    <p>Final score combines both signals to produce an explainable candidate ranking.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🔒 Privacy</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-subtitle">Use anonymized or sample resumes. Do not upload private candidate information without authorization.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📊 Features</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-subtitle">✓ TF-IDF text similarity<br />✓ Skill matching<br />✓ Candidate ranking<br />✓ Missing skill detection<br />✓ Explainable scoring</div>', unsafe_allow_html=True)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge"></div>
    <div class="hero-title">Resume <span>Intelligence</span></div>
    <div class="hero-subtitle">Screen, compare and rank candidates against your job description using explainable text similarity and skill matching.<br /><br />Built for faster, smarter and more consistent initial candidate screening.</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.markdown("""
<div class="section">
    <div class="section-number">01</div>
    <div class="section-title">Job Description</div>
</div>
""", unsafe_allow_html=True)

default_job = load_job_description()

job_description = st.text_area(
    "Enter or edit the job description",
    value=default_job,
    height=260
)


# =========================================================
# RESUME UPLOAD
# =========================================================

st.markdown("""
<div class="section">
    <div class="section-number">02</div>
    <div class="section-title">Upload Resumes</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload PDF or TXT resumes",
    type=[
        "pdf",
        "txt"
    ],
    accept_multiple_files=True
)


# =========================================================
# SAMPLE RESUMES
# =========================================================

sample_resumes = load_sample_resumes()

use_samples = st.checkbox(
    f"Use resumes from project folder ({len(sample_resumes)} found)",
    value=True
)


# =========================================================
# PREPARE RESUMES
# =========================================================

all_resumes = []

if use_samples:

    all_resumes.extend(
        sample_resumes
    )


if uploaded_files:

    for uploaded in uploaded_files:

        try:

            if uploaded.name.lower().endswith(".txt"):

                text = uploaded.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            else:

                import tempfile

                suffix = Path(
                    uploaded.name
                ).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp:

                    temp.write(
                        uploaded.getbuffer()
                    )

                    temp_path = temp.name

                text = extract_pdf_text(
                    Path(temp_path)
                )

                try:

                    os.remove(
                        temp_path
                    )

                except:

                    pass

            if text.strip():

                all_resumes.append(
                    {
                        "name": Path(
                            uploaded.name
                        ).stem,
                        "filename": uploaded.name,
                        "text": text,
                        "path": ""
                    }
                )

        except Exception as error:

            st.error(
                f"Could not read {uploaded.name}: {error}"
            )


# =========================================================
# SCREEN BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

screen_button = st.button(
    "🔎  SCREEN RESUMES",
    use_container_width=True
)


# =========================================================
# SCREENING
# =========================================================

if screen_button:

    if not job_description.strip():

        st.error(
            "Please enter a job description."
        )

    elif not all_resumes:

        st.warning(
            "Please upload resumes or select resumes from the project folder."
        )

    else:

        with st.spinner(
            "Analyzing resumes..."
        ):

            results_df = screen_resumes(
                job_description,
                all_resumes
            )

        st.session_state["results"] = results_df

        st.session_state["resumes"] = all_resumes

        st.success(
            f"Screening completed for {len(results_df)} candidates."
        )


# =========================================================
# RESULTS
# =========================================================

if "results" in st.session_state:

    results_df = st.session_state["results"]

    st.markdown("""
<div class="section">
    <div class="section-number">03</div>
    <div class="section-title">Ranked Candidates</div>
</div>
""", unsafe_allow_html=True)


    # METRICS

    col1, col2, col3, col4 = st.columns(4)

    best_score = (
        results_df["Match Score"].max()
        if not results_df.empty
        else 0
    )

    average_score = (
        results_df["Match Score"].mean()
        if not results_df.empty
        else 0
    )


    with col1:

        st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{len(results_df)}</div>
    <div class="metric-label">CANDIDATES</div>
</div>
""", unsafe_allow_html=True)


    with col2:

        st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{best_score:.1f}%</div>
    <div class="metric-label">TOP MATCH</div>
</div>
""", unsafe_allow_html=True)


    with col3:

        st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{average_score:.1f}%</div>
    <div class="metric-label">AVERAGE SCORE</div>
</div>
""", unsafe_allow_html=True)


    with col4:

        st.markdown("""
<div class="metric-card">
    <div class="metric-value">55/45</div>
    <div class="metric-label">TF-IDF / SKILLS</div>
</div>
""", unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # RESULTS TABLE

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        height=430
    )


# =========================================================
# CANDIDATE DETAILS
# =========================================================

if "results" in st.session_state:

    results_df = st.session_state["results"]

    resumes_data = st.session_state["resumes"]

    st.markdown("""
<div class="section">
    <div class="section-number">04</div>
    <div class="section-title">Candidate Details</div>
</div>
""", unsafe_allow_html=True)

    resume_map = {
        resume["name"]: resume
        for resume in resumes_data
    }

    for _, row in results_df.iterrows():

        candidate_name = row["Candidate"]

        score = row["Match Score"]

        with st.expander(
            f"#{int(row['Rank'])}  {candidate_name}  •  {score:.2f}% Match"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Final Match",
                    f"{score:.2f}%"
                )

            with c2:

                st.metric(
                    "TF-IDF",
                    f"{row['TF-IDF Score']:.2f}%"
                )

            with c3:

                st.metric(
                    "Skill Match",
                    f"{row['Skill Match']:.2f}%"
                )


            st.markdown(
                "### ✅ Matched Skills"
            )

            if row["Matched Skills"]:

                st.write(
                    row["Matched Skills"]
                )

            else:

                st.write(
                    "No matching skills detected."
                )


            st.markdown(
                "### ⚠️ Missing Skills"
            )

            if row["Missing Skills"]:

                st.write(
                    row["Missing Skills"]
                )

            else:

                st.success(
                    "No required skills are missing."
                )


            candidate = resume_map.get(
                candidate_name
            )

            if candidate:

                with st.expander(
                    "View Resume Text"
                ):

                    st.text(
                        candidate["text"][:10000]
                    )


# =========================================================
# HOW SCORING WORKS
# =========================================================

with st.expander(
    "ℹ️ How scoring works"
):

    st.markdown(
        """
### TF-IDF Similarity — 55%

Measures how similar the resume text is
to the provided job description.

### Skill Matching — 45%

Checks how many required skills from the
job description are present in the resume.

### Final Score

**Final Score = (TF-IDF × 55%) + (Skill Match × 45%)**

A higher score indicates stronger relevance
to the provided job description.

> This system is intended to support recruitment
> screening and should not be the sole basis for
> hiring decisions.
"""
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="app-footer">
    <b>Resume Intelligence</b> • AI-assisted candidate screening<br /><br />
    Recruitment assistance only — not a sole basis for hiring decisions.
</div>
""", unsafe_allow_html=True)