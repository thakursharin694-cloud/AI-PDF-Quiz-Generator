import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import json
import re
import time
import random
from fpdf import FPDF

# Load environment variables
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

model = ChatMistralAI(
    model="mistral-small",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.7  # Slightly higher for question variety
)

TEMPLATE = """
{
  "questions": [
    {
      "id": 1,
      "question": "What is the purpose of assembler directives?",
      "options": [
        "A. To define segments and allocate space for variables",
        "B. To represent specific machine instructions",
        "C. To simplify the programmer's task",
        "D. To provide information to the assembler"
      ],
      "correct_answer": "D"
    }
  ]
}
"""

DIFFICULTY_PROMPTS = {
    "Easy": "Create straightforward questions testing basic recall and understanding. Keep language simple.",
    "Medium": "Create questions requiring understanding and application of concepts. Include some analytical questions.",
    "Hard": "Create challenging questions requiring deep analysis, synthesis, and critical thinking. Include tricky distractors."
}

DIFFICULTY_COLORS = {
    "Easy": "#22c55e",
    "Medium": "#f59e0b",
    "Hard": "#ef4444"
}

TIME_LIMITS = {
    "No Limit": None,
    "15 sec / question": 15,
    "30 sec / question": 30,
    "60 sec / question": 60,
    "2 min / question": 120,
}


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


def get_questions(text, num_questions=5, difficulty="Medium", seed=None):
    difficulty_instruction = DIFFICULTY_PROMPTS[difficulty]
    seed_hint = f"Use variation seed {seed} to generate a fresh, unique set of questions." if seed else ""

    prompt = f"""
    Act as a teacher and create {num_questions} multiple-choice questions (MCQs)
    based on the text delimited by four backquotes.
    Difficulty level: {difficulty}. {difficulty_instruction}
    {seed_hint}
    Return ONLY valid JSON. No explanation, no markdown, no extra text.
    Each question must contain:
    - id (integer)
    - question (string)
    - options (list of exactly 4 strings, each prefixed with A., B., C., or D.)
    - correct_answer (single letter only: A, B, C, or D)

    Example format:
    {TEMPLATE}

    Text:
    ````{text[:4000]}````
    """
    response = model.invoke(prompt)
    raw_text = response.content.strip()
    cleaned = re.sub(r"^```json\s*", "", raw_text)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed_json = json.loads(cleaned)
        questions = parsed_json["questions"]
        # Shuffle option order for randomness
        random.shuffle(questions)
        return questions
    except Exception as e:
        st.error("Failed to parse AI response.")
        st.code(cleaned)
        return []


def init_session_state():
    defaults = {
        "questions": [],
        "current_q": 0,
        "answers": {},
        "score": 0,
        "quiz_started": False,
        "quiz_finished": False,
        "quiz_submitted": False,
        "start_time": None,
        "question_start_time": None,
        "time_per_question": None,
        "total_time_taken": 0,
        "pdf_text": "",
        "difficulty": "Medium",
        "generation_seed": 0,
        "review_mode": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_quiz():
    keys_to_reset = [
        "questions", "current_q", "answers", "score",
        "quiz_started", "quiz_finished", "quiz_submitted",
        "start_time", "question_start_time", "total_time_taken", "review_mode"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


def get_letter_from_option(option_str):
    """Extract just the letter from 'A. Some text' -> 'A'"""
    if option_str and len(option_str) >= 1:
        return option_str[0].upper()
    return ""


def get_correct_option_text(question):
    """Get the full option text matching the correct answer letter"""
    correct_letter = question.get("correct_answer", "").strip().upper()
    for opt in question["options"]:
        if opt.strip().upper().startswith(correct_letter + "."):
            return opt
    return correct_letter


def calculate_score(questions, answers):
    score = 0
    for q in questions:
        qid = str(q["id"])
        user_answer = answers.get(qid, "")
        correct = q.get("correct_answer", "").strip().upper()
        if user_answer.strip().upper() == correct:
            score += 1
    return score


def format_time(seconds):
    if seconds is None:
        return "—"
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def generate_pdf(questions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Quiz Questions", ln=True, align='C')

    pdf.ln(10)

    for i, q in enumerate(questions, 1):
        pdf.set_font("Arial", style='B', size=12)
        pdf.multi_cell(0, 10, f"Q{i}. {q['question']}")

        pdf.set_font("Arial", size=11)

        for opt in q["options"]:
            pdf.multi_cell(0, 8, opt)

        pdf.multi_cell(
            0,
            8,
            f"Correct Answer: {q['correct_answer']}"
        )

        pdf.ln(5)

    pdf.output("quiz_questions.pdf")


def render_progress_bar(current, total):
    pct = int((current / total) * 100) if total > 0 else 0
    st.markdown(f"""
    <div style="margin: 8px 0 20px 0;">
        <div style="display:flex; justify-content:space-between; font-size:13px; color:rgba(128,128,160,0.85); margin-bottom:6px;">
            <span>Question {current} of {total}</span>
            <span>{pct}% complete</span>
        </div>
        <div style="background:rgba(128,128,160,0.18); border-radius:99px; height:6px; overflow:hidden;">
            <div style="width:{pct}%; background:linear-gradient(90deg, #6c63ff, #a78bfa); height:6px; border-radius:99px; transition:width 0.3s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_timer_badge(time_limit, question_start):
    if time_limit is None or question_start is None:
        return False
    elapsed = time.time() - question_start
    remaining = time_limit - elapsed
    if remaining <= 0:
        return True  # Time's up
    color = "#ef4444" if remaining < 5 else "#f59e0b" if remaining < 10 else "#6c63ff"
    st.markdown(f"""
    <div style="display:inline-flex; align-items:center; gap:6px; background:{color}18;
         border:1px solid {color}40; border-radius:8px; padding:6px 14px; margin-bottom:16px;">
        <span style="font-size:18px;">⏱</span>
        <span style="font-size:15px; font-weight:600; color:{color};">{format_time(remaining)} left</span>
    </div>
    """, unsafe_allow_html=True)
    return False


def main():
    st.set_page_config(
        page_title="QuizMind — AI Quiz Generator",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # ─── CSS ────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 720px !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Option buttons: light & dark safe ── */
    div.stButton > button {
        width: 100%;
        text-align: left !important;
        background: transparent;
        border: 1.5px solid rgba(128,128,160,0.25);
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 15px;
        color: inherit;
        font-family: 'DM Sans', sans-serif;
        font-weight: 400;
        transition: all 0.15s ease;
        margin-bottom: 2px;
    }

    div.stButton > button:hover {
        background: rgba(108,99,255,0.08);
        border-color: #6c63ff;
        color: #6c63ff;
        transform: translateX(4px);
    }

    div.stButton > button:active {
        background: rgba(108,99,255,0.14);
        transform: scale(0.99);
    }

    /* Primary action buttons */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: #6c63ff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #5a52e0 !important;
        color: #ffffff !important;
        transform: none !important;
    }

    /* Form styling */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stNumberInput"] > div > div {
        border-radius: 10px !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(128,128,160,0.2) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(128,128,160,0.2) !important;
        margin: 1.2rem 0 !important;
    }

    /* ── Typography: inherit so light/dark both work ── */
    .quiz-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: inherit;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .quiz-sub {
        color: rgba(128,128,160,0.85);
        font-size: 15px;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }

    .question-text {
        font-size: 18px;
        font-weight: 500;
        color: inherit;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* ── Score card: always dark bg so white text is safe ── */
    .score-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        text-align: center;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }

    .score-big {
        font-family: 'DM Serif Display', serif;
        font-size: 4rem;
        line-height: 1;
        font-weight: 400;
        color: #ffffff;
    }

    .score-label {
        font-size: 15px;
        color: rgba(255,255,255,0.88);
        margin-top: 4px;
    }

    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 99px;
        padding: 6px 14px;
        font-size: 13px;
        color: #ffffff;
        margin: 4px;
    }

    .diff-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ── Answer feedback: strong explicit colors, no opacity tricks ── */
    .answer-correct {
        background: #14532d;
        border: 1.5px solid #16a34a;
        border-radius: 12px;
        padding: 12px 16px;
        color: #bbf7d0;
        font-size: 14px;
        margin: 6px 0;
    }

    .answer-wrong {
        background: #450a0a;
        border: 1.5px solid #dc2626;
        border-radius: 12px;
        padding: 12px 16px;
        color: #fecaca;
        font-size: 14px;
        margin: 6px 0;
    }

    .answer-skipped {
        background: rgba(128,128,160,0.12);
        border: 1.5px solid rgba(128,128,160,0.3);
        border-radius: 12px;
        padding: 12px 16px;
        color: inherit;
        font-size: 14px;
        margin: 6px 0;
        opacity: 0.7;
    }

    .answer-neutral {
        background: rgba(128,128,160,0.06);
        border: 1.5px solid rgba(128,128,160,0.2);
        border-radius: 12px;
        padding: 12px 16px;
        color: inherit;
        font-size: 14px;
        margin: 6px 0;
        opacity: 0.6;
    }

    .nav-hint {
        font-size: 13px;
        color: rgba(128,128,160,0.85);
        text-align: center;
        margin-top: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # ─── SETUP SCREEN ───────────────────────────────────────────────────────
    if not st.session_state.quiz_started and not st.session_state.quiz_finished:
        st.markdown('<div class="quiz-header">🧠 QuizMind</div>', unsafe_allow_html=True)
        st.markdown('<div class="quiz-sub">Upload a PDF and generate a smart, scored quiz instantly.</div>', unsafe_allow_html=True)

        with st.container():
            uploaded_file = st.file_uploader(
                "Upload a PDF file",
                type=["pdf"],
                help="Your notes, textbook chapter, or any study material"
            )

            col1, col2 = st.columns(2)
            with col1:
                num_questions = st.number_input(
                    "Number of Questions",
                    min_value=1, max_value=20, value=5,
                    help="How many questions to generate"
                )
            with col2:
                difficulty = st.selectbox(
                    "Difficulty",
                    ["Easy", "Medium", "Hard"],
                    index=1,
                    help="Easy = recall, Medium = understanding, Hard = analysis"
                )

            time_choice = st.selectbox(
                "Time Limit",
                list(TIME_LIMITS.keys()),
                index=0,
                help="Set a per-question time limit to challenge yourself"
            )

            st.markdown("---")

            gen_col, _ = st.columns([2, 1])
            with gen_col:
                generate = st.button(
                    "✨ Generate Quiz",
                    type="primary",
                    use_container_width=True,
                    disabled=uploaded_file is None
                )

            if uploaded_file is None:
                st.markdown('<div class="nav-hint">↑ Upload a PDF to get started</div>', unsafe_allow_html=True)

        if generate and uploaded_file:
            with st.spinner("📖 Reading your PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                st.session_state.pdf_text = text

            seed = random.randint(1, 99999)
            st.session_state.generation_seed = seed

            with st.spinner(f"🤖 Generating {num_questions} {difficulty} questions..."):
                questions = get_questions(text, num_questions, difficulty, seed)

            if questions:
                st.session_state.questions = questions
                st.session_state.difficulty = difficulty
                st.session_state.time_per_question = TIME_LIMITS[time_choice]
                st.session_state.quiz_started = True
                st.session_state.start_time = time.time()
                st.session_state.question_start_time = time.time()
                st.rerun()

    # ─── QUIZ SCREEN ────────────────────────────────────────────────────────
    elif st.session_state.quiz_started and not st.session_state.quiz_finished:
        questions = st.session_state.questions
        current = st.session_state.current_q
        total = len(questions)

        if current >= total:
            # All answered — auto-finish
            st.session_state.total_time_taken = time.time() - st.session_state.start_time
            st.session_state.score = calculate_score(questions, st.session_state.answers)
            st.session_state.quiz_started = False
            st.session_state.quiz_finished = True
            st.rerun()

        question = questions[current]
        qid = str(question["id"])

        # Header row
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            diff_color = DIFFICULTY_COLORS[st.session_state.difficulty]
            diff_text_colors = {"Easy": "#14532d", "Medium": "#78350f", "Hard": "#7f1d1d"}
            diff_bg_colors = {"Easy": "rgba(34,197,94,0.15)", "Medium": "rgba(245,158,11,0.15)", "Hard": "rgba(239,68,68,0.15)"}
            diff_border_colors = {"Easy": "rgba(34,197,94,0.4)", "Medium": "rgba(245,158,11,0.4)", "Hard": "rgba(239,68,68,0.4)"}
            d_text = diff_text_colors[st.session_state.difficulty]
            d_bg = diff_bg_colors[st.session_state.difficulty]
            d_border = diff_border_colors[st.session_state.difficulty]
            st.markdown(f"""
            <span class="diff-badge" style="background:{d_bg}; color:{d_text}; border:1px solid {d_border};">
                {st.session_state.difficulty.upper()}
            </span>
            """, unsafe_allow_html=True)
        with hcol2:
            answered = len(st.session_state.answers)
            st.markdown(f'<div style="text-align:right; font-size:13px; color:rgba(128,128,160,0.85); margin-top:4px;">✅ {answered}/{total} answered</div>', unsafe_allow_html=True)

        render_progress_bar(current + 1, total)

        # Timer
        time_limit = st.session_state.time_per_question
        if time_limit:
            time_up = render_timer_badge(time_limit, st.session_state.question_start_time)
            if time_up:
                # Auto-skip if time's up
                if qid not in st.session_state.answers:
                    st.session_state.answers[qid] = ""  # Skipped
                st.session_state.current_q += 1
                st.session_state.question_start_time = time.time()
                st.rerun()

        # Question
        st.markdown(f'<div class="question-text">Q{current + 1}. {question["question"]}</div>', unsafe_allow_html=True)

        # Options
        already_answered = qid in st.session_state.answers

        for option in question["options"]:
            letter = get_letter_from_option(option)
            label = option

            if already_answered:
                user_ans = st.session_state.answers[qid]
                correct = question.get("correct_answer", "").strip().upper()
                if letter == correct:
                    st.markdown(f'<div class="answer-correct">✅ {label}</div>', unsafe_allow_html=True)
                elif letter == user_ans:
                    st.markdown(f'<div class="answer-wrong">❌ {label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="answer-neutral">{label}</div>', unsafe_allow_html=True)
            else:
                if st.button(label, key=f"opt_{current}_{letter}"):
                    st.session_state.answers[qid] = letter
                    st.rerun()

        if already_answered:
            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # Navigation
        st.markdown("---")
        nav1, nav2, nav3 = st.columns([1, 2, 1])

        with nav1:
            if current > 0:
                if st.button("← Back", use_container_width=True):
                    st.session_state.current_q -= 1
                    st.session_state.question_start_time = time.time()
                    st.rerun()

        with nav2:
            if already_answered:
                if current < total - 1:
                    if st.button("Next Question →", type="primary", use_container_width=True):
                        st.session_state.current_q += 1
                        st.session_state.question_start_time = time.time()
                        st.rerun()
                else:
                    if st.button("🏁 Finish Quiz", type="primary", use_container_width=True):
                        st.session_state.total_time_taken = time.time() - st.session_state.start_time
                        st.session_state.score = calculate_score(questions, st.session_state.answers)
                        st.session_state.quiz_started = False
                        st.session_state.quiz_finished = True
                        st.rerun()
            else:
                if st.button("Skip →", use_container_width=True):
                    st.session_state.answers[qid] = ""
                    st.session_state.current_q += 1
                    st.session_state.question_start_time = time.time()
                    st.rerun()

        with nav3:
            # Quick quit
            if st.button("Quit", use_container_width=True):
                st.session_state.total_time_taken = time.time() - st.session_state.start_time
                st.session_state.score = calculate_score(questions, st.session_state.answers)
                st.session_state.quiz_started = False
                st.session_state.quiz_finished = True
                st.rerun()

        # Question map
        with st.expander("📋 Question Map", expanded=False):
            cols = st.columns(min(total, 10))
            for i, q in enumerate(questions):
                qkey = str(q["id"])
                ans = st.session_state.answers.get(qkey)
                if ans is None:
                    emoji = "⬜"
                elif ans == "":
                    emoji = "⏭"
                else:
                    correct = q.get("correct_answer", "").strip().upper()
                    emoji = "✅" if ans == correct else "❌"
                with cols[i % min(total, 10)]:
                    if st.button(f"{emoji} {i+1}", key=f"map_{i}"):
                        st.session_state.current_q = i
                        st.session_state.question_start_time = time.time()
                        st.rerun()

        if time_limit:
            time.sleep(0.5)
            st.rerun()

    # ─── RESULTS SCREEN ─────────────────────────────────────────────────────
    elif st.session_state.quiz_finished:
        questions = st.session_state.questions
        total = len(questions)
        score = st.session_state.score
        pct = round((score / total) * 100) if total > 0 else 0
        time_taken = st.session_state.total_time_taken
        skipped = sum(1 for v in st.session_state.answers.values() if v == "")
        wrong = total - score - skipped

        # Grade
        if pct >= 90:
            grade, grade_msg = "A+", "Outstanding! 🌟"
        elif pct >= 80:
            grade, grade_msg = "A", "Excellent work! 🎉"
        elif pct >= 70:
            grade, grade_msg = "B", "Good job! 👍"
        elif pct >= 60:
            grade, grade_msg = "C", "Keep practicing! 📚"
        else:
            grade, grade_msg = "F", "Study more and retry! 💪"

        diff_color = DIFFICULTY_COLORS[st.session_state.difficulty]

        # Score card
        st.markdown(f"""
        <div class="score-card">
            <div style="font-size:13px; opacity:0.8; margin-bottom:8px;">{grade_msg}</div>
            <div class="score-big">{score}/{total}</div>
            <div class="score-label">{pct}% correct · Grade {grade}</div>
            <div style="margin-top:14px;">
                <span class="stat-pill">⏱ {format_time(time_taken)}</span>
                <span class="stat-pill" style="color:#ffffff;">
                    {st.session_state.difficulty}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "✅ Correct", score, "#22c55e"),
            (c2, "❌ Wrong", wrong, "#ef4444"),
            (c3, "⏭ Skipped", skipped, "#f59e0b"),
            (c4, "⚡ Avg time", f"{format_time(time_taken/total if total else 0)}/Q", "#6c63ff"),
        ]
        for col, label, value, color in metrics:
            with col:
                st.markdown(f"""
                <div style="background:rgba(128,128,160,0.1); border:1px solid rgba(128,128,160,0.2); border-radius:12px; padding:12px; text-align:center;">
                    <div style="font-size:12px; color:rgba(128,128,160,0.85); margin-bottom:4px;">{label}</div>
                    <div style="font-size:22px; font-weight:600; color:{color};">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Review toggle
        review_col, retry_col, new_col = st.columns(3)
        with review_col:
            if st.button("📄 Download PDF", use_container_width=True):
                generate_pdf(questions)
                with open("quiz_questions.pdf", "rb") as file:
                    st.download_button(label="⬇ Download Now",data=file,file_name="quiz_questions.pdf",
            
            mime="application/pdf"
        )
            if st.button("📖 Review Answers", use_container_width=True):
                st.session_state.review_mode = not st.session_state.review_mode
                st.rerun()
        with retry_col:
            if st.button("🔄 Retry Same PDF", use_container_width=True):
                # Keep pdf_text, regenerate with new seed
                if st.session_state.pdf_text:
                    seed = random.randint(1, 99999)
                    st.session_state.generation_seed = seed
                    with st.spinner("🤖 Generating fresh questions..."):
                        questions_new = get_questions(
                            st.session_state.pdf_text,
                            len(questions),
                            st.session_state.difficulty,
                            seed
                        )
                    if questions_new:
                        reset_quiz()
                        st.session_state.questions = questions_new
                        st.session_state.difficulty = st.session_state.difficulty
                        st.session_state.quiz_started = True
                        st.session_state.start_time = time.time()
                        st.session_state.question_start_time = time.time()
                        st.rerun()
        with new_col:
            if st.button("🆕 New Quiz", type="primary", use_container_width=True):
                reset_quiz()
                st.rerun()
        
        # Review section
        if st.session_state.review_mode:
            st.markdown("### 📝 Answer Review")
            for i, q in enumerate(questions):
                qid = str(q["id"])
                user_ans = st.session_state.answers.get(qid, "")
                correct_letter = q.get("correct_answer", "").strip().upper()
                correct_text = get_correct_option_text(q)

                is_correct = user_ans == correct_letter
                is_skipped = user_ans == ""

                if is_skipped:
                    icon, box_class = "⏭", "answer-skipped"
                    status = "Skipped"
                elif is_correct:
                    icon, box_class = "✅", "answer-correct"
                    status = "Correct"
                else:
                    icon, box_class = "❌", "answer-wrong"
                    status = "Wrong"

                with st.expander(f"{icon} Q{i+1}. {q['question'][:70]}...", expanded=False):
                    st.markdown(f"**Full question:** {q['question']}")
                    st.markdown("**Options:**")
                    for opt in q["options"]:
                        letter = get_letter_from_option(opt)
                        if letter == correct_letter:
                            st.markdown(f"✅ **{opt}** ← correct")
                        elif letter == user_ans and not is_skipped:
                            st.markdown(f"❌ ~~{opt}~~ ← your answer")
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;{opt}")
                    if is_skipped:
                        st.markdown("*You skipped this question.*")


if __name__ == "__main__":
    main()