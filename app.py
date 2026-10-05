
import os
import io
import re

import streamlit as st
from PIL import Image
import pytesseract
from pypdf import PdfReader
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# AI STUDYLENS
# Multimodal AI Study Assistant
# ============================================================

st.set_page_config(
    page_title="AI StudyLens",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if genai is not None and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None
else:
    client = None


# ============================================================
# TESSERACT OCR SETUP
# ============================================================

possible_tesseract_paths = [
    r"C:\Users\prava\Downloads\AIFULLSTACK\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
]

for tesseract_path in possible_tesseract_paths:
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        break


# ============================================================
# SESSION STATE
# ============================================================

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "image_text" not in st.session_state:
    st.session_state.image_text = ""

if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""

if "all_text" not in st.session_state:
    st.session_state.all_text = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "quiz" not in st.session_state:
    st.session_state.quiz = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def update_all_text():
    """Combine PDF, image and manual notes."""

    combined = (
        st.session_state.pdf_text
        + "\n"
        + st.session_state.image_text
        + "\n"
        + st.session_state.manual_text
    )

    st.session_state.all_text = clean_text(combined)


def clean_text(text):
    """Clean unnecessary spaces and blank lines."""

    if not text:
        return ""

    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_pdf_text(uploaded_file):
    """Extract text from uploaded PDF."""

    try:
        pdf_bytes = uploaded_file.getvalue()

        pdf_stream = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_stream)

        pages = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

        return clean_text("\n".join(pages))

    except Exception as e:
        return f"ERROR: {e}"


def extract_image_text(image):
    """Extract text from image using Tesseract OCR."""

    try:
        text = pytesseract.image_to_string(image)

        return clean_text(text)

    except Exception as e:
        return f"ERROR: {e}"


def ask_gemini(question, study_material):
    """Ask Gemini a question using the uploaded study material."""

    if client is None:
        return None

    if not study_material:
        return None

    # Limit context to avoid excessively large prompts.
    context = study_material[:30000]

    prompt = f"""
You are AI StudyLens, an AI study assistant.

Answer the student's question using the study material provided below.

IMPORTANT RULES:
1. Give a clear and easy-to-understand answer.
2. Prefer information from the provided study material.
3. Do not invent information that is not supported by the material.
4. If the answer is not available in the material, clearly say:
   "This information is not available in the uploaded study material."
5. Use simple student-friendly language.
6. Use bullet points when useful.
7. For technical questions, give a short example when appropriate.

STUDY MATERIAL:
----------------
{context}
----------------

STUDENT QUESTION:
{question}

ANSWER:
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Gemini Error: {e}"


def generate_ai_summary(study_material):
    """Generate an AI summary."""

    if client is None:
        return None

    context = study_material[:30000]

    prompt = f"""
You are AI StudyLens.

Create a useful study summary from the following study material.

Requirements:
- Use simple language.
- Include the important concepts.
- Use headings and bullet points.
- Keep it suitable for college students.
- Do not add unrelated information.

STUDY MATERIAL:
----------------
{context}
----------------

SUMMARY:
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Gemini Error: {e}"


def generate_ai_quiz(study_material, number):
    """Generate quiz questions using Gemini."""

    if client is None:
        return None

    context = study_material[:25000]

    prompt = f"""
You are AI StudyLens.

Create {number} multiple-choice questions from the study material.

For every question provide:

Q1. Question
A. Option
B. Option
C. Option
D. Option
Answer: A
Explanation: Short explanation

Do not use information outside the study material.

STUDY MATERIAL:
----------------
{context}
----------------

QUIZ:
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Gemini Error: {e}"


def basic_summary(text):
    """Fallback summary when Gemini is unavailable."""

    if not text:
        return "No study material available."

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    useful_sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]

    return " ".join(useful_sentences[:5])


def basic_answer(text, question):
    """Fallback keyword-based answer."""

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower()
        )
    )

    stop_words = {
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "explain",
        "about",
        "please",
        "does",
        "can",
        "the",
        "and",
        "for",
        "from",
        "give",
        "tell"
    }

    question_words -= stop_words

    scored_sentences = []

    for sentence in sentences:

        sentence_words = set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                sentence.lower()
            )
        )

        common_words = question_words.intersection(
            sentence_words
        )

        if common_words:

            score = len(common_words)

            scored_sentences.append(
                (score, sentence)
            )

    scored_sentences.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        sentence
        for score, sentence in scored_sentences[:5]
    ]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        🔍 AI StudyLens
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;font-size:20px;">
        Multimodal AI Study Assistant
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        📄 PDF &nbsp; | &nbsp;
        🖼️ Images &nbsp; | &nbsp;
        🤖 AI Q&A &nbsp; | &nbsp;
        📝 Summary &nbsp; | &nbsp;
        ❓ Quiz
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 AI StudyLens")

    st.write(
        "Upload your study material and learn with AI."
    )

    st.divider()

    if client is not None:

        st.success("🤖 Gemini AI: Connected")

    else:

        st.warning(
            "🤖 Gemini AI: Not connected"
        )

        st.caption(
            "Add GEMINI_API_KEY to your .env file "
            "to enable AI answers."
        )

    st.divider()

    st.write("📄 PDF Study Material")
    st.write("🖼️ Image OCR")
    st.write("📝 Notes")
    st.write("🤖 AI Questions")
    st.write("📋 AI Summary")
    st.write("❓ AI Quiz")

    st.divider()

    if st.button("🗑️ Clear All Data"):

        st.session_state.pdf_text = ""
        st.session_state.image_text = ""
        st.session_state.manual_text = ""
        st.session_state.all_text = ""
        st.session_state.chat_history = []
        st.session_state.quiz = []

        st.success("All data cleared!")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📄 PDF",
        "🖼️ Image OCR",
        "📝 Notes",
        "🤖 Ask AI",
        "❓ Quiz"
    ]
)


# ============================================================
# PDF TAB
# ============================================================

with tab1:

    st.header("📄 Upload Study Material")

    st.write(
        "Upload a PDF containing your class notes, "
        "textbook or study material."
    )

    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_upload"
    )

    if pdf_file:

        st.success(
            f"PDF selected: {pdf_file.name}"
        )

        if st.button(
            "🔍 Read PDF",
            key="read_pdf"
        ):

            with st.spinner(
                "Reading your PDF..."
            ):

                extracted = extract_pdf_text(
                    pdf_file
                )

                if extracted.startswith("ERROR:"):

                    st.error(extracted)

                elif extracted:

                    st.session_state.pdf_text = extracted

                    update_all_text()

                    st.success(
                        "✅ PDF read successfully!"
                    )

                else:

                    st.warning(
                        "No selectable text was found "
                        "in this PDF."
                    )

        if st.session_state.pdf_text:

            st.subheader(
                "📖 Extracted Study Material"
            )

            st.text_area(
                "PDF content",
                st.session_state.pdf_text,
                height=350,
                key="pdf_content"
            )

            st.info(
                f"Characters extracted: "
                f"{len(st.session_state.pdf_text):,}"
            )


# ============================================================
# IMAGE OCR TAB
# ============================================================

with tab2:

    st.header("🖼️ Image to Text")

    st.write(
        "Upload a photo or screenshot of your study notes."
    )

    image_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        key="image_upload"
    )

    if image_file:

        image = Image.open(image_file)

        st.image(
            image,
            caption="Uploaded Study Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Extract Text from Image",
            key="ocr_button"
        ):

            with st.spinner(
                "Reading image with OCR..."
            ):

                extracted = extract_image_text(
                    image
                )

                if extracted.startswith("ERROR:"):

                    st.error(extracted)

                elif extracted:

                    st.session_state.image_text = extracted

                    update_all_text()

                    st.success(
                        "✅ Text extracted from image!"
                    )

                else:

                    st.warning(
                        "No text was detected in the image."
                    )

        if st.session_state.image_text:

            st.subheader(
                "📝 Extracted Image Text"
            )

            st.text_area(
                "OCR result",
                st.session_state.image_text,
                height=300,
                key="image_content"
            )


# ============================================================
# NOTES TAB
# ============================================================

with tab3:

    st.header("📝 Add Your Study Notes")

    notes = st.text_area(
        "Paste or type your notes here",
        height=300,
        placeholder=(
            "Example:\n\n"
            "A database management system (DBMS) "
            "is software used to create, manage and "
            "organize databases."
        )
    )

    if st.button(
        "➕ Add Notes",
        key="add_notes"
    ):

        if notes.strip():

            st.session_state.manual_text = notes

            update_all_text()

            st.success(
                "✅ Notes added successfully!"
            )

        else:

            st.warning(
                "Please enter some notes first."
            )


# ============================================================
# ASK AI TAB
# ============================================================

with tab4:

    st.header("🤖 Ask AI About Your Study Material")

    if not st.session_state.all_text:

        st.info(
            "👆 First upload a PDF, image, or add notes."
        )

    else:

        st.success(
            "✅ Your study material is ready for questions."
        )

        st.subheader(
            "❓ Ask a Question"
        )

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What is a DBMS? "
                "Explain it in simple words."
            ),
            key="question_input"
        )

        if st.button(
            "🤖 Get AI Answer",
            key="ask_ai"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "AI StudyLens is thinking..."
                ):

                    if client is not None:

                        answer = ask_gemini(
                            question,
                            st.session_state.all_text
                        )

                    else:

                        answers = basic_answer(
                            st.session_state.all_text,
                            question
                        )

                        if answers:

                            answer = "\n\n".join(
                                [
                                    "• " + item
                                    for item in answers
                                ]
                            )

                        else:

                            answer = (
                                "No matching information "
                                "was found in your study material."
                            )

                st.subheader(
                    "💡 Answer"
                )

                st.markdown(answer)

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

        if st.session_state.chat_history:

            st.divider()

            st.subheader(
                "💬 Previous Questions"
            )

            for item in reversed(
                st.session_state.chat_history
            ):

                with st.expander(
                    f"❓ {item['question']}"
                ):

                    st.markdown(
                        item["answer"]
                    )


# ============================================================
# QUIZ TAB
# ============================================================

with tab5:

    st.header("❓ AI Quiz Generator")

    if not st.session_state.all_text:

        st.info(
            "First upload study material or add notes."
        )

    else:

        number = st.slider(
            "Number of questions",
            min_value=3,
            max_value=10,
            value=5
        )

        if st.button(
            "🎯 Generate AI Quiz",
            key="generate_quiz"
        ):

            if client is not None:

                with st.spinner(
                    "Creating your quiz..."
                ):

                    quiz = generate_ai_quiz(
                        st.session_state.all_text,
                        number
                    )

                st.session_state.quiz = quiz

            else:

                st.warning(
                    "Gemini AI is not connected. "
                    "Add your API key to generate an AI quiz."
                )

        if st.session_state.quiz:

            st.subheader(
                "🧠 Your AI Quiz"
            )

            st.markdown(
                st.session_state.quiz
            )


# ============================================================
# SUMMARY
# ============================================================

st.divider()

st.header("📋 Study Summary")

if st.session_state.all_text:

    if st.button(
        "✨ Generate AI Summary",
        key="summary"
    ):

        with st.spinner(
            "Creating your study summary..."
        ):

            if client is not None:

                summary = generate_ai_summary(
                    st.session_state.all_text
                )

            else:

                summary = basic_summary(
                    st.session_state.all_text
                )

        st.subheader(
            "📚 Summary"
        )

        st.markdown(summary)

else:

    st.info(
        "Upload a PDF, image or notes to create a summary."
    )


# ============================================================
# MATERIAL STATUS
# ============================================================

st.divider()

st.subheader("📊 Study Material Status")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "PDF Characters",
        f"{len(st.session_state.pdf_text):,}"
    )

with col2:

    st.metric(
        "Image Characters",
        f"{len(st.session_state.image_text):,}"
    )

with col3:

    st.metric(
        "Total Characters",
        f"{len(st.session_state.all_text):,}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">

    <h3>🔍 AI StudyLens</h3>

    <p>
    Multimodal AI Study Assistant
    </p>

    <p>
    📄 PDF • 🖼️ Image • 📝 Text • 🤖 AI • ❓ Quiz
    </p>

    </div>
    """,
    unsafe_allow_html=True
)
