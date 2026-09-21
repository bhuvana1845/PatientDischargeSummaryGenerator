import os
import io
import re
import tempfile
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)

from huggingface_hub import InferenceClient


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Patient-Friendly Discharge Summary Generator",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏥 Patient-Friendly Discharge Summary Generator")

st.markdown(
    """
    This application converts complex hospital discharge information
    into simple, patient-friendly language using **Retrieval-Augmented
    Generation (RAG)** and a **Large Language Model (LLM)**.
    """
)

st.divider()


# ============================================================
# CONFIGURATION
# ============================================================

KNOWLEDGE_DIR = Path("med_knowledge")

# Hugging Face model
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_llm():
    token = os.getenv("HF_TOKEN")

    if not token:
        return None

    return InferenceClient(
        provider="hf-inference",
        api_key=token
    )


embedding_model = load_embedding_model()
llm_client = load_llm()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


# ============================================================
# TEXT CLEANING
# ============================================================

def preprocess_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"Page\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text, chunk_size=500, overlap=100):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================================
# LOAD MEDICAL KNOWLEDGE
# ============================================================

@st.cache_data
def load_medical_knowledge():

    documents = []

    if not KNOWLEDGE_DIR.exists():
        return documents

    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:

        try:

            reader = PdfReader(str(pdf_path))

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            if text.strip():

                chunks = create_chunks(
                    preprocess_text(text)
                )

                for chunk in chunks:

                    documents.append({
                        "text": chunk,
                        "source": pdf_path.name
                    })

        except Exception as e:

            st.warning(
                f"Could not read {pdf_path.name}: {e}"
            )

    return documents


# ============================================================
# CREATE VECTOR INDEX
# ============================================================

@st.cache_resource
def create_vector_index():

    documents = load_medical_knowledge()

    if not documents:
        return None, []

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index, documents


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_context(query, top_k=5):

    index, documents = create_vector_index()

    if index is None:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    retrieved = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        retrieved.append({
            "text": documents[index_id]["text"],
            "source": documents[index_id]["source"],
            "score": float(score)
        })

    return retrieved


# ============================================================
# GENERATE SUMMARY USING LLM
# ============================================================

def generate_summary(patient_text, retrieved_context):

    if llm_client is None:

        return None

    context_text = "\n\n".join(
        [
            f"Source: {item['source']}\n"
            f"{item['text']}"
            for item in retrieved_context
        ]
    )

    prompt = f"""
You are a healthcare communication assistant.

Your task is to convert the provided hospital discharge
information into a simple, patient-friendly discharge summary.

IMPORTANT:
- Do not invent medical information.
- Do not change medication doses.
- Do not create diagnoses that are not present.
- Use the retrieved medical knowledge only as supporting context.
- Preserve important clinical information.
- Use simple language that a patient or caregiver can understand.
- If information is unavailable, write "Not provided in the discharge information."

Create the following six sections:

1. Patient Diagnosis Overview
2. Medication Changes and Dosage Instructions
3. Follow-up Appointment Recommendations
4. Lifestyle and Recovery Guidelines
5. Important Warning Signs Requiring Medical Attention
6. Contact Information and Emergency Instructions

PATIENT DISCHARGE INFORMATION:
{patient_text}

RETRIEVED MEDICAL KNOWLEDGE:
{context_text}

Return a clear and structured patient-friendly discharge summary.
"""

    try:

        response = llm_client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical communication "
                        "assistant. Simplify information "
                        "without inventing facts."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=1800,

            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:

        st.error(
            f"LLM generation failed: {e}"
        )

        return None


# ============================================================
# PDF GENERATION
# ============================================================

def create_pdf(summary):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        spaceAfter=8
    )

    story = []

    story.append(
        Paragraph(
            "Patient-Friendly Discharge Summary",
            title_style
        )
    )

    sections = re.split(
        r"\n(?=\d+\.)",
        summary
    )

    for section in sections:

        lines = section.strip().split("\n")

        if not lines:
            continue

        heading = lines[0]

        content = " ".join(
            lines[1:]
        ).strip()

        if not content:

            content = heading

            story.append(
                Paragraph(
                    content,
                    heading_style
                )
            )

        else:

            story.append(
                Paragraph(
                    heading,
                    heading_style
                )
            )

            content = content.replace(
                "&",
                "&amp;"
            )

            story.append(
                Paragraph(
                    content,
                    body_style
                )
            )

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# USER INTERFACE
# ============================================================

st.subheader("📄 Upload Discharge Summary")

uploaded_file = st.file_uploader(
    "Upload the patient's discharge summary PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "✨ Generate Patient-Friendly Summary",
        type="primary"
    ):

        with st.spinner(
            "Extracting information from the discharge summary..."
        ):

            patient_text = extract_text_from_pdf(
                uploaded_file
            )

            patient_text = preprocess_text(
                patient_text
            )

        if not patient_text:

            st.error(
                "Could not extract text from the uploaded PDF."
            )

            st.stop()

        st.subheader(
            "🔍 Extracted Patient Information"
        )

        with st.expander(
            "View extracted information"
        ):

            st.write(patient_text)

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "Retrieving relevant medical knowledge..."
        ):

            retrieved_context = retrieve_context(
                patient_text,
                top_k=5
            )

        st.subheader(
            "📚 Retrieved Medical Context"
        )

        if retrieved_context:

            for item in retrieved_context:

                st.caption(
                    f"Source: {item['source']} "
                    f"| Similarity: {item['score']:.2f}"
                )

                st.write(
                    item["text"]
                )

        else:

            st.warning(
                "No medical knowledge documents were found."
            )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        with st.spinner(
            "Generating patient-friendly discharge summary..."
        ):

            summary = generate_summary(
                patient_text,
                retrieved_context
            )

        if summary:

            st.success(
                "Patient-friendly summary generated successfully!"
            )

            st.subheader(
                "📝 Patient-Friendly Discharge Summary"
            )

            st.markdown(summary)

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            pdf_file = create_pdf(
                summary
            )

            st.download_button(
                label="📥 Download Summary as PDF",
                data=pdf_file,
                file_name="patient_friendly_discharge_summary.pdf",
                mime="application/pdf"
            )

        else:

            st.error(
                "Summary generation could not be completed."
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About the System")

    st.write(
        """
        This system uses:

        • PDF/EHR information extraction  
        • Text preprocessing  
        • Semantic medical knowledge retrieval  
        • Retrieval-Augmented Generation (RAG)  
        • Large Language Model (LLM)  
        • Patient-friendly language generation  
        • PDF report generation
        """
    )

    st.warning(
        """
        This application is intended for
        educational/research purposes.

        Generated information should be
        reviewed by a qualified healthcare
        professional before clinical use.
        """
    )
