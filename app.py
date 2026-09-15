import streamlit as st
import tempfile
import os

from preprocess import extract_text_from_pdf
from rag import load_vector_store, retrieve_documents
from llm import generate_patient_summary


# ----------------------------
# Streamlit Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Patient-Friendly Discharge Summary Generator",
    
    layout="wide"
)

st.title("Patient-Friendly Discharge Summary Generator")
st.markdown("""
This application converts complex hospital discharge summaries into
simple, patient-friendly language using **Retrieval-Augmented Generation (RAG)**
and **Llama 3**.
""")

st.divider()

# ----------------------------
# Sidebar
# ----------------------------

#st.sidebar.header("Project Information")

#st.sidebar.info("""
### Technologies Used

#- Llama 3 (Ollama)
#- LangChain
#- FAISS
#- HuggingFace Embeddings
#- Streamlit
#- RAG Pipeline
#""")

#st.sidebar.success("Upload a discharge summary PDF to begin.")

# ----------------------------
# Upload PDF
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload Discharge Summary (PDF)",
    type=["pdf"]
)

# ----------------------------
# Main Workflow
# ----------------------------

if uploaded_file is not None:

    with st.spinner("Uploading PDF..."):

        temp_dir = tempfile.mkdtemp()

        pdf_path = os.path.join(temp_dir, uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    st.divider()

    # ----------------------------
    # Extract Text
    # ----------------------------

    with st.spinner("Extracting text from PDF..."):

        extracted_text = extract_text_from_pdf(pdf_path)

    st.subheader("Extracted Medical Text")

    with st.expander("View Extracted Text"):

        st.write(extracted_text)

    st.divider()

    # ----------------------------
    # Load Vector Store
    # ----------------------------

    with st.spinner("Loading Vector Database..."):

        vector_db = load_vector_store()

    # ----------------------------
    # Retrieve Relevant Chunks
    # ----------------------------

    with st.spinner("Retrieving relevant medical information..."):

        retrieved_docs = retrieve_documents(
            vector_db,
            extracted_text,
            top_k=5
        )

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    st.subheader("Retrieved Medical Context")

    with st.expander("View Retrieved Chunks"):

        st.write(context)

    st.divider()

    # ----------------------------
    # Generate Summary
    # ----------------------------

    if st.button("Generate Patient-Friendly Summary"):

        with st.spinner("Generating summary using Llama 3..."):

            summary = generate_patient_summary(
                context=context,
                discharge_text=extracted_text
            )

        st.success("Summary Generated Successfully!")

        st.subheader("Patient-Friendly Discharge Summary")

        st.write(summary)

        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="Patient_Friendly_Discharge_Summary.txt",
            mime="text/plain"
        )

else:

    st.info("Please upload a discharge summary PDF.")

st.divider()

#st.markdown(

#)
