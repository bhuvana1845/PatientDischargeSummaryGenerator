import streamlit as st
import os
import tempfile

from preprocess import (
    extract_text_from_pdf,
    extract_text_from_ehr,
    extract_patient_information,
    create_patient_context
)

from rag import MedicalRAG
from llm import generate_patient_summary
from validator import validate_summary
from pdf_generator import create_discharge_pdf


st.set_page_config(
    page_title="Patient-Friendly Discharge Summary Generator",
    page_icon="🏥"
)

st.title("Patient-Friendly Discharge Summary Generator")

st.write(
    "Generate a simple and understandable discharge summary "
    "from an Electronic Health Record (EHR)."
)

st.warning(
    "This is an academic prototype. The generated summary "
    "should be reviewed by a qualified healthcare professional."
)


st.subheader("Enter Patient Information")

input_method = st.radio(
    "Select input method",
    ["Upload EHR PDF", "Enter EHR Note"]
)


ehr_text = ""


if input_method == "Upload EHR PDF":

    uploaded_file = st.file_uploader(
        "Upload EHR PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_pdf_path = temp_file.name

            ehr_text = extract_text_from_pdf(temp_pdf_path)

            os.remove(temp_pdf_path)

            if ehr_text.strip():
                st.success("EHR information extracted successfully.")

                with st.expander("View EHR Information"):
                    st.text(ehr_text)

            else:
                st.error("No readable text was found in the PDF.")

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

else:

    ehr_text = st.text_area(
        "EHR Note",
        height=350,
        placeholder="""Patient Name: John Doe
Age: 55
Gender: Male

Diagnosis:
Type 2 Diabetes Mellitus and Hypertension.

Medications:
Metformin 500 mg twice daily.
Amlodipine 5 mg once daily.

Laboratory Results:
HbA1c: 7.2%
Blood Pressure: 138/86 mmHg.

Physician Recommendations:
Continue medications and maintain a healthy diet.

Follow-up:
Follow up with physician after 2 weeks.

Allergies:
No known drug allergies."""
    )


if st.button("Generate Discharge Summary"):

    if not ehr_text.strip():

        st.error("Please upload an EHR PDF or enter an EHR note.")

    else:

        with st.spinner("Processing EHR information..."):

            try:
                cleaned_ehr = extract_text_from_ehr(ehr_text)

                patient_info = extract_patient_information(
                    cleaned_ehr
                )

                patient_context = create_patient_context(
                    patient_info
                )

            except Exception as e:

                st.error(
                    f"Error during EHR preprocessing: {e}"
                )

                st.stop()


        st.subheader("Extracted Patient Information")

        st.write(
            "**Patient Name:**",
            patient_info.get("patient_name", "Not provided")
        )

        st.write(
            "**Age:**",
            patient_info.get("age", "Not provided")
        )

        st.write(
            "**Gender:**",
            patient_info.get("gender", "Not provided")
        )

        st.write(
            "**Diagnosis:**",
            patient_info.get("diagnosis", "Not provided")
        )

        st.write(
            "**Medications:**",
            patient_info.get("medications", "Not provided")
        )

        st.write(
            "**Laboratory Results:**",
            patient_info.get(
                "laboratory_results",
                "Not provided"
            )
        )

        st.write(
            "**Physician Recommendations:**",
            patient_info.get(
                "physician_recommendations",
                "Not provided"
            )
        )

        st.write(
            "**Follow-up:**",
            patient_info.get(
                "follow_up",
                "Not provided"
            )
        )

        st.write(
            "**Allergies:**",
            patient_info.get(
                "allergies",
                "Not provided"
            )
        )


        with st.spinner("Retrieving relevant medical information..."):

            try:

                rag = MedicalRAG()

                query = (
                    patient_info.get("diagnosis", "")
                    + " "
                    + patient_info.get("medications", "")
                    + " "
                    + patient_info.get("laboratory_results", "")
                )

                retrieved_documents = rag.retrieve_documents(
                    query,
                    top_k=3
                )

            except Exception as e:

                st.error(
                    f"Error during knowledge retrieval: {e}"
                )

                st.stop()


        medical_context = ""

        for document in retrieved_documents:

            medical_context += (
                "\nSource: "
                + document["filename"]
                + "\n"
                + document["text"]
                + "\n"
            )


        with st.spinner(
            "Generating patient-friendly discharge summary..."
        ):

            summary = generate_patient_summary(
                patient_context,
                medical_context
            )


        if summary.startswith("ERROR:"):

            st.error(summary)

            st.stop()


        st.subheader("Patient-Friendly Discharge Summary")

        st.markdown(summary)


        st.subheader("Validation")

        try:

            validation_result = validate_summary(
                summary,
                patient_info
            )

            if isinstance(validation_result, dict):

                score = validation_result.get(
                    "score",
                    0
                )

                valid = validation_result.get(
                    "valid",
                    False
                )

                missing_sections = validation_result.get(
                    "missing_sections",
                    []
                )

                missing_information = validation_result.get(
                    "missing_information",
                    []
                )

                message = validation_result.get(
                    "message",
                    ""
                )

                st.write(
                    f"Validation Score: {score}%"
                )

                if valid:
                    st.success(
                        "The generated summary passed basic validation."
                    )
                else:
                    st.warning(
                        "The generated summary requires review."
                    )

                if message:
                    st.write(message)

                if missing_sections:

                    st.write(
                        "Missing Sections:",
                        ", ".join(missing_sections)
                    )

                if missing_information:

                    st.write(
                        "Missing Information:",
                        ", ".join(missing_information)
                    )

        except Exception as e:

            st.warning(
                f"Validation could not be completed: {e}"
            )


        st.subheader("Download")

        try:

            os.makedirs(
                "output",
                exist_ok=True
            )

            pdf_path = create_discharge_pdf(
                summary,
                output_path="output/discharge_summary.pdf"
            )

            if os.path.exists(pdf_path):

                with open(
                    pdf_path,
                    "rb"
                ) as pdf_file:

                    st.download_button(
                        "Download Discharge Summary PDF",
                        data=pdf_file,
                        file_name="patient_friendly_discharge_summary.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:

            st.error(
                f"Error generating PDF: {e}"
            )
