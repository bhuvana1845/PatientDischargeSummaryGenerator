import re
from pypdf import PdfReader


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_path):
    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return clean_text(text)


def extract_section(text, section_name, next_sections):
    pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)(?=\n(?:{'|'.join(map(re.escape, next_sections))})\s*:|\Z)"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return "Not provided"


def extract_patient_information(text):

    text = clean_text(text)

    patient_info = {
        "patient_name": "Not provided",
        "age": "Not provided",
        "gender": "Not provided",
        "diagnosis": "Not provided",
        "medications": "Not provided",
        "laboratory_results": "Not provided",
        "physician_recommendations": "Not provided",
        "follow_up": "Not provided",
        "allergies": "Not provided"
    }

    name_match = re.search(
        r"Patient\s+Name\s*:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if name_match:
        patient_info["patient_name"] = name_match.group(1).strip()

    age_match = re.search(
        r"Age\s*:\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if age_match:
        patient_info["age"] = age_match.group(1)

    gender_match = re.search(
        r"Gender\s*:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if gender_match:
        patient_info["gender"] = gender_match.group(1).strip()

    sections = [
        "Diagnosis",
        "Medications",
        "Laboratory Results",
        "Physician Recommendations",
        "Follow-up",
        "Allergies"
    ]

    for section in sections:

        next_sections = [
            s for s in sections
            if s != section
        ]

        value = extract_section(
            text,
            section,
            next_sections
        )

        key = {
            "Diagnosis": "diagnosis",
            "Medications": "medications",
            "Laboratory Results": "laboratory_results",
            "Physician Recommendations": "physician_recommendations",
            "Follow-up": "follow_up",
            "Allergies": "allergies"
        }[section]

        if value:
            patient_info[key] = value

    return patient_info


def extract_text_from_ehr(text):
    return clean_text(text)


def create_patient_context(patient_info):

    context = f"""
Patient Name: {patient_info.get('patient_name', 'Not provided')}
Age: {patient_info.get('age', 'Not provided')}
Gender: {patient_info.get('gender', 'Not provided')}

Diagnosis:
{patient_info.get('diagnosis', 'Not provided')}

Medications:
{patient_info.get('medications', 'Not provided')}

Laboratory Results:
{patient_info.get('laboratory_results', 'Not provided')}

Physician Recommendations:
{patient_info.get('physician_recommendations', 'Not provided')}

Follow-up:
{patient_info.get('follow_up', 'Not provided')}

Allergies:
{patient_info.get('allergies', 'Not provided')}
"""

    return context.strip()