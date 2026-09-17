import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

def generate_patient_summary(patient_context, medical_context):
    prompt = f'''You are a medical discharge summary assistant.

Use ONLY the patient record and retrieved medical context. Do not invent diagnoses, medicines, doses, test results, or instructions. Do not change medication names or doses. If information is missing, write "Not provided". This is a draft for review by a qualified healthcare professional.

PATIENT RECORD:
{patient_context}

RETRIEVED MEDICAL CONTEXT:
{medical_context}

Generate exactly these sections:
# Patient-Friendly Discharge Summary
## 1. Patient Information
## 2. Diagnosis
## 3. Medicines
## 4. Laboratory Results
## 5. Doctor's Recommendations
## 6. Follow-up
## 7. Allergies
## 8. Warning Signs
## 9. Emergency Guidance
## 10. Important Note

Keep the language clear, concise and patient-friendly.
'''
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1}}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        result = r.json()
        return result.get("response", "").strip() or "Error: Llama 3 returned an empty response."
    except requests.exceptions.ConnectionError:
        return ("ERROR: Cannot connect to Ollama. Make sure Ollama is installed and running, "
                "then run: ollama pull llama3")
    except requests.exceptions.Timeout:
        return "ERROR: Llama 3 took too long to generate the summary."
    except requests.exceptions.RequestException as e:
        return f"ERROR: Ollama request failed: {e}"
    except Exception as e:
        return f"ERROR: {e}"
