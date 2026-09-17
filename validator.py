def validate_summary(summary, patient_data):
    required = ["Patient Information","Diagnosis","Medicines","Laboratory Results",
                "Doctor's Recommendations","Follow-up","Allergies",
                "Warning Signs","Emergency Guidance","Important Note"]
    missing_sections = [s for s in required if s.lower() not in summary.lower()]
    missing_information = []
    info = patient_data.get("patient_info", {})
    if info.get("Patient Name","Not provided") != "Not provided" and info["Patient Name"].lower() not in summary.lower():
        missing_information.append("Patient Name")
    if info.get("Age","Not provided") != "Not provided" and info["Age"].lower() not in summary.lower():
        missing_information.append("Age")
    if patient_data.get("Medications","Not provided") != "Not provided":
        first_med = patient_data["Medications"].splitlines()[0].strip()
        if first_med and first_med.lower() not in summary.lower():
            missing_information.append("Medications")
    score = max(0, 100 - len(missing_sections)*7 - len(missing_information)*5)
    return {"valid": not missing_sections, "score": score,
            "missing_sections": missing_sections,
            "missing_information": missing_information,
            "message": "Validation passed." if not missing_sections else "Some required sections are missing."}
