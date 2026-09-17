from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import mm
import os


def create_discharge_pdf(summary, output_path="output/discharge_summary.pdf"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    story = []

    lines = summary.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        if line.startswith("# Patient-Friendly Discharge Summary"):

            story.append(
                Paragraph(
                    "Patient-Friendly Discharge Summary",
                    title_style
                )
            )

        elif line.startswith("## "):

            heading = line.replace("## ", "").strip()

            story.append(
                Paragraph(
                    heading,
                    heading_style
                )
            )

        else:

            safe_line = (
                line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            story.append(
                Paragraph(
                    safe_line,
                    body_style
                )
            )

    document.build(story)

    return output_path