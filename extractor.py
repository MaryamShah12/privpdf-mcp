import pytesseract
import pdfplumber
from pdf2image import convert_from_path


def is_text_based(text, min_chars=50):
    return len(text.strip()) > min_chars


def extract_with_pdfplumber(path: str):
    pages_data = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages_data.append({
                "page_no": i,
                "text": text,
                "tables": tables,
                "source": "pdfplumber"
            })
    return pages_data


def extract_with_ocr(path: str):
    pages_data = []
    images = convert_from_path(path)
    for i, page in enumerate(images, start=1):
        text = pytesseract.image_to_string(page)
        pages_data.append({
            "page_no": i,
            "text": text,
            "tables": [],
            "source": "ocr"
        })
    return pages_data


def extract_pdf(path: str):
    """
    Try pdfplumber first; fall back to OCR if the PDF has no readable text.
    Returns (pages_data, method_used).
    """
    pages_data = extract_with_pdfplumber(path)
    full_text = "".join(p["text"] for p in pages_data)

    if is_text_based(full_text):
        return pages_data, "pdfplumber"

    pages_data = extract_with_ocr(path)
    return pages_data, "ocr"