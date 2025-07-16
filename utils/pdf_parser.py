# utils/pdf_parser.py

import fitz  # PyMuPDF  # type: ignore
from typing import List

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts and returns all text from the given PDF file.
    
    Parameters:
        file_path (str): Path to the input PDF file.

    Returns:
        str: Combined text from all pages of the PDF.
    """
    doc = fitz.open(file_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    doc.close()
    return all_text

def extract_text_per_page(file_path: str) -> List[str]:
    """
    Extracts and returns text from each page of the PDF separately.

    Parameters:
        file_path (str): Path to the input PDF file.

    Returns:
        List[str]: A list where each element is the text of a single page.
    """
    doc = fitz.open(file_path)
    text_pages = [page.get_text() for page in doc]
    doc.close()
    return text_pages
