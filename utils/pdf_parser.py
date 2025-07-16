# utils/pdf_parser.py

import PyPDF2
from typing import List

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts and returns all text from the given PDF file.
    
    Parameters:
        file_path (str): Path to the input PDF file.

    Returns:
        str: Combined text from all pages of the PDF.
    """
    all_text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text

def extract_text_per_page(file_path: str) -> List[str]:
    """
    Extracts and returns text from each page of the PDF separately.

    Parameters:
        file_path (str): Path to the input PDF file.

    Returns:
        List[str]: A list where each element is the text of a single page.
    """
    text_pages = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text_pages.append(page.extract_text() or "")
    return text_pages
