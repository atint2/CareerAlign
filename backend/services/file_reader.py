import pdfplumber
import pymupdf4llm
from docx import Document

"""
This module provides functions to read and extract text from PDF and DOCX files. 
It uses pdfplumber for PDFs and python-docx for DOCX files. 
The extracted text is returned as a single string, which can then be processed further by the application.
"""    

def read_pdf_with_mupdf(file):
    md_text = pymupdf4llm.to_markdown(file)
    return md_text

def read_pdf(file):
    text_parts = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = "\t".join(cell or "" for cell in row)
                    text_parts.append(row_text)

    return "\n".join(text_parts)

def read_docx(file):
    doc = Document(file)
    text_parts = []

    for paragraph in doc.paragraphs:
        stripped = paragraph.text.strip()
        if stripped:
            text_parts.append(stripped)

    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells)
            text_parts.append(row_text)

    return "\n".join(text_parts)