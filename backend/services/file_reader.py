import pdfplumber
import pymupdf4llm
import pymupdf
from docx import Document

"""
This module provides functions to read and extract text from PDF and DOCX files. 
It uses pdfplumber for PDFs and python-docx for DOCX files. 
The extracted text is returned as a single string, which can then be processed further by the application.
"""    

def read_pdf(file):
    file.seek(0)
    pdf_bytes = file.read()
    md_text = pymupdf4llm.to_markdown(pymupdf.open(stream=pdf_bytes, filetype="pdf"))
    return md_text

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