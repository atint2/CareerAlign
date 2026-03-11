"""
This module provides functions to read and extract text from PDF and DOCX files. 
It uses pdfplumber for PDFs and python-docx for DOCX files. 
The extracted text is returned as a single string, which can then be processed further by the application.
"""    

from docx import Document
import tempfile
import os
import nest_asyncio

nest_asyncio.apply()

def parse_with_llama(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        from llama_parse import LlamaParse
        parser = LlamaParse(
            api_key="llx-FbXeScgjqS6NQrSxwn9b2ei28jg3jNKIP8otw0mAgzAkEHec",
            result_type="markdown",  
            verbose=True
        )

        documents = parser.load_data(tmp_file_path, extra_info={"file_name": file.name})

        if documents:
            file_contents = " ".join([doc.get_content() for doc in documents])
            print(file_contents)
        else:
            file_contents = ""
            print("No content extracted.")

    finally:
        # 5. Clean up the temporary file
        os.remove(tmp_file_path)
        
    return file_contents