#!/usr/bin/env python3

import pdfplumber
import PyPDF2

def test_pdf_extraction():
    pdf_path = "data/icici/icic_sample.pdf"
    
    print("Testing PDF text extraction...")
    
    # Try pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"PDF has {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                print(f"Page {i+1} text (first 500 chars):")
                print(text[:500] if text else "No text extracted")
                print("-" * 50)
    except Exception as e:
        print(f"pdfplumber error: {e}")
    
    # Try PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"PyPDF2: PDF has {len(pdf_reader.pages)} pages")
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                print(f"Page {i+1} text (first 500 chars):")
                print(text[:500] if text else "No text extracted")
                print("-" * 50)
    except Exception as e:
        print(f"PyPDF2 error: {e}")

if __name__ == "__main__":
    test_pdf_extraction()
