#!/usr/bin/env python3

import pdfplumber
import re

def debug_pdf_parsing():
    pdf_path = "data/icici/icic_sample.pdf"
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    print("Full extracted text:")
    print(text)
    print("\n" + "="*80 + "\n")
    
    # Extract transaction lines
    lines = text.split('\n')
    print(f"Total lines: {len(lines)}")
    
    for i, line in enumerate(lines):
        print(f"Line {i}: '{line}'")
        
        # Check if line starts with date pattern
        if re.match(r'\d{2}-\d{2}-\d{4}', line.strip()):
            print(f"  -> MATCHES DATE PATTERN!")
            
            # Split by multiple spaces
            parts = re.split(r'\s{2,}', line.strip())
            print(f"  -> Parts: {parts}")
            
            if len(parts) >= 4:
                date = parts[0].strip()
                description = parts[1].strip()
                print(f"  -> Date: '{date}', Description: '{description}'")
                
                # Extract amounts
                amounts = []
                for j, part in enumerate(parts[2:]):
                    print(f"  -> Part {j+2}: '{part}'")
                    numeric_match = re.search(r'-?\d+\.?\d*', part)
                    if numeric_match:
                        amounts.append(numeric_match.group())
                        print(f"  -> Found amount: {numeric_match.group()}")
                
                print(f"  -> All amounts: {amounts}")
                print(f"  -> Amounts count: {len(amounts)}")
        
        print("-" * 50)

if __name__ == "__main__":
    debug_pdf_parsing()
