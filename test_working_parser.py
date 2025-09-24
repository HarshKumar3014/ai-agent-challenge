#!/usr/bin/env python3

import pandas as pd
import re
import pdfplumber

def parse_icici_pdf(pdf_path: str) -> pd.DataFrame:
    """Parse ICICI bank statement PDF"""
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    # Extract transaction lines
    lines = text.split('\n')
    transactions = []
    
    for line in lines:
        # Look for lines that start with date pattern (DD-MM-YYYY)
        if re.match(r'\d{2}-\d{2}-\d{4}', line.strip()):
            # Split by multiple spaces (table format)
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 4:
                date = parts[0].strip()
                description = parts[1].strip()
                
                # Extract amounts from the remaining parts
                amounts = []
                for part in parts[2:]:
                    # Extract numeric values (including negative)
                    numeric_match = re.search(r'-?\d+\.?\d*', part)
                    if numeric_match:
                        amounts.append(numeric_match.group())
                
                print(f"Line: {line}")
                print(f"Parts: {parts}")
                print(f"Amounts: {amounts}")
                print("-" * 50)
                
                if len(amounts) >= 2:
                    # The format appears to be: Date, Description, Debit/Credit, Balance
                    # We need to determine which is debit and which is credit
                    
                    # If there are 2 amounts, one is debit/credit and one is balance
                    if len(amounts) == 2:
                        # Determine if first amount is debit or credit based on description
                        if any(keyword in description.lower() for keyword in ['debit', 'withdrawal', 'payment', 'transfer', 'purchase', 'bill', 'emi', 'charge', 'upi', 'neft', 'imps', 'card', 'cheque']):
                            debit_amt = amounts[0]
                            credit_amt = ''
                        else:
                            debit_amt = ''
                            credit_amt = amounts[0]
                        balance = amounts[1]
                    else:
                        # If there are 3 amounts, it might be debit, credit, balance
                        debit_amt = amounts[0] if amounts[0] else ''
                        credit_amt = amounts[1] if amounts[1] else ''
                        balance = amounts[2] if amounts[2] else ''
                    
                    transactions.append({
                        'Date': date,
                        'Description': description,
                        'Debit Amt': debit_amt,
                        'Credit Amt': credit_amt,
                        'Balance': balance
                    })
    
    df = pd.DataFrame(transactions)
    
    # Convert data types to match expected format
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%d-%m-%Y')
        df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce')
        df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce')
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    
    return df

if __name__ == "__main__":
    result = parse_icici_pdf("data/icici/icic_sample.pdf")
    print("Result:")
    print(result.head(10))
    print(f"Total rows: {len(result)}")
