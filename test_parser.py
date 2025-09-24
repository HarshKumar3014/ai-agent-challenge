#!/usr/bin/env python3
"""
Test suite for bank statement parsers

This module provides comprehensive testing for the Agent-as-Coder generated parsers.
It implements the T4 requirement from the specification: strict DataFrame.equals() validation.

Key Features:
- DataFrame.equals() validation for exact CSV matching
- Detailed error reporting with data comparison
- Support for multiple bank parsers
- Integration with pytest framework

Test Requirements (T4):
- Assert that parse() output equals the provided CSV via DataFrame.equals
- Validate shape, columns, data types, and content
- Provide detailed debugging information on failures

Usage:
    python test_parser.py                    # Run tests directly
    python -m pytest test_parser.py -v      # Run with pytest
    ./demo.sh                               # Run as part of demo

Author: AI Assistant
License: MIT
"""

import pandas as pd
import sys
import os

def test_icici_parser() -> None:
    """
    Test the ICICI bank statement parser.
    
    This test implements the T4 requirement: strict DataFrame.equals() validation.
    It ensures that the generated parser produces output that exactly matches
    the expected CSV file in all aspects (shape, columns, data types, content).
    
    Test Process:
    1. Import the generated parser from custom_parsers.icici_parser
    2. Parse the sample PDF file
    3. Load the expected CSV output
    4. Perform DataFrame.equals() validation
    5. Provide detailed error reporting if validation fails
    
    Validation Criteria:
        - Shape: (100, 5) - 100 transactions, 5 columns
        - Columns: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
        - Data types: Exact match with expected CSV
        - Content: All values identical to expected CSV
    
    Raises:
        AssertionError: If DataFrame.equals() validation fails
        ImportError: If parser module cannot be imported
        FileNotFoundError: If PDF or CSV files are missing
        
    Example:
        >>> test_icici_parser()
        Testing ICICI parser...
        📊 Comparing DataFrames using DataFrame.equals...
        ✅ DataFrames are exactly equal!
        ✅ ICICI parser test PASSED!
    """
    print("Testing ICICI parser...")
    
    try:
        # Import the generated parser
        from custom_parsers.icici_parser import parse
        
        # Parse the sample PDF file
        result_df = parse("data/icici/icici sample.pdf")
        expected_df = pd.read_csv("data/icici/result.csv")
        
        # T4: Use DataFrame.equals for validation as specified in image
        print(f"📊 Comparing DataFrames using DataFrame.equals...")
        print(f"   Result shape: {result_df.shape}")
        print(f"   Expected shape: {expected_df.shape}")
        
        # Perform strict DataFrame.equals() validation
        if result_df.equals(expected_df):
            print("✅ DataFrames are exactly equal!")
        else:
            # Detailed error reporting for debugging
            print("❌ DataFrames are not equal")
            print("   Result columns:", list(result_df.columns))
            print("   Expected columns:", list(expected_df.columns))
            print("   Result dtypes:", result_df.dtypes.to_dict())
            print("   Expected dtypes:", expected_df.dtypes.to_dict())
            
            # Show first few rows for debugging
            print("   Result head:")
            print(result_df.head())
            print("   Expected head:")
            print(expected_df.head())
            
            assert False, "DataFrame.equals() failed - parser output doesn't match expected CSV"
        
        print("✅ ICICI parser test PASSED!")
        print(f"✅ Parsed {len(result_df)} transactions successfully")
        
    except Exception as e:
        print(f"❌ ICICI parser test ERROR: {e}")
        assert False, f"Parser test failed with error: {e}"

def main():
    """Run all tests"""
    print("🧪 Running parser tests...")
    
    try:
        test_icici_parser()
        print("🎉 All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Some tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
