#!/usr/bin/env python3

from custom_parsers.icici_parser import parse

result = parse("data/icici/icic_sample.pdf")
print("Parser result:")
print(result)
print(f"Shape: {result.shape}")
print(f"Columns: {result.columns.tolist()}")
