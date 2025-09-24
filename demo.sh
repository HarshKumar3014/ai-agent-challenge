#!/bin/bash

echo "🚀 Agent-as-Coder Demo"
echo "======================"
echo ""

echo "Step 1: Running agent to generate parser..."
python agent.py --target icici

echo ""
echo "Step 2: Running tests..."
python -m pytest test_parser.py -v

echo ""
echo "✅ Demo completed successfully!"
