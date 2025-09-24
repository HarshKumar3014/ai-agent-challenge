# Agent-as-Coder: Bank Statement Parser Generator

An intelligent coding agent that automatically generates custom parsers for bank statement PDFs using LangGraph and LLM capabilities.

## 🚀 Quick Start (5 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables
Create a `.env` file with your API keys:
```bash
# Copy the example file
cp env.example .env

# Edit .env and add your API keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**API Key Setup:**
- **Groq (Primary)**: Free tier available at [console.groq.com](https://console.groq.com/) - Uses Llama-3.1-8b-instant
- **OpenAI (Fallback)**: Secondary option if Groq is not available
- **Mock Parser (Backup)**: Reliable fallback when LLM-generated parsers fail tests

### Step 3: Prepare Sample Data
Ensure you have sample data in the correct format:
- `data/{bank_name}/{bank_name}_sample.pdf` - Sample bank statement
- `data/{bank_name}/{bank_name}_sample.csv` - Expected output format

### Step 4: Run the Complete Demo (≤60 seconds)
```bash
./demo.sh
```

Or manually:
```bash
python agent.py --target icici
python -m pytest test_parser.py -v
```

### Step 5: Verify Results
```bash
python test_parser.py
```

## 🧠 Agent Architecture

The agent follows a **plan → generate → test → self-fix** loop pattern:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│    Plan     │───▶│   Generate   │───▶│    Write    │───▶│    Test     │
│   Task      │    │   Parser     │    │   Parser    │    │   Parser    │
│             │    │    Code      │    │    File     │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│     End     │◀───│   Self-Fix   │◀───│   Retry?    │◀───│   Passed?   │
│             │    │   (≤3x)      │    │             │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
```

**Agent Loop Details:**
1. **Plan**: Analyze target bank and locate sample files
2. **Generate**: Use LLM to create parser code based on PDF/CSV samples
3. **Write**: Save parser to `custom_parsers/{bank}_parser.py`
4. **Test**: Validate parser output matches expected CSV using `DataFrame.equals`
5. **Self-Fix**: Automatically retry up to 3 times if tests fail

## 📁 Project Structure

```
├── agent.py                 # Main agent implementation
├── test_parser.py          # Test suite
├── requirements.txt        # Dependencies
├── README.md              # This file
├── data/
│   └── icici/
│       ├── icic_sample.pdf # Sample bank statement
│       └── icic_sample.csv # Expected output
└── custom_parsers/
    └── icici_parser.py     # Generated parser (after running agent)
```

## 🔧 Parser Contract

All generated parsers implement the standard contract:

```python
def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse bank statement PDF and return structured data
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        pandas.DataFrame with columns: Date, Description, Debit, Credit, Balance
    """
```

## 🧪 Testing

The agent automatically tests generated parsers using `DataFrame.equals()` to ensure perfect match with expected CSV output. Run tests manually:

```bash
python test_parser.py
```

## 🎯 Supported Banks

Currently configured for:
- **ICICI Bank** (`--target icici`)

To add new banks, simply add sample data in `data/{bank_name}/` directory and run the agent.

## 🛠️ Technical Details

- **Framework**: LangGraph for agent orchestration
- **LLM**: OpenAI GPT-3.5-turbo (with fallback to mock parsers)
- **Testing**: pandas DataFrame comparison
- **Self-correction**: Up to 3 retry attempts
- **Output**: Clean, minimal Python parsers

## 📝 Example Usage

```bash
# Generate ICICI parser
python agent.py --target icici

# Test the generated parser
python test_parser.py

# The parser is now available at:
# custom_parsers/icici_parser.py
```

## 🔍 Troubleshooting

- **Missing API Key**: Agent uses mock parsers if OpenAI API is unavailable
- **File Not Found**: Ensure sample PDF and CSV files exist in correct directories
- **Parser Fails**: Agent automatically retries up to 3 times with self-correction
- **Test Failures**: Check that PDF format matches expected CSV schema
