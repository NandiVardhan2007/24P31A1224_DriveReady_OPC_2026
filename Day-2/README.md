# Day 2: Building Your Own AI Agent (BrightPage Stationery)

## 📌 Task Overview
Day 2 focuses on moving from simple prompt completions to designing a robust, domain-specific AI Agent for **BrightPage Stationery & Office Supplies**. The implementation covers few-shot prompting, schema-constrained structured output generation, multi-turn conversation memory, and systematic agent verification.

---

## 🏗️ Key Components & Implementation

### 1. Few-Shot Intent Classification
* Implements a specialized intent classifier using few-shot in-context learning.
* Categorizes customer queries strictly into one of three standard classes:
  - `order`: Requests to purchase or reserve items.
  - `inquiry`: Questions regarding price, availability, or store policies.
  - `complaint`: Quality defects, missing items, or delivery grievances.

### 2. Structured JSON Order Extraction
* Parses unstructured conversational messages into a clean, machine-readable JSON schema containing:
  - `customer_name`
  - `items` (list of item name and quantity)
  - `delivery_urgency` (`standard` or `urgent`)
  - `delivery_location`
* Cleans code fences and markdown tags to ensure safe downstream system ingestion.

### 3. Stateful Multi-Turn Agent (`StationeryAgent`)
* LLMs are stateless by default. This agent maintains a conversation history buffer.
* **Sliding Window Memory:** Uses a configurable `max_turns` boundary to prevent token overflow while preserving recent context.
* **Error Resilience:** Gracefully catches `RateLimitError`, `APIConnectionError`, and authentication issues.

### 4. Student's 10% Verification Suite
* **Temperature Dial Test:** Compares deterministic generation (`T=0.2`) versus creative variability (`T=0.8`).
* **Hallucination Stress Test:** Ensures the agent strictly enforces catalogue boundaries when asked for unstocked goods (e.g. 3D printer filaments, monitors).
* **Memory Window Trimming:** Confirms older messages are trimmed once the window limit is exceeded.

---

## 📁 Folder Structure

```text
Day-2/
├── .env.example       # Sample environment template
├── Day-02.ipynb       # Interactive Colab notebook with interactive UI & widgets
├── task.py            # Standalone executable Python agent & test runner
└── README.md          # Documentation and implementation details
```

---

## ⚙️ Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or individually:
pip install openai langchain_groq python-dotenv ipywidgets
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (or use the existing key from Day 1):
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Run the Script
```bash
python task.py
```

Or open and run `Day-02.ipynb` in Google Colab / JupyterLab.
