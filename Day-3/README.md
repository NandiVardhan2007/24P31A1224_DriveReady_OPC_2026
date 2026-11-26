# Day 3: Composable LCEL Chains, Memory, Prompt Evaluation & Agent Sneak Peek

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NandiVardhan2007/24P31A1224_DriveReady_OPC_2026/blob/main/Day-3/task.ipynb)

Comprehensive implementation and technical documentation for **Day 3** of the Agentic AI Workshop.

---

## 🎯 Task Overview
Refactor raw API interactions into composable **LangChain Expression Language (LCEL)** chains using `ChatPromptTemplate`, `MessagesPlaceholder`, and `StrOutputParser`. Systematically evaluate prompt robustness and accuracy against structured CSV test suites, analyze temperature determinism, and preview autonomous tool-using agents (`create_agent`).

- **Primary Notebook:** [`task.ipynb`](./task.ipynb)
- **Standalone Script:** [`task.py`](./task.py)
- **Evaluation Test Suites:**
  - [`tests.csv`](./tests.csv) — Sweet Crumbs Home Bakery (10 validation cases)
  - [`brightpage_tests.csv`](./brightpage_tests.csv) — BrightPage Stationery & Office Supplies (10 validation cases)
- **Supported Models:** Google Gemini (`gemini-1.5-flash` / `gemini-3.5-flash-lite`) via `langchain-google-genai` and Groq (`llama-3.3-70b-versatile`) via `langchain-groq`.

---

## 🧠 Core Engineering Concepts

### 1. Composable LCEL Chains (`prompt | llm | StrOutputParser()`)
Instead of manually composing and parsing dictionary messages:
* **`ChatPromptTemplate.from_messages(...)`**: Structures system prompts, conversational state, and user input.
* **`MessagesPlaceholder("history")`**: Injects multi-turn chat memory directly into the message chain.
* **`StrOutputParser()`**: Extracts clean text string content from the model's `AIMessage` output.

### 2. Automated Prompt Evaluation Test Harness
Manual ad-hoc testing misses regressions. By running automated test suites (`tests.csv` and `brightpage_tests.csv`), each customer query is evaluated against expected `must_include` ground truth substrings to calculate an exact pass rate (`passed / total`).

### 3. Prompt Optimization Iteration (V1 ➔ V2)
* **V1 (Baseline):** Standard system prompt outlining inventory and store policies.
* **V2 (Optimized):** Constrains numerical formats (e.g., removing commas: `Rs 1800`), explicitly establishes boundary rules (e.g., specifying `local city limits`), and enforces strict conciseness.

### 4. Temperature & Output Stability (`0.3` vs `1.0`)
* **`temperature=0.3`**: Produces consistent, deterministic outputs essential for factual business inquiries, prices, and refund rules.
* **`temperature=1.0`**: Increases creative variation, which introduces wording drift and inconsistency across test runs.

### 5. Architectural Decision Framework

| Business Task (BrightPage Stationery) | Selected Approach | Architectural Rationale |
| :--- | :--- | :--- |
| Calculate cart subtotal, 10% discount, & delivery fee | **Plain Code** | Deterministic arithmetic — exact, zero latency, zero token cost. |
| Classify message as `order`, `inquiry`, or `complaint` | **One LLM Call** | Single-step natural language understanding without requiring tools. |
| Check inventory, compute bulk pricing, and log order | **Agent** | Multi-step reasoning where the model dynamically decides tool calls. |

---

## 📁 Folder Structure

```text
Day-3/
├── .env.example            # Environment variables template (Gemini / Groq)
├── brightpage_tests.csv    # 10 test cases for BrightPage Stationery
├── tests.csv               # 10 test cases for Sweet Crumbs Bakery
├── task.ipynb              # Interactive Google Colab / Jupyter notebook
├── task.py                 # Standalone executable test runner & agent preview
└── README.md               # Technical documentation & implementation details
```

---

## ⚙️ Setup & Execution

### 1. Install Dependencies
```bash
pip install langchain langchain-core langchain-google-genai langchain-groq python-dotenv
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
# or
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Verification Script
```bash
python task.py
```

Or open and run [`task.ipynb`](./task.ipynb) in Google Colab.
