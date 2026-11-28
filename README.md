# 24P31A1224 - DriveReady OPC 2026: Agentic AI Track

Repository tracking daily tasks, source code, and documentation for the **DriveReady OPC 2026** program.

---

## 📂 Repository Structure

```text
24P31A1224_DriveReady_OPC_2026/
├── Day-1/
│   ├── .env.example
│   ├── Day-01.ipynb
│   ├── task.py
│   └── README.md
├── Day-2/
│   ├── .env.example
│   ├── Day-02.ipynb
│   ├── task.py
│   └── README.md
├── Day-3/
│   ├── .env.example
│   ├── brightpage_tests.csv
│   ├── tests.csv
│   ├── task.ipynb
│   ├── task.py
│   └── README.md
├── Day-4/
│   ├── .env.example
│   ├── brightpage_orders.csv
│   ├── sweet_crumbs_orders.csv
│   ├── task.ipynb
│   ├── task.py
│   └── README.md
├── Day-5/
│   ├── .env.example
│   ├── brightpage_orders.csv
│   ├── story_rag.pdf
│   ├── tests.csv
│   ├── task.ipynb
│   ├── task.py
│   └── README.md
├── Day-6/
│   ├── .env.example
│   ├── brightpage_orders.csv
│   ├── brightpage_router_graph.png
│   ├── brightpage_extended_graph.png
│   ├── task.ipynb
│   ├── task.py
│   └── README.md
└── README.md
```

---

## 🗓️ Daily Tasks Summary

| Day | Topic / Project | Core Concepts | Files |
| :--- | :--- | :--- | :--- |
| **Day 1** | Sweet Crumbs Bakery & Prompt Guard | System Prompt Engineering, Groq API, LangChain ChatGroq, Llama Prompt Guard | [`Day-1/`](./Day-1/) |
| **Day 2** | BrightPage Stationery AI Agent | Few-Shot Intent Classification, Structured JSON Extraction, Multi-Turn Memory Buffer, Error Resilience | [`Day-2/`](./Day-2/) |
| **Day 3** | Composable LCEL Chains & Evaluation | LCEL (`prompt \| llm \| parser`), `MessagesPlaceholder`, Automated CSV Evaluation, `create_agent` Sneak Peek | [`Day-3/`](./Day-3/) |
| **Day 4** | Tool-Calling Agents & Persistent Sheets | `@tool` Decorators, Pydantic Schemas, Agent Message Traces, Local CSV Order Logging | [`Day-4/`](./Day-4/) |
| **Day 5** | Thread Memory & RAG | `InMemorySaver` + `thread_id`, ChromaDB Vector Store, Semantic FAQ Retrieval, PDF RAG Lab | [`Day-5/`](./Day-5/) |
| **Day 6** | Workflows, Planning & LangGraph | `StateGraph`, Focused Specialist Agents (`faq_agent`, `order_agent`), Intent Routing, `review_order` Node | [`Day-6/`](./Day-6/) |

---

## 🛠️ General Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NandiVardhan2007/24P31A1224_DriveReady_OPC_2026.git
   cd 24P31A1224_DriveReady_OPC_2026
   ```

2. **Install required dependencies:**
   ```bash
   pip install openai langchain langchain-core langchain-google-genai langchain-groq groq \
               langgraph chromadb sentence-transformers pypdf \
               pydantic python-dotenv ipywidgets gspread ipython
   ```

3. **Configure API Keys:**
   - In each day's directory, copy `.env.example` to `.env` and insert your API keys (`GROQ_API_KEY` and/or `GOOGLE_API_KEY`).
