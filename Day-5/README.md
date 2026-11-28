# Day 5: Thread Memory (`InMemorySaver` + `thread_id`) & RAG (`ChromaDB`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NandiVardhan2007/24P31A1224_DriveReady_OPC_2026/blob/main/Day-5/task.ipynb)

Comprehensive implementation guide and technical documentation for **Day 5** of the Agentic AI Workshop.

---

## 🎯 Task Overview
Upgrade the **BrightPage Stationery & Office Supplies** agent with:
- **Short-term thread memory** (`InMemorySaver` + `thread_id`) to isolate concurrent customer conversations.
- **Retrieval-Augmented Generation (RAG)** via `ChromaDB` + `all-MiniLM-L6-v2` sentence embeddings to answer policy and FAQ questions without hallucination.
- A standalone **PDF RAG Lab** that chunks and queries a 3-paragraph story PDF (`story_rag.pdf`).

- **Primary Notebook:** [`task.ipynb`](./task.ipynb)
- **Standalone Script:** [`task.py`](./task.py)
- **PDF RAG Document:** [`story_rag.pdf`](./story_rag.pdf) — *The Whispering Umbrella* (3 paragraphs, ~1,421 chars)
- **Sign-Off Test Suite:** [`tests.csv`](./tests.csv) — 18 rows (13 from Days 3–4 + 5 new RAG policy tests)
- **Local Sheet Output:** [`brightpage_orders.csv`](./brightpage_orders.csv)
- **Primary Model:** Groq `llama-3.3-70b-versatile` + `all-MiniLM-L6-v2` embeddings

---

## 🧠 Core Engineering Concepts

### 1. Three Kinds of Memory
| Type | Implementation | Scope |
| :--- | :--- | :--- |
| **Conversation Memory (Short-Term)** | `InMemorySaver` + `thread_id` | Lives in RAM within one process run |
| **Customer Memory (Long-Term)** | CSV / Google Sheets / Database | Persists across sessions |
| **Knowledge (RAG)** | ChromaDB vector store + semantic search | Retrieved on-demand via `search_brightpage_faq` tool |

### 2. `InMemorySaver` & `thread_id` for Isolated Threads
- Passing `checkpointer=InMemorySaver()` to `create_agent` activates LangGraph's checkpoint persistence.
- Each request is tagged with `config={"configurable": {"thread_id": "..."}}` so concurrent customer threads (e.g., `"ravi"`, `"sravani"`) are completely isolated.
- Thread `"ravi"` remembers Ravi's name and order context without re-asking; Thread `"sravani"` stays separate.

### 3. Embeddings, Vector Search & Policy Retrieval
- **16 BrightPage FAQs** indexed in ChromaDB using `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")`.
- `search_brightpage_faq` tool performs semantic nearest-neighbour retrieval (top-3), grounding all policy answers in real data.
- Paraphrased query test: `"If a notebook arrives torn, how many days do I have to swap it?"` → correctly retrieves the 3-day return policy.

### 4. Overlapping Text Chunking
- `chunk_text(text, chunk_size=280, overlap=50)` splits long documents into overlapping windows to avoid cutting mid-sentence.
- Used for both the BrightPage policy handbook demo and the PDF RAG Lab.

### 5. Multi-Paragraph PDF RAG Lab
- `pypdf.PdfReader` extracts text from `story_rag.pdf` (*The Whispering Umbrella*).
- Text chunked and indexed into a `story_pdf_rag` ChromaDB collection.
- `ask_pdf(question, top_k=2)` answers grounded questions and returns `"The answer is not available in the PDF."` for out-of-scope queries (e.g., `"What was the postman's pet dog named?"`).

### 6. Sign-Off Evaluation (18-Row Test Suite)
- 18 questions covering pricing, hours, delivery, return policy, UPI payment, cancellation, custom printing, and out-of-scope items.
- Each test runs in its own fresh `thread_id` to prevent memory bleed.
- Expected pass rate: **≥ 80%** (benchmark: 17/18 = 94%).

---

## 📁 Folder Structure

```text
Day-5/
├── .env.example         # Environment variables template
├── brightpage_orders.csv # Persistent stationery orders CSV mirror
├── story_rag.pdf        # The Whispering Umbrella — PDF RAG Lab document
├── tests.csv            # 18-row sign-off evaluation test suite
├── task.ipynb           # Interactive Google Colab notebook
├── task.py              # Standalone executable script
└── README.md            # Technical documentation
```

---

## ⚙️ Setup & Execution

### 1. Install Dependencies
```bash
pip install langchain langchain-groq langgraph chromadb sentence-transformers pypdf pydantic python-dotenv
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and add your API key:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Run the Script
```bash
python task.py
```

Or open and run [`task.ipynb`](./task.ipynb) in Google Colab.
