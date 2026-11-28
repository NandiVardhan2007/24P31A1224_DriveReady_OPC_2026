# Day 6: Workflows, Planning & LangGraph

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NandiVardhan2007/24P31A1224_DriveReady_OPC_2026/blob/main/Day-6/task.ipynb)

Comprehensive implementation guide and technical documentation for **Day 6** of the Agentic AI Workshop — refactoring the **BrightPage Stationery & Office Supplies** assistant into structured **LangGraph** workflows with specialized agents and conditional routing.

---

## 🎯 Task Overview
Upgrade from a single monolithic agent to a deterministic **LangGraph workflow** with focused specialist sub-agents, structured state (`TypedDict`), conditional branching, and a post-order validation review node.

- **Primary Notebook:** [`task.ipynb`](./task.ipynb)
- **Standalone Script:** [`task.py`](./task.py)
- **Domain:** BrightPage Stationery & Office Supplies
- **Visualized Graph Diagrams:**
  - [`brightpage_router_graph.png`](./brightpage_router_graph.png) — 3-Intent Core Router Graph
  - [`brightpage_extended_graph.png`](./brightpage_extended_graph.png) — 4-Intent Extended Graph with `review_order` Node
- **Local Sheet Output:** [`brightpage_orders.csv`](./brightpage_orders.csv)
- **Primary Model:** `openai/gpt-oss-120b` / `llama-3.3-70b-versatile` via Groq (`ChatGroq`, `temperature=0.2`).

---

## 🧠 Core Engineering Concepts

### 1. Plain-Python Router (`handle`) vs. LangGraph
- **Plain-Python:** Classifies customer message intent via `llm.with_structured_output(Intent)` and routes using basic `if / elif / else` branches.
- **LangGraph StateGraph:** Converts procedural control flow into an observable, inspectable cyclic graph of nodes and conditional edges.

### 2. State & Focused Specialist Agents
Instead of giving one agent every tool (which increases risk of hallucinations or unauthorized database writes):
* **`faq_agent`:** Equipped with `[search_brightpage_faq, get_stationery_price, is_store_open, check_delivery_fee]` — strictly read-only, **cannot** log orders.
* **`order_agent`:** Equipped with `[get_stationery_price, check_delivery_fee, log_stationery_order]` — dedicated solely to completing orders.
* **`other` node:** Fixed polite greeting with zero extra LLM token calls.

```
       [START]
          │
      ▼
   ┌──────────┐
   │ classify │
   └────┬─────┘
        │ (conditional routing)
   ┌────┼──────────────┐
   ▼    ▼              ▼
┌─────┐┌─────┐   ┌───────────┐
│ faq ││order│   │   other   │
└──┬──┘└──┬──┘   └─────┬─────┘
   │      │            │
   └──────┴─────┬──────┘
                ▼
              [END]
```

### 3. Real-Time Execution Streaming (`stream_mode="updates"`)
By invoking `graph.stream({"message": msg}, stream_mode="updates")`, each node emits its discrete state update as it executes, providing complete observability into:
- The classified intent.
- The specialist agent invoked.
- The final response generated.

### 4. Extended 4-Branch Workflow & Order Review (`review_order`)
The extended graph handles four customer intents (`order`, `faq`, `complaint`, `other`) and includes a downstream validation node:
- **`complaint`:** Empathetically resolves defects and damaged items according to BrightPage's 3-day replacement policy.
- **`review_order`:** Inspects newly logged orders; flags bulk/high-value purchases (`qty >= 10`, `discount > 0`, or `total > 1000`) as `owner_review_required`, while auto-clearing standard purchases.

---

## 📁 Folder Structure

```text
Day-6/
├── .env.example                     # Environment variables template
├── brightpage_orders.csv            # Persistent stationery orders CSV mirror
├── brightpage_router_graph.png      # Rendered 3-intent core LangGraph diagram
├── brightpage_extended_graph.png    # Rendered 4-intent extended LangGraph diagram
├── task.ipynb                       # Interactive Google Colab notebook
├── task.py                          # Standalone executable test runner & graph runner
└── README.md                        # Technical documentation & architecture details
```

---

## ⚙️ Setup & Execution

### 1. Install Dependencies
```bash
pip install langchain langchain-groq langgraph chromadb sentence-transformers pydantic python-dotenv ipython
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
