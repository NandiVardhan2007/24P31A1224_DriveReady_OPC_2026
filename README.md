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
└── README.md
```

---

## 🗓️ Daily Tasks Summary

| Day | Topic / Project | Core Concepts | Files |
| :--- | :--- | :--- | :--- |
| **Day 1** | Sweet Crumbs Bakery & Prompt Guard | System Prompt Engineering, Groq API, LangChain ChatGroq, Llama Prompt Guard | [`Day-1/`](./Day-1/) |
| **Day 2** | BrightPage Stationery AI Agent | Few-Shot Intent Classification, Structured JSON Extraction, Multi-Turn Memory Buffer, Error Resilience | [`Day-2/`](./Day-2/) |
| **Day 3** | *Upcoming* | Tool Calling & Autonomous Agent Loop | `Day-3/` |

---

## 🛠️ General Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NandiVardhan2007/24P31A1224_DriveReady_OPC_2026.git
   cd 24P31A1224_DriveReady_OPC_2026
   ```

2. **Install required dependencies:**
   ```bash
   pip install openai langchain_groq groq python-dotenv ipywidgets
   ```

3. **Configure API Key:**
   - In each day's directory, copy `.env.example` to `.env` and insert your `GROQ_API_KEY`.
