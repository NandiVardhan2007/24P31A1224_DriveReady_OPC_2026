# Day 1: Introduction to Prompt Engineering, Groq API & Guardrails

## 📌 Task Overview
Day 1 introduces the fundamentals of building LLM-powered applications using the **Groq API** and **LangChain**. The task focuses on creating a specialized AI customer assistant for **Sweet Crumbs Home Bakery** with strict behavioral constraints and testing prompt safety using **Llama Prompt Guard**.

---

## 🏗️ Key Components & Implementation

### 1. System Prompt Engineering (The 4 Pillars)
A well-structured system prompt sets boundaries to avoid hallucinations and maintain business rules:
* **Role:** Customer assistant for Sweet Crumbs Home Bakery.
* **Facts:** All items are baked to order; orders require 24 hours' advance notice; all cakes have eggless variations.
* **Rules:** Never promise exact delivery times; politely admit when an answer is unknown.
* **Format:** Polite response strictly under 60 words.

### 2. Groq LLM Integration (`langchain_groq`)
* Connected to Groq using `ChatGroq` (`openai/gpt-oss-120b` or standard Llama 3 models) with `temperature=0` for consistent, deterministic responses.

### 3. Prompt Safety & Guardrails (`meta-llama/llama-prompt-guard-2-22m`)
* Implemented safety evaluation using Groq's dedicated Prompt Guard model.
* Detects prompt injection attempts (e.g. *"Ignore your previous instructions..."*) and classifies input safety based on confidence threshold scores (`score >= 0.5` flags unsafe input).

---

## 📁 Folder Structure

```text
Day-1/
├── .env.example       # Sample environment template
├── Day-01.ipynb       # Interactive Google Colab / Jupyter notebook
├── task.py            # Standalone executable Python script
└── README.md          # Documentation and implementation details
```

---

## ⚙️ Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or individually:
pip install langchain_groq groq python-dotenv
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Groq API key:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Run the Script
```bash
python task.py
```

Or open and run `Day-01.ipynb` in Google Colab or JupyterLab.
