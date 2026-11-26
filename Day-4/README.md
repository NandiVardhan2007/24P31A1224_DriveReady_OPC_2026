# Day 4: Tools, Structured Records, Sheets & Agent Traces

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NandiVardhan2007/24P31A1224_DriveReady_OPC_2026/blob/main/Day-4/task.ipynb)

Comprehensive implementation guide and technical documentation for **Day 4** of the Agentic AI Workshop.

---

## 🎯 Task Overview
Upgrade from tool-less prompt chains to **tool-calling autonomous agents** using LangChain's `create_agent`, `@tool` decorators, Pydantic data schemas (`Order`, `StationeryOrder`), and persistent order logging with a CSV-backed worksheet (`sweet_crumbs_orders.csv` and `brightpage_orders.csv`) mirroring Google Sheets (`gspread`).

- **Primary Notebook:** [`task.ipynb`](./task.ipynb)
- **Standalone Script:** [`task.py`](./task.py)
- **Domains Covered:**
  - **Sweet Crumbs Home Bakery:** Core workshop agent with customer edge-case testing and `tool1` break-it experiment.
  - **BrightPage Stationery & Office Supplies:** Homework extension featuring `@tool` pricing with bulk discounts, store hours / delivery fee calculations, and persistent order logging.
- **Local Sheet Outputs:** [`sweet_crumbs_orders.csv`](./sweet_crumbs_orders.csv), [`brightpage_orders.csv`](./brightpage_orders.csv)
- **Primary Model:** `llama-3.3-70b-versatile` / `openai/gpt-oss-20b` via Groq (`ChatGroq`, `temperature=0.2`).

---

## 🧠 Core Engineering Concepts

### 1. `@tool` Decorators & Why Docstrings Are the Model's API Docs
- Tools expose deterministic Python functions to the LLM with typed parameters and detailed Google-style docstrings.
- **The Break-It Experiment:**
  - Renaming `get_price` to `tool1` with vague docstring `"does stuff"` caused the agent to make **0 tool calls** on `"How much are 3 plum cakes?"`.
  - Restoring the explicit name `get_price` and a descriptive docstring restored tool selection (`['get_price']`).

### 2. Pydantic Schemas & Data Validation
- **`Order` Schema:** Validates `name`, `item` (`Literal["plum cake", "chocolate cake", "cupcake"]`), `qty >= 1`, `date`, and `eggless` preference.
- **`StationeryOrder` Schema:** Validates `customer_name`, `item` (10-item `Literal`), `qty >= 1`, `delivery_type` (`"home_delivery"` | `"store_pickup"`), `delivery_date`, and optional `notes`.

### 3. Agent Execution Traces with `pretty_print()`
Inspecting message sequences demonstrates the reasoning loop:
1. `HumanMessage` -> User asks: `"Are you open on Sunday? How much are 2 chocolate cakes?"`
2. `AIMessage` -> LLM decides to call tools: `is_open(day="sunday")` and `get_price(item="chocolate cake", qty=2)`
3. `ToolMessage` -> Returns factual results from Python code execution.
4. `AIMessage` -> LLM synthesizes tool outputs into a final customer-ready response.

### 4. Customer Edge-Case Testing & Sheet Integrity
Adversarial and incomplete inputs are handled gracefully without corrupting database records:
1. **Unlisted item (`"donut"`):** Explains that donuts are not on the menu and lists available options.
2. **Missing pickup date:** Prompts the customer for missing details without calling `log_order`.
3. **Zero quantity (`0 cupcakes`):** Informs the customer that minimum quantity is at least 1.
- **Sheet Integrity:** Verified that 0 invalid rows were written to the order records during invalid tests.

---

## 📁 Folder Structure

```text
Day-4/
├── .env.example              # Environment variables template
├── sweet_crumbs_orders.csv   # Persistent bakery orders CSV mirror
├── brightpage_orders.csv     # Persistent stationery orders CSV mirror
├── task.ipynb                # Interactive Google Colab notebook
├── task.py                   # Standalone executable test runner & agent
└── README.md                 # Technical documentation & implementation details
```

---

## ⚙️ Setup & Execution

### 1. Install Dependencies
```bash
pip install langchain langchain-groq pydantic python-dotenv gspread
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API key:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Run the Script
```bash
python task.py
```

Or open and execute [`task.ipynb`](./task.ipynb) in Google Colab or JupyterLab.
