"""
Day 3 Task: Composable LCEL Chains, Memory, Prompt Evaluation & Agent Sneak Peek
DriveReady OPC 2026 - Agentic AI Track
"""

import os
import sys
import csv
import time
import warnings
from dotenv import load_dotenv, find_dotenv

warnings.filterwarnings("ignore")

# 1. Environment & API Key Resolution
load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    try:
        from google.colab import userdata
        GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY") or userdata.get("GEMINI_API_KEY")
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    except Exception:
        pass

# 2. LLM Initialization (Google Gemini with optional Groq fallback)
def get_llm(temperature: float = 0.3):
    """Initializes LLM using Google Gemini (preferred) or Groq fallback."""
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=3,
            timeout=30
        )
    elif os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature
        )
    else:
        raise ValueError(
            "Neither GOOGLE_API_KEY nor GROQ_API_KEY found. "
            "Please configure your API key in .env or environment."
        )

# 3. Sweet Crumbs Bakery System Prompt & LCEL Chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

BAKERY_SYSTEM = """ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
FACTS:  Plum cake Rs 500. Chocolate cake Rs 650. Cupcake Rs 80 each.
        Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Never promise a delivery time or place. If it isn't in FACTS,
        say "I'll check with the owner." Work out totals carefully.
FORMAT: Under 50 words, friendly, English or Telugu matching the customer.
        Write prices as plain numbers without commas (e.g. Rs 1800)."""

def create_bakery_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", BAKERY_SYSTEM),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()

# 4. BrightPage Stationery Catalogue, System Prompts & LCEL Chains
STATIONERY_CATALOGUE = {
    "spiral notebook (200 pgs)": 85,
    "a4 copier paper ream (500 sheets)": 320,
    "gel pen pack (10 pens, blue/black)": 150,
    "pastel highlighter set (6 colors)": 180,
    "heavy-duty stapler + pins": 240,
    "sticky notes pack (400 sheets)": 95,
    "geometry box set": 160,
    "whiteboard marker set (4 colors)": 120,
    "permanent marker (black)": 35,
    "clear file folder (pack of 5)": 110,
}

CATALOGUE_TEXT = "\n".join(
    f"  - {item}: Rs {price}" for item, price in STATIONERY_CATALOGUE.items()
)

BRIGHTPAGE_SYSTEM_V1 = f"""ROLE:   You are the friendly assistant for BrightPage Stationery & Office Supplies.
FACTS:  Catalogue (INR):
{CATALOGUE_TEXT}
        Hours: Mon-Sat 9:00 AM - 8:30 PM, Sunday 10:00 AM - 4:00 PM.
        Delivery: Free local delivery on orders above Rs 500; Rs 40 fee below Rs 500.
        Bulk discount: 10% off on 10+ identical items or orders above Rs 1,000.
        Returns: Damaged items replaced within 3 days with receipt.
RULES:  Never promise delivery outside local city limits. If an item isn't in FACTS,
        say "I'll check with the store manager about availability and get back to you shortly."
        Work out totals carefully.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer."""

BRIGHTPAGE_SYSTEM_V2 = BRIGHTPAGE_SYSTEM_V1 + """
        Write prices without commas (e.g. Rs 1800), mention 'local city limits' when asked
        about out-of-city delivery, and answer only what the customer asked."""

def create_brightpage_chain(llm, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()

# 5. Automated CSV Test Suite Runner
def run_tests(path: str, test_chain, delay: float = 0.5):
    """Runs questions from a CSV file against a chain and verifies must_include strings."""
    if not os.path.exists(path):
        print(f"⚠️ Test file not found: {path}")
        return 0.0, []

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    passed = 0
    answers = []
    print(f"\nEvaluating test suite: {path} ({len(rows)} test cases)")
    print("-" * 65)

    for row in rows:
        answer = ""
        for attempt in range(4):
            try:
                answer = test_chain.invoke({"history": [], "question": row["question"]})
                break
            except Exception as err:
                if attempt == 3:
                    print(f"❌ Error invoking model: {err}")
                    answer = f"ERROR: {err}"
                    break
                wait_s = 2 ** (attempt + 1)
                time.sleep(wait_s)

        ok = row["must_include"].lower() in answer.lower()
        passed += int(ok)
        answers.append(answer)
        status_tag = "✅ PASS" if ok else "❌ FAIL"
        clean_ans = answer.replace("\n", " ")[:60]
        print(f"{status_tag} | {row['question']} -> {clean_ans}...")
        if delay:
            time.sleep(delay)

    rate = passed / len(rows) if rows else 0
    print("-" * 65)
    print(f"Result: {passed}/{len(rows)} passed ({rate:.0%})\n")
    return rate, answers

# 6. Main Runner
def main():
    print("=" * 70)
    print("🧩 Day 3: LCEL Chains, Memory, Automated Evaluation & Agent Sneak Peek")
    print("=" * 70)

    try:
        llm = get_llm(temperature=0.3)
        print("✅ LLM initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("Please check your .env file.")
        sys.exit(1)

    # Demo 1: Sweet Crumbs Bakery Chain with memory
    print("\n--- [Demo 1] Sweet Crumbs Bakery LCEL Chain with Memory ---")
    bakery_chain = create_bakery_chain(llm)
    history = []
    
    q1 = "Hi, I'm Ravi. Do you make eggless cakes?"
    a1 = bakery_chain.invoke({"history": history, "question": q1})
    history.extend([("human", q1), ("ai", a1)])
    print(f"User: {q1}\nBot:  {a1}\n")

    q2 = "What's my name?"
    a2 = bakery_chain.invoke({"history": history, "question": q2})
    history.extend([("human", q2), ("ai", a2)])
    print(f"User: {q2}\nBot:  {a2}\n")

    # Demo 2: Evaluation on tests.csv
    tests_csv_path = os.path.join(os.path.dirname(__file__), "tests.csv")
    if os.path.exists(tests_csv_path):
        print("--- [Demo 2] Automated Evaluation: Sweet Crumbs (tests.csv) ---")
        run_tests(tests_csv_path, bakery_chain)

    # Demo 3: BrightPage Stationery V1 vs V2 Evaluation
    bp_tests_path = os.path.join(os.path.dirname(__file__), "brightpage_tests.csv")
    if os.path.exists(bp_tests_path):
        print("--- [Demo 3] Automated Evaluation: BrightPage Stationery V1 vs V2 ---")
        bp_chain_v1 = create_brightpage_chain(llm, BRIGHTPAGE_SYSTEM_V1)
        v1_rate, _ = run_tests(bp_tests_path, bp_chain_v1)

        bp_chain_v2 = create_brightpage_chain(llm, BRIGHTPAGE_SYSTEM_V2)
        v2_rate, _ = run_tests(bp_tests_path, bp_chain_v2)
        print(f"Prompt Optimization Summary: V1: {v1_rate:.0%} -> V2: {v2_rate:.0%}")

    # Demo 4: Plain Code vs One LLM Call vs Agent Decision Matrix
    print("\n--- [Demo 4] Engineering Decision Matrix ---")
    decision_table = [
        (
            "Calculate cart subtotal, apply 10% bulk discount, and add Rs 40 delivery fee",
            "PLAIN CODE",
            "Deterministic arithmetic with fixed business rules requires zero LLM reasoning.",
        ),
        (
            "Classify an incoming WhatsApp message as order, inquiry, or complaint",
            "ONE LLM CALL",
            "Single-step natural language classification without requiring external tools.",
        ),
        (
            "Check live inventory, compute bulk pricing, and record a confirmed order",
            "AGENT",
            "Multi-step task where the model decides which external tools to call dynamically.",
        ),
    ]
    for task, tech, reason in decision_table:
        print(f"[{tech}] {task}\n   Rationale: {reason}\n")

    print("=" * 70)
    print("Day 3 execution completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
