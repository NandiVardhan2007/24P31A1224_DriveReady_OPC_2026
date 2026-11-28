"""
Day 5 Task: Thread Memory (InMemorySaver + thread_id) & RAG (ChromaDB)
DriveReady OPC 2026 - Agentic AI Track
"""

import os
import sys
import csv
import time
import logging
import warnings
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
# 1. Environment & LLM Initialization
# ==============================================================================
load_dotenv(find_dotenv())

def get_llm(temperature: float = 0.2):
    """Initialize LLM using Groq (preferred) or Google Gemini as fallback."""
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model_name, temperature=temperature, max_retries=3, timeout=30)
    elif os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=3, timeout=30)
    else:
        raise ValueError("Neither GROQ_API_KEY nor GOOGLE_API_KEY found. Configure your .env file.")

def invoke_retry(runnable, payload, config=None, retries: int = 4):
    """Invoke a LangChain runnable with exponential backoff on transient errors."""
    for attempt in range(retries):
        try:
            return runnable.invoke(payload, config) if config else runnable.invoke(payload)
        except (KeyError, ValueError, TypeError):
            raise
        except Exception as err:
            if attempt == retries - 1:
                raise
            wait_s = 2 ** (attempt + 1)
            time.sleep(wait_s)

# ==============================================================================
# 2. Local CSV Worksheet Mirror (simulates Google Sheets / gspread)
# ==============================================================================
class LocalWorksheet:
    """Local CSV-backed worksheet mirror with append_row and get_all_records interface."""
    def __init__(self, csv_path: str, headers: list):
        self.csv_path = csv_path
        self.headers = headers
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.headers)

    def append_row(self, row: list):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def get_all_records(self) -> list:
        if not os.path.exists(self.csv_path):
            return []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

# ==============================================================================
# 3. ChromaDB Knowledge Base: BrightPage FAQs & Policies
# ==============================================================================
FAQS = {
    "delivery":       "We deliver locally within city limits. Local home delivery is free for orders above Rs 500, and Rs 40 for orders of Rs 500 or below.",
    "out_of_city":    "We only offer delivery within local city limits and do not ship outside the city.",
    "payment":        "We accept UPI, cash on delivery, and card payments. Orders above Rs 1000 require a 50% advance by UPI.",
    "cancellation":   "Order cancellation is free up to 12 hours before the scheduled delivery or pickup time; after that, the advance is not refunded.",
    "bulk_discount":  "Customers get a 10% bulk discount automatically when buying 10 or more identical items or when order subtotal exceeds Rs 1000.",
    "returns":        "Damaged or defective stationery items are replaced free of charge within 3 days with the original receipt.",
    "custom_print":   "Custom notebook cover printing and school branding orders require 48 hours' notice and a minimum of 20 units.",
    "gift_wrapping":  "Gift wrapping is free on request for geometry box sets and highlighter gift packs.",
    "hours":          "Store opening hours are Monday to Saturday 9:00 AM to 8:30 PM, and Sunday 10:00 AM to 4:00 PM.",
    "pickup":         "Store pickup is always free during opening hours with zero delivery fee.",
    "same_day":       "Orders placed before 3:00 PM are delivered the same evening; orders after 3:00 PM are delivered the next morning.",
    "gst_invoice":    "GST tax invoices are provided for all office and school bulk orders on request.",
    "school_kits":    "Custom back-to-school stationery kits can be packed together when ordered 24 hours ahead.",
    "paper_quality":  "Our A4 copier paper reams are 75 GSM bright white sheets suitable for laser and inkjet printers.",
    "pen_refills":    "Gel pens and permanent markers are sold in sealed packs; individual loose refills are not stocked.",
    "unlisted_items": "If a customer asks for an unlisted product like laptops, electronics, or textbooks, say you will check with the store manager.",
}

def build_chroma_faq():
    """Build in-memory ChromaDB with BrightPage FAQ embeddings."""
    import chromadb
    from chromadb.utils import embedding_functions
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception:
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="brightpage_faq", embedding_function=embedding_fn)
    collection.upsert(ids=list(FAQS.keys()), documents=list(FAQS.values()))
    return collection

# ==============================================================================
# 4. Text Chunking Helper
# ==============================================================================
def chunk_text(text: str, chunk_size: int = 280, overlap: int = 50) -> list:
    """Split a long text into overlapping chunks of roughly chunk_size characters."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    pieces, start = [], 0
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            pieces.append(piece)
        start += chunk_size - overlap
    return pieces

# ==============================================================================
# 5. Domain Data: Stationery Catalogue & Hours
# ==============================================================================
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

BRIGHTPAGE_HOURS = {
    "monday": "9:00 AM to 8:30 PM", "tuesday": "9:00 AM to 8:30 PM",
    "wednesday": "9:00 AM to 8:30 PM", "thursday": "9:00 AM to 8:30 PM",
    "friday": "9:00 AM to 8:30 PM", "saturday": "9:00 AM to 8:30 PM",
    "sunday": "10:00 AM to 4:00 PM",
}

def _resolve_item(query: str):
    cleaned = query.lower().strip()
    if cleaned in STATIONERY_CATALOGUE:
        return cleaned, STATIONERY_CATALOGUE[cleaned]
    for cat_item, price in STATIONERY_CATALOGUE.items():
        if cleaned in cat_item or cat_item in cleaned:
            return cat_item, price
    return None, None

# ==============================================================================
# 6. Pydantic Schema & @tool Definitions
# ==============================================================================
StationeryItemName = Literal[
    "spiral notebook (200 pgs)", "a4 copier paper ream (500 sheets)", "gel pen pack (10 pens, blue/black)",
    "pastel highlighter set (6 colors)", "heavy-duty stapler + pins", "sticky notes pack (400 sheets)",
    "geometry box set", "whiteboard marker set (4 colors)", "permanent marker (black)", "clear file folder (pack of 5)"
]

class StationeryOrder(BaseModel):
    customer_name: str = Field(description="Customer's name")
    item: StationeryItemName = Field(description="Stationery item from catalogue")
    qty: int = Field(ge=1, description="Quantity (must be >= 1)")
    delivery_type: Literal["home_delivery", "store_pickup"] = Field(description="Delivery method")
    delivery_date: str = Field(description="Delivery or pickup date")
    notes: Optional[str] = Field(default=None, description="Optional special instructions")


def build_tools(ws, faq_collection):
    """Build all @tool functions bound to the given worksheet and FAQ collection."""

    @tool
    def get_stationery_price(item: str, qty: int = 1) -> str:
        """Calculate total price in rupees for a BrightPage Stationery item with auto-applied bulk discount.
        Applies 10% bulk discount when qty >= 10 or subtotal > Rs 1000.

        Args:
            item: Stationery catalogue item name
            qty: Quantity requested (must be at least 1)
        """
        if qty < 1:
            return f"Quantity must be at least 1 (got {qty}). Ask the customer for a valid quantity."
        matched, unit_price = _resolve_item(item)
        if matched is None:
            return f"'{item}' is not in the BrightPage catalogue. I'll check with the store manager."
        subtotal = unit_price * qty
        discount = round(subtotal * 0.10, 2) if (qty >= 10 or subtotal > 1000) else 0.0
        final_total = subtotal - discount
        if discount > 0:
            return f"{qty} x {matched} @ Rs {unit_price} = Rs {subtotal}, 10% bulk discount (-Rs {discount:.0f}) -> Rs {final_total:.0f}"
        return f"{qty} x {matched} @ Rs {unit_price} = Rs {final_total:.0f}"

    @tool
    def is_store_open(day: str) -> str:
        """Check whether BrightPage Stationery is open on a given day and return its hours.

        Args:
            day: Day of the week (e.g. 'sunday')
        """
        hours = BRIGHTPAGE_HOURS.get(day.lower().strip())
        if not hours:
            return f"'{day}' is not a valid day of the week."
        return f"Open {hours} on {day}."

    @tool
    def check_delivery_fee(order_total: float) -> str:
        """Return the local delivery fee for the given order total in rupees.
        Delivery is free above Rs 500; Rs 40 otherwise.

        Args:
            order_total: Order amount in rupees after discounts
        """
        fee = 0 if order_total > 500 else 40
        if fee == 0:
            return "Free local delivery (order above Rs 500)."
        return "Local delivery fee is Rs 40 (order below Rs 500)."

    @tool
    def search_brightpage_faq(question: str) -> str:
        """Search BrightPage Stationery policies and FAQs using semantic vector search.
        Use for all policy, payment, UPI, cancellation, return, printing, gift-wrap, or shipping questions.

        Args:
            question: Customer's policy or FAQ question
        """
        results = faq_collection.query(query_texts=[question], n_results=3)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant policy found. I'll check with the store manager."
        return "\n".join(docs)

    @tool
    def log_stationery_order(customer_name: str, item: str, qty: int, delivery_type: str,
                              delivery_date: str, notes: str = "") -> str:
        """Save a confirmed BrightPage Stationery order to the orders sheet.
        Only call this once customer_name, item, qty >= 1, delivery_type, and delivery_date are confirmed.

        Args:
            customer_name: Customer name
            item: Catalogue item name
            qty: Quantity (at least 1)
            delivery_type: 'home_delivery' or 'store_pickup'
            delivery_date: Delivery or pickup date
            notes: Optional special instructions
        """
        matched, unit_price = _resolve_item(item)
        if matched is None:
            return f"Validation error: '{item}' not found in catalogue."
        try:
            record = StationeryOrder(
                customer_name=customer_name, item=matched, qty=qty,
                delivery_type=delivery_type, delivery_date=delivery_date,
                notes=notes or None
            )
        except Exception as err:
            return f"Validation error: {err}."

        subtotal = unit_price * record.qty
        discount = round(subtotal * 0.10, 2) if (record.qty >= 10 or subtotal > 1000) else 0.0
        discounted = subtotal - discount
        delivery_fee = 0 if (record.delivery_type == "store_pickup" or discounted > 500) else 40
        total = discounted + delivery_fee

        ws.append_row([
            record.customer_name, record.item, record.qty, record.delivery_type,
            record.delivery_date, unit_price, discount, delivery_fee, total, record.notes or ""
        ])
        return (
            f"Order saved for {record.customer_name}: {record.qty} x {record.item} "
            f"({record.delivery_type} on {record.delivery_date}). "
            f"Discount: Rs {discount:.0f}, Delivery: Rs {delivery_fee}, Total: Rs {total:.0f}."
        )

    return [get_stationery_price, is_store_open, check_delivery_fee, search_brightpage_faq, log_stationery_order]

# ==============================================================================
# 7. PDF RAG Helper
# ==============================================================================
def build_pdf_rag(pdf_path: str):
    """Extract text from a PDF, chunk it, index in ChromaDB, and return a query function."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None, "pypdf not installed. Run: pip install pypdf"

    import chromadb
    from chromadb.utils import embedding_functions
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception:
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    reader = PdfReader(pdf_path)
    raw_pdf = "\n".join(page.extract_text().strip() for page in reader.pages if page.extract_text())
    lines = [ln.strip() for ln in raw_pdf.splitlines() if ln.strip()]
    joined = " ".join(lines[1:]).replace("thank- you", "thank-you")
    chunks = chunk_text(joined, chunk_size=500, overlap=60)
    if not chunks:
        return None, "No text extracted from PDF."

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="story_pdf_rag", embedding_function=embedding_fn)
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks
    )

    def ask_pdf(question: str, top_k: int = 2) -> str:
        results = collection.query(query_texts=[question], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "The answer is not available in the PDF."
        context = "\n".join(docs)
        return f"Based on the PDF:\n{context}"

    return ask_pdf, f"Indexed {len(chunks)} chunks from {os.path.basename(pdf_path)}"

# ==============================================================================
# 8. Main Execution & Demos
# ==============================================================================
BRIGHTPAGE_SYSTEM = """ROLE:   You are the friendly assistant for BrightPage Stationery & Office Supplies.
RULES:  ALWAYS call get_stationery_price FIRST whenever a customer asks about or tries to order an item; NEVER guess prices.
        For any policy, payment, UPI, cancellation, return, printing, gift wrap, or shipping rule,
        ALWAYS use search_brightpage_faq and answer ONLY from what it returns.
        If the answer or item isn't in the catalogue or FAQs (e.g. laptops, electronics),
        NEVER invent details — say "I'll check with the store manager about availability."
        Match customer wording to catalogue items directly (e.g. "spiral notebooks" -> "spiral notebook (200 pgs)").
        Before log_stationery_order, confirm: customer_name, item, qty >= 1, delivery_type, delivery_date; ask if any missing.
        After saving, read back the full order summary and total.
FORMAT: Under 65 words, friendly, plain numbers without commas."""

def run_tests_csv(tests_path: str, agent, config_prefix: str = "eval"):
    """Run the sign-off CSV evaluation suite with one thread_id per question."""
    with open(tests_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    passed = 0
    print(f"\nRunning {len(rows)} evaluation tests from {tests_path}:")
    print("-" * 65)
    for idx, row in enumerate(rows):
        config = {"configurable": {"thread_id": f"{config_prefix}_{idx}"}}
        try:
            res = invoke_retry(agent, {"messages": [{"role": "user", "content": row["question"]}]}, config=config)
            answer = res["messages"][-1].content
        except Exception as e:
            answer = f"ERROR: {e}"
        expected = row["must_include"].lower()
        answer_norm = " ".join(answer.replace("\u202f", " ").replace("\xa0", " ").split()).lower()
        ok = expected in answer_norm
        passed += int(ok)
        print(f"{'✅ PASS' if ok else '❌ FAIL'} | {row['question']} -> {answer[:60]}...")
        time.sleep(0.3)
    rate = passed / len(rows) if rows else 0
    print("-" * 65)
    print(f"Sign-Off Result: {passed}/{len(rows)} passed ({rate:.0%})")
    return rate

def main():
    print("=" * 70)
    print("🧠 Day 5: Thread Memory (InMemorySaver) & RAG (ChromaDB)")
    print("=" * 70)

    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("❌ No API key found. Set GROQ_API_KEY or GOOGLE_API_KEY in .env")
        sys.exit(1)

    try:
        llm = get_llm(temperature=0.2)
        print("✅ LLM initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        sys.exit(1)

    # --- Setup Worksheet & ChromaDB ---
    brightpage_csv = os.path.join(os.path.dirname(__file__), "brightpage_orders.csv")
    ws = LocalWorksheet(brightpage_csv, headers=[
        "customer_name", "item", "qty", "delivery_type", "delivery_date",
        "unit_price", "discount", "delivery_fee", "total", "notes"
    ])
    print("✅ Local order sheet initialized.")

    try:
        faq_collection = build_chroma_faq()
        print(f"✅ ChromaDB FAQ store ready: {faq_collection.count()} entries.")
    except Exception as e:
        print(f"⚠️  ChromaDB init failed: {e}. Install: pip install chromadb sentence-transformers")
        sys.exit(1)

    tools = build_tools(ws, faq_collection)

    # --- Demo chunking ---
    sample = " ".join(list(FAQS.values())[:3])
    sample_chunks = chunk_text(sample, chunk_size=220, overlap=40)
    print(f"\n--- Chunking Demo: {len(sample)} chars -> {len(sample_chunks)} overlapping chunks ---")
    for i, c in enumerate(sample_chunks, 1):
        print(f"  Chunk {i}: {c[:80]}...")

    # --- Thread Memory Agent with InMemorySaver ---
    memory = InMemorySaver()
    agent = create_agent(model=llm, tools=tools, system_prompt=BRIGHTPAGE_SYSTEM, checkpointer=memory)

    print("\n--- [Demo 1] Thread Memory: Ravi's multi-turn order (thread='ravi') ---")
    ravi_turns = [
        "Hi, I'm Ravi. How much are 2 A4 copier paper reams?",
        "Great! Please book that for home delivery on 29 Nov.",
    ]
    for turn in ravi_turns:
        config = {"configurable": {"thread_id": "ravi"}}
        res = invoke_retry(agent, {"messages": [{"role": "user", "content": turn}]}, config=config)
        print(f"Ravi: {turn}\nAgent: {res['messages'][-1].content}\n")

    print("--- [Demo 2] Separate Thread: Sravani (thread='sravani') ---")
    sravani_q = "Do you offer gift wrapping for highlighters?"
    config = {"configurable": {"thread_id": "sravani"}}
    res = invoke_retry(agent, {"messages": [{"role": "user", "content": sravani_q}]}, config=config)
    print(f"Sravani: {sravani_q}\nAgent: {res['messages'][-1].content}\n")

    print("--- [Demo 3] Policy RAG - Paraphrased Return Policy ---")
    policy_q = "If a notebook arrives torn, how many days do I have to swap it?"
    config = {"configurable": {"thread_id": "policy_test"}}
    res = invoke_retry(agent, {"messages": [{"role": "user", "content": policy_q}]}, config=config)
    print(f"Customer: {policy_q}\nAgent: {res['messages'][-1].content}\n")

    print("--- [Demo 4] Break-It Test: Out-of-Scope (Laptops) ---")
    break_q = "Do you sell laptops or wireless keyboards?"
    config = {"configurable": {"thread_id": "break_test"}}
    res = invoke_retry(agent, {"messages": [{"role": "user", "content": break_q}]}, config=config)
    print(f"Customer: {break_q}\nAgent: {res['messages'][-1].content}\n")

    # --- Sign-Off Evaluation ---
    tests_path = os.path.join(os.path.dirname(__file__), "tests.csv")
    if os.path.exists(tests_path):
        print("--- [Demo 5] Sign-Off Evaluation Suite (tests.csv) ---")
        run_tests_csv(tests_path, agent)

    # --- PDF RAG Lab ---
    pdf_path = os.path.join(os.path.dirname(__file__), "story_rag.pdf")
    if os.path.exists(pdf_path):
        print("\n--- [Demo 6] PDF RAG Lab: story_rag.pdf (The Whispering Umbrella) ---")
        ask_pdf, msg = build_pdf_rag(pdf_path)
        print(f"Status: {msg}")
        if ask_pdf:
            pdf_questions = [
                "Who did Meera run the stationery and bookshop with?",
                "Why did Meera keep a bright yellow umbrella outside every morning?",
                "What was the postman's pet dog named?",
            ]
            for q in pdf_questions:
                ans = ask_pdf(q, top_k=2)
                print(f"Q: {q}\nA: {ans[:200]}\n")

    print("=" * 70)
    print("Day 5 execution completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
