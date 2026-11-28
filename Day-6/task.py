"""
Day 6 Task: Workflows, Planning & LangGraph
DriveReady OPC 2026 - Agentic AI Track
"""

import os
import sys
import csv
import time
import logging
import warnings
from typing import Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END

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
    """Initialize LLM using Groq (preferred) or Google Gemini fallback."""
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
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
# 2. Local CSV Worksheet Mirror
# ==============================================================================
class LocalWorksheet:
    """Local CSV-backed worksheet mirror simulating Google Sheets."""
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
# 3. ChromaDB Knowledge Base: BrightPage FAQs
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
    """Build in-memory ChromaDB collection with BrightPage FAQ embeddings."""
    import chromadb
    from chromadb.utils import embedding_functions
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception:
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="brightpage_faq_d6", embedding_function=embedding_fn)
    collection.upsert(ids=list(FAQS.keys()), documents=list(FAQS.values()))
    return collection

# ==============================================================================
# 4. Domain Data & @tool Definitions
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

class StationeryOrder(BaseModel):
    customer_name: str = Field(description="Customer's name")
    item: Literal[
        "spiral notebook (200 pgs)",
        "a4 copier paper ream (500 sheets)",
        "gel pen pack (10 pens, blue/black)",
        "pastel highlighter set (6 colors)",
        "heavy-duty stapler + pins",
        "sticky notes pack (400 sheets)",
        "geometry box set",
        "whiteboard marker set (4 colors)",
        "permanent marker (black)",
        "clear file folder (pack of 5)",
    ] = Field(description="Closest matching item from the BrightPage Stationery catalogue")
    qty: int = Field(description="Quantity ordered", ge=1)
    delivery_type: Literal["home_delivery", "store_pickup"] = Field(
        description="Whether customer wants home_delivery or store_pickup"
    )
    delivery_date: str = Field(description="Requested delivery or pickup date")
    notes: Optional[str] = Field(default=None, description="Optional delivery or packing notes")

def _resolve_stationery_item(query: str):
    cleaned = query.lower().strip()
    if cleaned in STATIONERY_CATALOGUE:
        return cleaned, STATIONERY_CATALOGUE[cleaned]
    for cat_item, price in STATIONERY_CATALOGUE.items():
        if cleaned in cat_item or cat_item in cleaned:
            return cat_item, price
    tokens = [t.rstrip("s") for t in cleaned.split() if len(t) > 1]
    best_item, best_score = None, 0
    for cat_item in STATIONERY_CATALOGUE:
        score = sum(1 for tok in tokens if tok in cat_item)
        if score > best_score:
            best_item, best_score = cat_item, score
    if best_score >= 2:
        return best_item, STATIONERY_CATALOGUE[best_item]
    return None, None

def build_tools(ws, faq_collection):
    @tool
    def get_stationery_price(item: str, qty: int = 1) -> str:
        """Look up a stationery item in the BrightPage catalogue and calculate total price in rupees.
        Applies a 10% bulk discount automatically when quantity is 10 or more, or subtotal exceeds Rs 1000.

        Args:
            item: Stationery item name from catalogue
            qty: Quantity requested (must be at least 1)
        """
        if qty < 1:
            return f"Minimum quantity is at least 1 (got {qty}). Ask the customer for a valid quantity."
        matched_item, unit_price = _resolve_stationery_item(item)
        if matched_item is None:
            return f"'{item}' is not in our catalogue. I'll check with the store manager about availability."
        subtotal = unit_price * qty
        discount = round(subtotal * 0.10, 2) if (qty >= 10 or subtotal > 1000) else 0.0
        final_total = subtotal - discount
        if discount > 0:
            return f"{qty} x {matched_item} @ Rs {unit_price} = Rs {subtotal}, 10% bulk discount (-Rs {discount:.0f}) -> Rs {final_total:.0f}"
        return f"{qty} x {matched_item} @ Rs {unit_price} = Rs {final_total:.0f}"

    @tool
    def is_store_open(day: str) -> str:
        """Check BrightPage Stationery's opening hours for a day of the week, e.g. 'sunday' or 'monday'."""
        hours = BRIGHTPAGE_HOURS.get(day.lower().strip())
        return f"Open {hours} on {day}." if hours else f"'{day}' is not a valid day of the week."

    @tool
    def check_delivery_fee(order_total: float, delivery_type: str = "home_delivery") -> str:
        """Check BrightPage Stationery's local delivery fee for an order total.
        Store pickup is always free; local home delivery is free for orders above Rs 500 and Rs 40 otherwise.
        """
        if "pick" in delivery_type.lower():
            return "Store pickup is free (Rs 0)."
        if order_total > 500:
            return f"Delivery is free (Rs 0) because order total (Rs {order_total:.0f}) is above Rs 500."
        return f"Local delivery fee is Rs 40 because order total (Rs {order_total:.0f}) is Rs 500 or below."

    @tool
    def search_brightpage_faq(question: str) -> str:
        """Search BrightPage Stationery's policies and FAQs: delivery, out-of-city shipping, payment, UPI,
        cancellation, bulk discounts, returns, custom notebook printing, gift wrapping, GST invoices,
        school kits, paper GSM quality, pen refills, and store pickup.
        Use for policy or store rule questions, not for live item pricing or opening hours."""
        results = faq_collection.query(query_texts=[question], n_results=3)
        return "\n".join(results["documents"][0])

    @tool
    def log_stationery_order(
        customer_name: str,
        item: str,
        qty: int,
        delivery_type: str,
        delivery_date: str,
        notes: str = "",
    ) -> str:
        """Save a confirmed BrightPage Stationery order to the Orders sheet.
        Use ONLY when customer_name, catalogue item, qty (>= 1), delivery_type ('home_delivery' or 'store_pickup'), and delivery_date are confirmed."""
        matched_item, unit_price = _resolve_stationery_item(item)
        if matched_item is None:
            return f"Could not save order: '{item}' is not in the BrightPage catalogue."
        dtype = delivery_type.lower().strip().replace(" ", "_")
        if dtype not in ("home_delivery", "store_pickup"):
            dtype = "home_delivery" if "deliver" in dtype else "store_pickup" if "pick" in dtype else dtype
        try:
            record = StationeryOrder(
                customer_name=customer_name.strip(),
                item=matched_item,
                qty=qty,
                delivery_type=dtype,
                delivery_date=delivery_date.strip(),
                notes=notes.strip() or None,
            )
        except Exception as e:
            return f"Could not save order: {e}. Ask customer to correct it."

        subtotal = unit_price * record.qty
        discount = round(subtotal * 0.10, 2) if (record.qty >= 10 or subtotal > 1000) else 0.0
        after_discount = subtotal - discount
        delivery_fee = 0 if (record.delivery_type == "store_pickup" or after_discount > 500) else 40
        total = after_discount + delivery_fee

        ws.append_row([
            record.customer_name, record.item, record.qty, record.delivery_type,
            record.delivery_date, unit_price, discount, delivery_fee, total,
            record.notes or "",
        ])
        return (
            f"Saved order for {record.customer_name}: {record.qty} x {record.item} "
            f"({record.delivery_type} on {record.delivery_date}). "
            f"Discount: Rs {discount:.0f}, Delivery: Rs {delivery_fee}, Total: Rs {total:.0f}."
        )

    return get_stationery_price, is_store_open, check_delivery_fee, search_brightpage_faq, log_stationery_order

# ==============================================================================
# 5. Core LangGraph Router (3 Intents: order, faq, other)
# ==============================================================================
class BrightPageState(TypedDict):
    message: str
    intent: str
    reply: str

class Intent(BaseModel):
    intent: Literal["order", "faq", "other"] = Field(
        description=(
            "order = explicitly wants to buy or book stationery; "
            "faq = asks about prices, availability, timings, delivery or policies; "
            "other = greetings or anything unrelated to the stationery shop"
        )
    )

FAQ_PROMPT = """You answer questions for BrightPage Stationery & Office Supplies.
Use get_stationery_price for prices, is_store_open for timings, check_delivery_fee for delivery fees, and search_brightpage_faq for policies. Never guess.
If the answer isn't found, say "I'll check with the store manager." Under 50 words."""

ORDER_PROMPT = """You take stationery orders for BrightPage Stationery & Office Supplies.
Use get_stationery_price for totals and match clear customer wording directly.
Before log_stationery_order, make sure you have customer_name, item, qty, delivery_type ('home_delivery' or 'store_pickup') and delivery_date; ask for anything missing. After saving, read the order back. Under 60 words."""

def build_3intent_graph(llm, faq_agent, order_agent):
    classifier = llm.with_structured_output(Intent)

    def classify(state: BrightPageState) -> dict:
        result = invoke_retry(
            classifier,
            [("system", "Classify the customer message into Intent."), ("human", state["message"])],
        )
        return {"intent": result.intent}

    def ask_agent(agent, message: str) -> str:
        result = invoke_retry(agent, {"messages": [{"role": "user", "content": message}]})
        return result["messages"][-1].content

    def answer_faq(state: BrightPageState) -> dict:
        return {"reply": ask_agent(faq_agent, state["message"])}

    def take_order(state: BrightPageState) -> dict:
        return {"reply": ask_agent(order_agent, state["message"])}

    def other(state: BrightPageState) -> dict:
        return {"reply": "Hello from BrightPage Stationery! I can help with stationery items, prices, orders and our policies."}

    def route(state: BrightPageState) -> str:
        return state["intent"]

    builder = StateGraph(BrightPageState)
    builder.add_node("classify", classify)
    builder.add_node("faq", answer_faq)
    builder.add_node("order", take_order)
    builder.add_node("other", other)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route, {"order": "order", "faq": "faq", "other": "other"})
    builder.add_edge("faq", END)
    builder.add_edge("order", END)
    builder.add_edge("other", END)
    return builder.compile()

# ==============================================================================
# 6. Student's 50% Extension: 4-Intent Graph with Order Review Node
# ==============================================================================
class CustomShopState(TypedDict):
    message: str
    intent: str
    order_record: dict
    approval_status: str
    reply: str

class CustomIntent(BaseModel):
    intent: Literal["order", "faq", "complaint", "other"] = Field(
        description=(
            "order = wants to place or book a stationery order; "
            "faq = asks about prices, hours, delivery or policies; "
            "complaint = reports damaged, defective or missing items or asks for replacement; "
            "other = greetings or unrelated chat"
        )
    )

def build_4intent_graph(llm, ws, faq_agent, order_agent, search_faq_tool):
    my_classifier = llm.with_structured_output(CustomIntent)

    def my_classify(state: CustomShopState) -> dict:
        result = invoke_retry(
            my_classifier,
            [("system", "Classify the customer message into CustomIntent."), ("human", state["message"])],
        )
        return {"intent": result.intent}

    def my_faq(state: CustomShopState) -> dict:
        res = invoke_retry(faq_agent, {"messages": [{"role": "user", "content": state["message"]}]})
        return {"reply": res["messages"][-1].content}

    def my_order(state: CustomShopState) -> dict:
        rows_before = len(ws.get_all_records())
        res = invoke_retry(order_agent, {"messages": [{"role": "user", "content": state["message"]}]})
        reply = res["messages"][-1].content
        records = ws.get_all_records()
        new_record = records[-1] if len(records) > rows_before else {}
        return {"reply": reply, "order_record": new_record}

    def review_order(state: CustomShopState) -> dict:
        rec = state.get("order_record") or {}
        if not rec:
            return {"approval_status": "pending_customer_details"}
        qty = int(rec.get("qty", 0))
        discount = float(rec.get("discount", 0.0))
        total = float(rec.get("total", 0.0))
        status = "owner_review_required" if (qty >= 10 or discount > 0 or total > 1000) else "auto_cleared"
        return {"approval_status": status}

    def my_complaint(state: CustomShopState) -> dict:
        policy = search_faq_tool.invoke("damaged defective replacement 3 days")
        reply = invoke_retry(llm, [
            ("system", f"Respond empathetically using this BrightPage policy only, under 45 words:\n{policy}"),
            ("human", state["message"]),
        ]).content
        return {"reply": reply}

    def my_other(state: CustomShopState) -> dict:
        return {"reply": "Welcome to BrightPage Stationery! Ask me about prices, orders, policies, or damaged item replacements."}

    def my_route(state: CustomShopState) -> str:
        return state["intent"]

    my_builder = StateGraph(CustomShopState)
    my_builder.add_node("classify", my_classify)
    my_builder.add_node("faq", my_faq)
    my_builder.add_node("order", my_order)
    my_builder.add_node("review_order", review_order)
    my_builder.add_node("complaint", my_complaint)
    my_builder.add_node("other", my_other)
    my_builder.add_edge(START, "classify")
    my_builder.add_conditional_edges(
        "classify",
        my_route,
        {"order": "order", "faq": "faq", "complaint": "complaint", "other": "other"},
    )
    my_builder.add_edge("order", "review_order")
    my_builder.add_edge("review_order", END)
    my_builder.add_edge("faq", END)
    my_builder.add_edge("complaint", END)
    my_builder.add_edge("other", END)
    return my_builder.compile()

# ==============================================================================
# 7. Main Runner & Verification Suite
# ==============================================================================
def main():
    print("=" * 70)
    print("🔀 Day 6: Workflows, Planning & LangGraph")
    print("=" * 70)

    try:
        llm = get_llm(temperature=0.2)
        print("✅ LLM initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        sys.exit(1)

    # Setup Sheet & ChromaDB FAQ
    brightpage_csv = os.path.join(os.path.dirname(__file__), "brightpage_orders.csv")
    ws = LocalWorksheet(brightpage_csv, headers=[
        "customer_name", "item", "qty", "delivery_type", "delivery_date",
        "unit_price", "discount", "delivery_fee", "total", "notes"
    ])
    faq_col = build_chroma_faq()
    get_price_t, is_open_t, check_deliv_t, search_faq_t, log_order_t = build_tools(ws, faq_col)

    # 1. Plain-Python Router Demo
    print("\n--- [Demo 1] Plain-Python Router (handle) ---")
    plain_classifier = llm.with_structured_output(Intent)
    def handle(message: str) -> str:
        intent = invoke_retry(
            plain_classifier,
            [("system", "Classify customer message into Intent."), ("human", message)],
        ).intent
        if intent == "order":
            return f"[{intent}] Route to take_order"
        elif intent == "faq":
            return f"[{intent}] Route to answer_faq"
        else:
            return f"[{intent}] Route to greeting"

    for sample in ["Do you deliver to my area?", "Book 2 spiral notebooks", "Good morning!"]:
        print(f"'{sample}' -> {handle(sample)}")

    # 2. Build 3-Intent LangGraph
    print("\n--- [Demo 2] Focused Agents & 3-Intent LangGraph ---")
    faq_agent = create_agent(
        model=llm,
        tools=[search_faq_t, get_price_t, is_open_t, check_deliv_t],
        system_prompt=FAQ_PROMPT,
    )
    order_agent = create_agent(
        model=llm,
        tools=[get_price_t, check_deliv_t, log_order_t],
        system_prompt=ORDER_PROMPT,
    )
    graph = build_3intent_graph(llm, faq_agent, order_agent)
    print("✅ 3-Intent LangGraph compiled successfully.")

    # Stream 3-intent graph tests
    test_messages = [
        "Do you deliver to my area?",
        "I'm Ravi, book 2 a4 copier paper reams for home delivery on 24 Dec",
        "Good morning!",
        "Is the spiral notebook 200 pages?",
        "Naku 2 geometry box sets kavali, store pickup on 28 Dec, my name is Priya",
    ]
    for msg in test_messages:
        print(f"\nCustomer: {msg}")
        for step in graph.stream({"message": msg}, stream_mode="updates"):
            print("  Update:", step)

    # 3. Build 4-Intent Extended Graph with Review Node
    print("\n--- [Demo 3] Extended 4-Intent Graph with review_order Node ---")
    my_graph = build_4intent_graph(llm, ws, faq_agent, order_agent, search_faq_t)
    print("✅ Extended LangGraph compiled successfully.")

    ext_tests = [
        "What are your opening hours on Sunday?",
        "I'm Meena, order 10 spiral notebooks for store pickup on 30 Dec",
        "My geometry box arrived cracked yesterday, can you replace it?",
        "Thanks, have a great day!",
    ]
    for msg in ext_tests:
        print(f"\nCustomer: {msg}")
        for step in my_graph.stream({"message": msg}, stream_mode="updates"):
            print("  Update:", step)

    print("\n" + "=" * 70)
    print("Final Orders Logged in Sheet:")
    for row in ws.get_all_records():
        print(" ", row)
    print("=" * 70)
    print("Day 6 execution completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
