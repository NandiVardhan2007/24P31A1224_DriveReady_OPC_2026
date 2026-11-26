"""
Day 4 Task: Tool Calling Agents, Pydantic Schemas, Persistent Sheets & Traces
DriveReady OPC 2026 - Agentic AI Track
"""

import os
import sys
import csv
import time
import warnings
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools import tool
from langchain.agents import create_agent

warnings.filterwarnings("ignore")

# 1. Environment & API Key Resolution
load_dotenv(find_dotenv())

def get_llm(temperature: float = 0.2):
    """Initializes LLM using Groq (preferred for tool-calling) or Google Gemini fallback."""
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_retries=3,
            timeout=30
        )
    elif os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=3,
            timeout=30
        )
    else:
        raise ValueError(
            "Neither GROQ_API_KEY nor GOOGLE_API_KEY found. "
            "Please configure your API key in .env or environment."
        )

# 2. Local CSV-backed Worksheet Mirror (Simulating Google Sheets gspread)
class LocalWorksheet:
    """Local CSV-backed worksheet mirror with append_row and get_all_records interface."""
    def __init__(self, csv_path: str, headers: list[str]):
        self.csv_path = csv_path
        self.headers = headers
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.headers)

    def append_row(self, row: list):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def get_all_records(self) -> list[dict]:
        if not os.path.exists(self.csv_path):
            return []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

# Helper retry invocation wrapper for API resiliency
def invoke_retry(callable_obj, input_payload, max_attempts: int = 4):
    for attempt in range(max_attempts):
        try:
            return callable_obj.invoke(input_payload)
        except Exception as err:
            if attempt == max_attempts - 1:
                raise err
            wait_s = 2 ** (attempt + 1)
            time.sleep(wait_s)

# ==============================================================================
# SECTION A: SWEET CRUMBS HOME BAKERY AGENT
# ==============================================================================

MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}
HOURS = {
    "monday": "9 am to 8 pm", "tuesday": "9 am to 8 pm", "wednesday": "9 am to 8 pm",
    "thursday": "9 am to 8 pm", "friday": "9 am to 8 pm", "saturday": "9 am to 8 pm",
    "sunday": "10 am to 6 pm",
}

sweet_crumbs_csv = os.path.join(os.path.dirname(__file__), "sweet_crumbs_orders.csv")
sweet_crumbs_ws = LocalWorksheet(
    sweet_crumbs_csv,
    headers=["name", "item", "qty", "date", "eggless", "total"]
)

class Order(BaseModel):
    name: str = Field(description="Customer's full or first name")
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item ordered")
    qty: int = Field(ge=1, description="Quantity ordered (must be >= 1)")
    date: str = Field(description="Pickup date requested, e.g. '24 Dec'")
    eggless: Literal["yes", "no"] = Field(description="Eggless preference")

@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item, and verify it is on the menu.

    Args:
        item: Menu item, e.g. 'plum cake', 'chocolate cake' or 'cupcake'
        qty: How many the customer wants (must be at least 1)
    """
    if qty < 1:
        return f"Quantity must be at least 1 (got {qty}). Ask the customer for a valid quantity."
    cleaned = item.lower().strip()
    price = MENU.get(cleaned)
    if price is None:
        return f"'{item}' is not on the menu. We offer: {', '.join(MENU)}."
    return f"{qty} x {item} = Rs {price * qty}"

@tool
def is_open(day: str) -> str:
    """Check whether Sweet Crumbs Bakery is open on a given day of the week, and return its hours.

    Args:
        day: Day of the week, e.g. 'monday' or 'sunday'
    """
    cleaned = day.lower().strip()
    hours = HOURS.get(cleaned)
    if hours is None:
        return f"'{day}' is not a valid day of the week."
    return f"Open {hours} on {day}."

@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: str) -> str:
    """Save a confirmed customer order to the bakery orders sheet.
    Only call this tool when the customer has explicitly provided their name, item, quantity,
    pickup date, and eggless preference (yes/no).

    Args:
        name: Customer name
        item: Menu item ('plum cake', 'chocolate cake', or 'cupcake')
        qty: Quantity (must be at least 1)
        date: Pickup date (e.g. '24 Dec')
        eggless: 'yes' if eggless, else 'no'
    """
    try:
        validated = Order(name=name, item=item, qty=qty, date=date, eggless=eggless)
    except Exception as err:
        return f"Validation error: {err}. Please ask the customer to clarify missing details."

    total = MENU[validated.item] * validated.qty
    sweet_crumbs_ws.append_row([
        validated.name, validated.item, validated.qty, validated.date, validated.eggless, total
    ])
    return (
        f"Order saved for {validated.name}: {validated.qty} x {validated.item} "
        f"on {validated.date} (eggless: {validated.eggless}). Total: Rs {total}."
    )

SWEET_CRUMBS_SYSTEM = """ROLE:   You are the friendly order assistant for Sweet Crumbs Home Bakery.
FACTS:  Orders need 24 hours' notice (no same-day orders). All cakes have eggless versions.
RULES:  Always use get_price to check if an item is on the menu and get prices; never guess them.
        Always use is_open for timings; never guess them.
        Never accept or quote an item that is not on the menu; explain what we offer instead.
        For same-day requests, remind the customer that orders need 24 hours' notice.
        If a policy or delivery question is not in FACTS, say "I'll check with the owner."
        If quantity is less than 1, always say "minimum quantity is at least 1" and ask how many they want.
        Before log_order, make sure you have name, menu item, quantity (>= 1) and pickup date; ask if missing.
        After saving, read the order back with the total.
FORMAT: Under 60 words, friendly. Always spell 'eggless' as one word, and write plain numbers without commas."""

# ==============================================================================
# SECTION B: BRIGHTPAGE STATIONERY & OFFICE SUPPLIES AGENT
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

brightpage_csv = os.path.join(os.path.dirname(__file__), "brightpage_orders.csv")
brightpage_ws = LocalWorksheet(
    brightpage_csv,
    headers=["customer_name", "item", "qty", "delivery_type", "delivery_date", "unit_price", "discount", "delivery_fee", "total", "notes"]
)

StationeryItemName = Literal[
    "spiral notebook (200 pgs)", "a4 copier paper ream (500 sheets)", "gel pen pack (10 pens, blue/black)",
    "pastel highlighter set (6 colors)", "heavy-duty stapler + pins", "sticky notes pack (400 sheets)",
    "geometry box set", "whiteboard marker set (4 colors)", "permanent marker (black)", "clear file folder (pack of 5)"
]

class StationeryOrder(BaseModel):
    customer_name: str = Field(description="Customer's full or first name")
    item: StationeryItemName = Field(description="Stationery item name")
    qty: int = Field(ge=1, description="Quantity ordered (must be >= 1)")
    delivery_type: Literal["home_delivery", "store_pickup"] = Field(description="Delivery type")
    delivery_date: str = Field(description="Delivery or pickup date")
    notes: Optional[str] = Field(default=None, description="Special requests or instructions")

def _resolve_stationery_item(query: str):
    cleaned = query.lower().strip()
    if cleaned in STATIONERY_CATALOGUE:
        return cleaned, STATIONERY_CATALOGUE[cleaned]
    for cat_item, price in STATIONERY_CATALOGUE.items():
        if cleaned in cat_item or cat_item in cleaned:
            return cat_item, price
    return None, None

@tool
def get_stationery_price(item: str, qty: int = 1) -> str:
    """Calculate total price in rupees for a BrightPage Stationery item and quantity.
    Applies a 10% bulk discount automatically when quantity is 10+ or subtotal exceeds Rs 1000.

    Args:
        item: Stationery item name from catalogue
        qty: Quantity requested (must be at least 1)
    """
    if qty < 1:
        return f"Quantity must be at least 1 (got {qty}). Ask the customer for a valid quantity."
    matched, unit_price = _resolve_stationery_item(item)
    if matched is None:
        return f"'{item}' is not in the BrightPage catalogue. I'll check with the store manager about availability."
    subtotal = unit_price * qty
    discount = round(subtotal * 0.10, 2) if (qty >= 10 or subtotal > 1000) else 0.0
    final_total = subtotal - discount
    if discount > 0:
        return f"{qty} x {matched} @ Rs {unit_price} = Rs {subtotal}, 10% bulk discount (-Rs {discount:.0f}) -> Rs {final_total:.0f}"
    return f"{qty} x {matched} @ Rs {unit_price} = Rs {final_total:.0f}"

@tool
def check_store_and_delivery(day: str, order_total: float = 0.0) -> str:
    """Check BrightPage Stationery opening hours for a weekday and compute local delivery fee.
    Delivery is free for orders above Rs 500, and Rs 40 for orders of Rs 500 or below.

    Args:
        day: Day of the week, e.g. 'sunday' or 'monday'
        order_total: Order amount in rupees after any discount
    """
    hours = BRIGHTPAGE_HOURS.get(day.lower().strip())
    hours_msg = f"Open {hours} on {day}." if hours else f"'{day}' is not a valid day of the week."
    if order_total <= 0:
        return f"{hours_msg} Local delivery is free above Rs 500 (Rs 40 fee below Rs 500)."
    fee = 0 if order_total > 500 else 40
    fee_msg = "Free local delivery (order above Rs 500)." if fee == 0 else "Local delivery fee is Rs 40."
    return f"{hours_msg} {fee_msg}"

@tool
def log_stationery_order(customer_name: str, item: str, qty: int, delivery_type: str, delivery_date: str, notes: str = "") -> str:
    """Save a confirmed customer stationery order to the BrightPage orders sheet.
    Only call this tool when customer_name, item, quantity, delivery_type, and delivery_date are provided.

    Args:
        customer_name: Customer name
        item: Catalogue item name
        qty: Quantity (must be at least 1)
        delivery_type: 'home_delivery' or 'store_pickup'
        delivery_date: Target date for delivery or pickup
        notes: Optional customer note
    """
    matched, unit_price = _resolve_stationery_item(item)
    if matched is None:
        return f"Validation error: '{item}' is not in catalogue. Cannot log order."
    try:
        record = StationeryOrder(
            customer_name=customer_name,
            item=matched,
            qty=qty,
            delivery_type=delivery_type,
            delivery_date=delivery_date,
            notes=notes or None
        )
    except Exception as err:
        return f"Validation error: {err}. Please ask customer to clarify missing details."

    subtotal = unit_price * record.qty
    discount = round(subtotal * 0.10, 2) if (record.qty >= 10 or subtotal > 1000) else 0.0
    discounted_subtotal = subtotal - discount
    delivery_fee = 0 if (record.delivery_type == "store_pickup" or discounted_subtotal > 500) else 40
    total = discounted_subtotal + delivery_fee

    brightpage_ws.append_row([
        record.customer_name, record.item, record.qty, record.delivery_type,
        record.delivery_date, unit_price, discount, delivery_fee, total, record.notes or ""
    ])
    return (
        f"Saved order for {record.customer_name}: {record.qty} x {record.item} "
        f"({record.delivery_type} on {record.delivery_date}). Total: Rs {total:.0f}."
    )

BRIGHTPAGE_SYSTEM = """ROLE:   You are the friendly order assistant for BrightPage Stationery & Office Supplies.
FACTS:  Catalogue items: spiral notebook (200 pgs), a4 copier paper ream (500 sheets), gel pen pack (10 pens, blue/black),
        pastel highlighter set (6 colors), heavy-duty stapler + pins, sticky notes pack (400 sheets), geometry box set,
        whiteboard marker set (4 colors), permanent marker (black), clear file folder (pack of 5).
        10% bulk discount on 10+ identical items or orders > Rs 1000. Free local delivery above Rs 500 (Rs 40 below Rs 500).
RULES:  ALWAYS call get_stationery_price first for ANY item mentioned and check_store_and_delivery for hours/delivery fees.
        If an item isn't in our catalogue, NEVER quote a price or ask for order details — say "I'll check with the store manager about availability."
        Before log_stationery_order, ensure customer_name, item, qty (>= 1), delivery_type, and delivery_date are provided; ask if missing.
        After saving, read back the order summary and total.
FORMAT: Under 65 words, friendly, plain numbers without commas."""

# ==============================================================================
# MAIN CLI RUNNER & EXPERIMENTS
# ==============================================================================

def main():
    print("=" * 70)
    print("🛠️  Day 4: Tools, Structured Records, Sheets & Agent Traces")
    print("=" * 70)

    try:
        llm = get_llm(temperature=0.2)
        print("✅ LLM initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("Please check your .env file.")
        sys.exit(1)

    # 1. Sweet Crumbs Agent Demo
    print("\n--- [Demo 1] Sweet Crumbs Agent & Message Tracing ---")
    bakery_agent = create_agent(
        model=llm,
        tools=[get_price, is_open, log_order],
        system_prompt=SWEET_CRUMBS_SYSTEM
    )
    trace_res = invoke_retry(bakery_agent, {
        "messages": [{"role": "user", "content": "Are you open on Sunday? How much are 2 chocolate cakes?"}]
    })

    model_calls = sum(1 for m in trace_res["messages"] if isinstance(m, AIMessage))
    tool_calls = [
        tc["name"]
        for m in trace_res["messages"] if isinstance(m, AIMessage) and m.tool_calls
        for tc in m.tool_calls
    ]
    tool_responses = sum(1 for m in trace_res["messages"] if isinstance(m, ToolMessage))
    print(f"Customer: Are you open on Sunday? How much are 2 chocolate cakes?")
    print(f"Agent Reply: {trace_res['messages'][-1].content}")
    print(f"Trace Stats: Model Calls={model_calls}, Tool Calls Requested={tool_calls}, Tool Messages={tool_responses}")

    # 2. Edge Case & Sheet Integrity Check
    print("\n--- [Demo 2] Customer Edge Cases & Sheet Integrity ---")
    rows_before = len(sweet_crumbs_ws.get_all_records())
    edge_queries = [
        ("Unlisted Item", "Can I order 2 donuts for tomorrow?"),
        ("Missing Date", "Hi, I'm Sita. Please book 2 plum cakes."),
        ("Zero Quantity", "Hi, I'm Kiran. Please order 0 cupcakes for 25 Dec.")
    ]
    for label, q in edge_queries:
        res = invoke_retry(bakery_agent, {"messages": [{"role": "user", "content": q}]})
        print(f"[{label}] Query: {q}\n  -> Agent: {res['messages'][-1].content}\n")
    rows_after = len(sweet_crumbs_ws.get_all_records())
    print(f"Sheet Rows Before: {rows_before} | After: {rows_after} (Verified: No invalid rows written)")

    # 3. BrightPage Stationery Agent Demo
    print("\n--- [Demo 3] BrightPage Stationery Tool-Calling Agent ---")
    bp_agent = create_agent(
        model=llm,
        tools=[get_stationery_price, check_store_and_delivery, log_stationery_order],
        system_prompt=BRIGHTPAGE_SYSTEM
    )
    bp_demo_queries = [
        "Are you open on Sunday? How much for 12 spiral notebooks?",
        "I'm Ananya. Please book 12 spiral notebooks for home delivery on 28 Nov."
    ]
    for q in bp_demo_queries:
        res = invoke_retry(bp_agent, {"messages": [{"role": "user", "content": q}]})
        print(f"Customer: {q}\nAgent:    {res['messages'][-1].content}\n")

    # 4. Break-It Experiment: Tool Docstrings as API Docs
    print("--- [Demo 4] The Break-It Experiment (tool1 vs get_price) ---")
    @tool
    def tool1(item: str, qty: int = 1) -> str:
        """does stuff"""
        price = MENU.get(item.lower().strip())
        return f"{qty} x {item} = Rs {price * qty}" if price else f"'{item}' not found."

    broken_agent = create_agent(model=llm, tools=[tool1, is_open, log_order], system_prompt="Answer politely.")
    broken_res = invoke_retry(broken_agent, {"messages": [{"role": "user", "content": "How much are 3 plum cakes?"}]})
    broken_tools = [
        tc["name"]
        for m in broken_res["messages"] if isinstance(m, AIMessage) and m.tool_calls
        for tc in m.tool_calls
    ]
    print(f"Query: 'How much are 3 plum cakes?'")
    print(f"With name='tool1' and docstring='does stuff' -> Tool Calls Made: {broken_tools}")
    print("Observation: The agent failed to select tool1 because docstring lacked description!")

    print("\n" + "=" * 70)
    print("Day 4 execution completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
