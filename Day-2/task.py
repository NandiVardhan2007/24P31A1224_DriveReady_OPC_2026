"""
Day 2 Task: Building Your Own AI Agent (BrightPage Stationery)
DriveReady OPC 2026 - Agentic AI Track
"""

import os
import sys
import json
import re
import functools
import time
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError

# 1. Load API Configuration
load_dotenv(find_dotenv())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        from google.colab import userdata
        GROQ_API_KEY = userdata.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    except Exception:
        pass

# Groq OpenAI-compatible client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", "")
)

MODEL_NAME = "llama-3.3-70b-versatile"

# 2. Domain Data & Inventory
STATIONERY_CATALOGUE = {
    "spiral notebook (200 pgs)": 85,
    "a4 copier paper ream (500 sheets)": 320,
    "gel pen pack (10 pens, blue/black)": 150,
    "pastel highlighter set (6 colors)": 180,
    "permanent marker (black, pack of 4)": 120,
    "sticky notes cube (400 sheets, 3x3)": 95,
    "heavy duty stapler + pins": 240,
    "correction tape (2 pack)": 90,
    "transparent tape roll (1 inch)": 35,
    "plastic expanding file folder": 160
}

# 3. System Prompt for BrightPage Stationery
SYSTEM_PROMPT = """You are the virtual customer assistant for BrightPage Stationery & Office Supplies.
Role: Help customers check product availability, answer pricing queries in INR, and assist with orders.
Facts:
- We stock standard stationery items: notebooks, copier paper, gel pens, highlighters, markers, sticky notes, staplers, tape, and folders.
- Free delivery on orders above ₹500; standard delivery charge is ₹40.
- Same-day dispatch for orders placed before 2 PM.
Rules:
- Never quote prices for items not in the inventory; politely say: "I'll check with our store manager and confirm."
- Always confirm order items and calculate totals accurately in INR (₹).
- Keep responses friendly, professional, and under 80 words.
"""

# 4. Helper Decorator for Logging Calls
def log_call(func):
    """Decorator that logs function execution, arguments, and return values."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
        print(f"👉 [CALL] {func.__name__}({arg_str})")
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"✅ [DONE] {func.__name__} returned in {elapsed:.2f}s")
        return result
    return wrapper

# 5. Few-Shot Intent Classifier
def classify_intent(customer_message: str) -> str:
    """Classifies customer message into 'order', 'inquiry', or 'complaint' using few-shot prompting."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer intent classifier for a stationery business. "
                "Classify the customer's message into exactly one category: 'order', 'inquiry', or 'complaint'. "
                "Reply ONLY with the category name in lowercase. Do not include extra text or punctuation."
            )
        },
        {"role": "user", "content": "I would like to buy 3 packs of blue gel pens."},
        {"role": "assistant", "content": "order"},
        {"role": "user", "content": "What is the price of an A4 copier paper ream?"},
        {"role": "assistant", "content": "inquiry"},
        {"role": "user", "content": "The markers I received yesterday were dried out and not working."},
        {"role": "assistant", "content": "complaint"},
        {"role": "user", "content": customer_message}
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=messages
    )
    return response.choices[0].message.content.strip().lower()

# 6. Structured Output: JSON Order Extraction
ORDER_EXTRACTION_SYSTEM = """You are an order extraction engine for BrightPage Stationery.
Extract customer order details from the message into a strictly valid JSON object.
Return ONLY raw JSON. No markdown backticks, no code fences, no commentary.
Format:
{
    "customer_name": string or null,
    "items": [
        {"item": string, "quantity": integer}
    ],
    "delivery_urgency": "standard" or "urgent",
    "delivery_location": string or null
}
"""

def extract_order_json(message: str) -> dict:
    """Extracts structured JSON order details from free-form customer text."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ORDER_EXTRACTION_SYSTEM},
            {"role": "user", "content": message}
        ]
    )
    raw = response.choices[0].message.content.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", raw)
    clean = re.sub(r"\s*```$", "", clean)
    return json.loads(clean)

# 7. Stateful Multi-Turn Agent Class
class StationeryAgent:
    """Conversational Agent for BrightPage Stationery with stateful memory and error resilience."""
    
    def __init__(self, system_prompt: str = SYSTEM_PROMPT, max_turns: int = 6):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.history = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        # Memory Window: Keep only the last max_turns exchanges (user + assistant = 2 items per turn)
        truncated_history = self.history[-(self.max_turns * 2):]
        messages = [{"role": "system", "content": self.system_prompt}] + truncated_history

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.2,
                messages=messages
            )
            reply = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except RateLimitError:
            return "⚠️ Service is busy due to rate limits. Please try again shortly."
        except APIConnectionError:
            return "⚠️ Connection error to AI service. Please check your network."
        except AuthenticationError:
            return "⚠️ Authentication error. Please verify your GROQ_API_KEY."
        except Exception as e:
            return f"⚠️ Unexpected error: {e}"

    def reset_memory(self):
        self.history = []

# 8. Main CLI Execution & Verification Tests
def main():
    print("=" * 65)
    print("📚 BrightPage Stationery & Office Supplies AI Agent (Day 2)")
    print("=" * 65)

    if not os.environ.get("GROQ_API_KEY"):
        print("Please configure GROQ_API_KEY in your .env file to run live calls.")
        sys.exit(1)

    print("\n--- [Demo 1] Few-Shot Customer Intent Classification ---")
    samples = [
        "Can you send 2 boxes of pastel highlighters to our office?",
        "Do you have sticky notes cube available, and how much is it?",
        "The paper ream I ordered was torn and wet when delivered!"
    ]
    for msg in samples:
        intent = classify_intent(msg)
        print(f"Message: \"{msg}\"\nDetected Intent: [{intent.upper()}]\n")

    print("--- [Demo 2] Structured JSON Order Extraction ---")
    order_msg = "Hi, I am Dr. Ramesh from Green Valley Clinic. Please send 2 reams of A4 copier paper and 3 pastel highlighter sets urgently."
    try:
        extracted = extract_order_json(order_msg)
        print(f"Customer Input: {order_msg}")
        print("Extracted Structured JSON:")
        print(json.dumps(extracted, indent=2))
    except Exception as e:
        print(f"Error extracting JSON: {e}")

    print("\n--- [Demo 3] Stateful Multi-Turn Conversation ---")
    agent = StationeryAgent(max_turns=4)
    turns = [
        "Hi, my name is Priya. Do you have gel pens in stock?",
        "Great, what is the price for 2 packs?",
        "By the way, do you remember my name?"
    ]
    for t in turns:
        print(f"Customer: {t}")
        ans = agent.chat(t)
        print(f"Agent:    {ans}\n")

    print("=" * 65)
    print("Day 2 execution completed successfully.")
    print("=" * 65)

if __name__ == "__main__":
    main()
