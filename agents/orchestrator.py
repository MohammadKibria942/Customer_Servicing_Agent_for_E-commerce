from typing import Literal, Tuple, Optional

from utils.llm_client import chat_completion, classify_intent_llm
from tools.order_tools import find_order_by_id, find_orders_by_email
from tools.faq_tools import find_best_faq_match
import re



def classify_intent(user_message: str) -> Literal["ORDER", "FAQ", "OTHER"]:
    """
    Main intent classifier.

    1) Use the LLM classifier as the primary signal.
    2) Apply a very small heuristic override ONLY when clearly necessary
       (e.g. pure tracking requests with an order ID).
    """
    # 1) Let the LLM decide first
    intent = classify_intent_llm(user_message)

    text = user_message.lower()
    tokens = text.split()
    has_number = any(t.isdigit() for t in tokens)

    # 2) Tiny heuristic override: if the user very clearly provides an order id
    #    and talks about tracking/status, force ORDER.
    tracking_keywords = ["track", "tracking", "where is", "status", "shipped", "delivery status"]

    if any(k in text for k in tracking_keywords) and has_number:
        return "ORDER"

    # Otherwise, trust the LLM's classification
    if intent not in {"ORDER", "FAQ", "OTHER"}:
        intent = "OTHER"
    return intent




def handle_order_query(user_message: str) -> str:
    """
    v1: expect the user to include an order ID in the message.
    Later you can extend this to ask follow-up questions.
    """
    # Extract the first number-like token as order ID (more robust)
    tokens = user_message.replace("#", " ").replace(":", " ").split()
    possible_ids = [t for t in tokens if t.isdigit()]

    if not possible_ids:
        return (
            "I can help you with your order, but I’ll need your order ID.\n"
            "Please provide it in your next message, for example: 'My order ID is 1001'."
        )

    order_id = possible_ids[0]
    order = find_order_by_id(order_id)

    if not order:
        return (
            f"I couldn’t find any order with ID {order_id}. "
            "Please check the ID and try again, or provide the email you used for the order."
        )

    # Use LLM just to turn raw data into a nice explanation
    system_prompt = (
        "You are a helpful e-commerce customer service agent. "
        "Explain the order status to the customer in a friendly and concise way. "
        "Use the provided order data exactly, do not invent new information."
    )

    # Build a short description of the order for the model
    order_summary = (
        f"Order ID: {order['order_id']}\n"
        f"Customer email: {order['user_email']}\n"
        f"Status: {order['status']}\n"
        f"Courier: {order['courier']}\n"
        f"Tracking URL: {order['tracking_url']}\n"
        f"Expected delivery date: {order['expected_delivery']}\n"
        f"Last updated: {order['last_updated']}"
    )

    user_prompt = (
        "Here is the order data:\n\n"
        f"{order_summary}\n\n"
        "Please explain to the customer where their order is and what will happen next."
    )

    return chat_completion(system_prompt, user_prompt)


def handle_faq_query(user_message: str) -> str:
    """
    Use RAG-style FAQ retrieval:
    - Retrieve top FAQ entries semantically using embeddings
    - Let the LLM synthesize the best answer using those entries
    """
    faqs = find_best_faq_match(user_message, k=3)  # get top 3 for more context
    # Build a context string with the retrieved FAQs
    faq_context_parts = []
    for i, faq in enumerate(faqs, start=1):
        faq_context_parts.append(
            f"FAQ {i} Question: {faq['question']}\nFAQ {i} Answer: {faq['answer']}"
        )
    faq_context = "\n\n".join(faq_context_parts)

    system_prompt = (
        "You are a helpful and professional e-commerce customer service agent. "
        "You will be given several FAQ entries (questions and answers) and the customer's question. "
        "Your job is to answer the customer using ONLY the information from the FAQs. "
        "If the FAQs do not contain enough information, say so briefly and suggest contacting human support. "
        "Do not invent new policies or contradict the FAQs."
    )

    user_prompt = (
        f"Customer question:\n{user_message}\n\n"
        f"Here are relevant FAQ entries:\n{faq_context}\n\n"
        "Using ONLY this information, provide the best possible answer to the customer."
    )

    return chat_completion(system_prompt, user_prompt)



def handle_message(user_message: str, state) -> Tuple[str, str]:
    """
    Main orchestrator with multi-turn order handling.
    """

    # --- 1. Multi-turn: If user previously needed to provide an order ID ---
    if state.pending_intent == "ORDER_NEEDS_ID":
        # Try extracting order ID
        match = re.search(r"\b(\d+)\b", user_message)
        if match:
            order_id = match.group(1)
            state.reset()  # clear pending state
            return "ORDER", handle_order_query(f"order {order_id}")
        else:
            return "ORDER", (
                "I still need your order ID to look up your order.\n"
                "Please provide it, for example: 'My order ID is 1001'."
            )

    # --- 2. Normal first-time intent detection ---
    intent = classify_intent(user_message)
    state.last_intent = intent

    # --- 3. ORDER: check if an order ID exists ---
    if intent == "ORDER":
        # Try to extract an order ID
        match = re.search(r"\b(\d+)\b", user_message)
        if match:
            order_id = match.group(1)
            return "ORDER", handle_order_query(f"order {order_id}")

        # No order ID provided → ask for it, and store pending state
        state.pending_intent = "ORDER_NEEDS_ID"
        state.pending_order_message = user_message
        return "ORDER", (
            "I can help with your order, but I’ll need your order ID.\n"
            "Please provide it in your next message, for example: 'My order ID is 1001'."
        )

    # --- 4. FAQ ---
    if intent == "FAQ":
        return "FAQ", handle_faq_query(user_message)

    # --- 5. OTHER ---
    system_prompt = (
        "You are a professional e-commerce customer support assistant. "
        "You ONLY assist with order tracking, returns, refunds, shipping, and payments. "
        "If the user asks something unrelated, politely decline and redirect them."
    )
    return "OTHER", chat_completion(system_prompt, user_message)


