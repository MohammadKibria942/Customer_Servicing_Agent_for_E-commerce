from typing import Literal, Tuple, Optional

from utils.llm_client import chat_completion, classify_intent_llm
from tools.order_tools import find_order_by_id, find_orders_by_email
from tools.faq_tools import find_best_faq_match


def classify_intent(user_message: str) -> Literal["ORDER", "FAQ", "OTHER"]:
    """
    Hybrid classifier:
    1) Fast heuristic for obvious cases (to save tokens).
    2) Fallback to LLM-based classification for everything else.
    """
    text = user_message.lower()

    # 1) Simple heuristic for very clear cases
    order_keywords = ["order", "track", "tracking", "delivery", "shipped", "package", "parcel"]
    faq_keywords = ["return", "refund", "policy", "delivery time", "shipping", "payment", "pay", "international"]

    # If message is mostly digits or mentions 'order', treat as ORDER
    tokens = text.split()
    has_number = any(t.isdigit() for t in tokens)

    if any(k in text for k in order_keywords) or has_number:
        return "ORDER"
    if any(k in text for k in faq_keywords):
        return "FAQ"

    # 2) Fallback to LLM classifier
    intent = classify_intent_llm(user_message)
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
    Use FAQ tools to find the best matching FAQ and then have the LLM answer.
    """
    faq = find_best_faq_match(user_message)

    system_prompt = (
        "You are a helpful e-commerce customer service agent. "
        "You will be given an FAQ entry (question + answer) and the customer's question. "
        "Answer the customer using the FAQ answer. "
        "Do not contradict the FAQ. If the FAQ doesn't fully answer, say so briefly."
    )

    user_prompt = (
        f"Customer question: {user_message}\n\n"
        f"Relevant FAQ question: {faq['question']}\n"
        f"Relevant FAQ answer: {faq['answer']}\n\n"
        "Using this FAQ, provide the best possible answer to the customer."
    )

    return chat_completion(system_prompt, user_prompt)


def handle_message(user_message: str) -> Tuple[str, str]:
    """
    Main orchestration function:
    - Classify intent (ORDER / FAQ / OTHER)
    - Call the appropriate specialist agent
    Returns (intent, response_text)
    """
    intent = classify_intent(user_message)

    if intent == "ORDER":
        response = handle_order_query(user_message)
    elif intent == "FAQ":
        response = handle_faq_query(user_message)
    else:
        # Fallback for OTHER – stay professional and focused on support
        system_prompt = (
            "You are a professional e-commerce customer support assistant. "
            "Your ONLY job is to help with:\n"
            "- order tracking and delivery issues\n"
            "- returns, refunds, and product issues\n"
            "- general store policies (shipping, payments, etc.)\n\n"
            "If the user asks for anything unrelated (for example: jokes, small talk, "
            "personal questions, or general chit-chat), you MUST politely decline and "
            "redirect them back to support-related topics.\n\n"
            "Rules:\n"
            "- Do NOT tell jokes or entertain unrelated requests.\n"
            "- Keep responses short, polite, and professional.\n"
            "- When declining, briefly explain that you are focused on customer support "
            "and suggest they ask a question about an order or store policy."
        )

        response = chat_completion(system_prompt, user_message)

    return intent, response

