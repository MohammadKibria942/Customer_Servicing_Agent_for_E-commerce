import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MODEL = "gpt-4.1-mini"  # or another model name available to you


def chat_completion(system_prompt: str, user_message: str) -> str:
    """
    Simple helper to send a single-turn chat completion and return the text.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()

def classify_intent_llm(user_message: str) -> str:
    """
    Use the LLM to classify the user's intent for routing.
    Returns one of: ORDER, FAQ, OTHER
    """
    system_prompt = (
        "You are an intent classification assistant for an e-commerce support chatbot. "
        "Given a single user message, classify it into one of three categories:\n"
        "- ORDER: questions about order status, delivery, tracking, shipment, package, etc.\n"
        "- FAQ: general questions about return policy, shipping times, payment methods, etc.\n"
        "- OTHER: anything else.\n\n"
        "Answer with ONLY one word: ORDER, FAQ, or OTHER."
    )

    response = chat_completion(system_prompt, user_message)
    response = response.strip().upper()

    if "ORDER" in response:
        return "ORDER"
    if "FAQ" in response:
        return "FAQ"
    return "OTHER"
