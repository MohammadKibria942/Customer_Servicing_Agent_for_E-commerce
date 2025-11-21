import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MODEL = "gpt-4.1-mini"  # or another model name available to you
EMBEDDING_MODEL = "text-embedding-3-small"  # or another embedding model you have access to


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
        "Given ONE user message, classify it into exactly one of three categories:\n\n"
        "1) ORDER\n"
        "   - The user is asking about the status, location, tracking, or delivery of a "
        "     specific existing order.\n"
        "   - Examples: 'Where is my order 1001?', 'Track my package', 'Has my order shipped?'\n\n"
        "2) FAQ\n"
        "   - The user is asking about general policies or information such as: "
        "     delivery times, shipping options (standard vs express), return policy, "
        "     refunds, payment methods, international shipping, etc.\n"
        "   - IMPORTANT: If the user asks how long delivery will take in general "
        "     (even if they say 'my order'), classify this as FAQ unless they clearly "
        "     refer to tracking a specific existing order.\n"
        "   - Examples: 'How long does express shipping take?', "
        "     'What is your return policy?', 'Do you ship internationally?'\n\n"
        "3) OTHER\n"
        "   - Anything not related to customer support for this store.\n\n"
        "Answer with ONLY one word: ORDER, FAQ, or OTHER."
    )

    response = chat_completion(system_prompt, user_message)
    response = response.strip().upper()

    if "ORDER" in response:
        return "ORDER"
    if "FAQ" in response:
        return "FAQ"
    return "OTHER"



def get_embedding(text: str) -> List[float]:
    """
    Get an embedding vector for the given text using OpenAI embeddings.
    """
    # OpenAI recommends stripping newlines
    clean_text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[clean_text]
    )
    return response.data[0].embedding
