import json
import os
import math
from typing import List, Dict, Tuple

from utils.llm_client import get_embedding



DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FAQ_JSON = os.path.join(DATA_DIR, "faq.json")

_FAQ_CACHE: List[Dict[str, str]] = []
_FAQ_EMBEDDINGS: List[List[float]] = []
_CACHE_INITIALIZED = False


def _init_faq_cache():
    """
    Load FAQs and pre-compute embeddings for semantic search.
    This is called lazily the first time we need it.
    """
    global _FAQ_CACHE, _FAQ_EMBEDDINGS, _CACHE_INITIALIZED

    if _CACHE_INITIALIZED:
        return

    faqs = load_faq()
    embeddings: List[List[float]] = []

    for faq in faqs:
        # You can embed question, answer, or both. Here we embed both concatenated.
        text_to_embed = f"Q: {faq['question']}\nA: {faq['answer']}"
        emb = get_embedding(text_to_embed)
        embeddings.append(emb)

    _FAQ_CACHE = faqs
    _FAQ_EMBEDDINGS = embeddings
    _CACHE_INITIALIZED = True



def load_faq() -> List[Dict[str, str]]:
    with open(FAQ_JSON, encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))



def find_best_faq_match(user_question: str, k: int = 1) -> List[Dict[str, str]]:
    """
    RAG-style retrieval:
    - Compute embedding for user question
    - Compare with precomputed FAQ embeddings
    - Return top-k most similar FAQ entries
    """
    _init_faq_cache()

    query_emb = get_embedding(user_question)
    scored: List[Tuple[float, Dict[str, str]]] = []

    for faq, faq_emb in zip(_FAQ_CACHE, _FAQ_EMBEDDINGS):
        score = cosine_similarity(query_emb, faq_emb)
        scored.append((score, faq))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    top_k = [faq for score, faq in scored[:k]]
    return top_k

