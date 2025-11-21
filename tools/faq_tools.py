import json
import os
from typing import List, Dict


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FAQ_JSON = os.path.join(DATA_DIR, "faq.json")


def load_faq() -> List[Dict[str, str]]:
    with open(FAQ_JSON, encoding="utf-8") as f:
        return json.load(f)


def find_best_faq_match(user_question: str) -> Dict[str, str]:
    """
    Very simple matching for v1:
    - look for FAQ whose question shares the most words with user_question.
    You can upgrade this later to embeddings / RAG.
    """
    faqs = load_faq()
    user_words = set(user_question.lower().split())
    best_score = -1
    best_faq = faqs[0]

    for faq in faqs:
        faq_words = set(faq["question"].lower().split())
        overlap = len(user_words.intersection(faq_words))
        if overlap > best_score:
            best_score = overlap
            best_faq = faq

    return best_faq
