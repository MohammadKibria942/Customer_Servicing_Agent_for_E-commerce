import csv
import os
from typing import Optional, Dict, List


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")


def load_orders() -> List[Dict[str, str]]:
    orders = []
    with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orders.append(row)
    return orders


def find_order_by_id(order_id: str) -> Optional[Dict[str, str]]:
    """
    Simple lookup by order_id.
    """
    orders = load_orders()
    for order in orders:
        if order["order_id"] == order_id:
            return order
    return None


def find_orders_by_email(user_email: str) -> List[Dict[str, str]]:
    """
    Return all orders for a given email.
    """
    orders = load_orders()
    return [o for o in orders if o["user_email"].lower() == user_email.lower()]
