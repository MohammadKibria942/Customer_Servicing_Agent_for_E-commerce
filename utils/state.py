class ConversationState:
    """
    Holds short-term memory for multi-turn order conversations.
    """
    def __init__(self):
        self.pending_intent = None         # e.g., "ORDER_NEEDS_ID"
        self.pending_order_message = None  # store user’s original question
        self.last_intent = None            # last classified intent

    def reset(self):
        self.pending_intent = None
        self.pending_order_message = None
        self.last_intent = None
