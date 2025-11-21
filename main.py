from agents.orchestrator import handle_message
from utils.state import ConversationState


def main():
    print("=== E-commerce Customer Service Agent ===")
    print("Type 'exit' to quit.\n")

    state = ConversationState()  # <-- NEW

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye! 👋")
            break

        if not user_input:
            continue

        intent, response = handle_message(user_input, state)  # <-- pass state
        print(f"[Intent detected: {intent}]")
        print(f"Agent: {response}\n")



if __name__ == "__main__":
    main()
