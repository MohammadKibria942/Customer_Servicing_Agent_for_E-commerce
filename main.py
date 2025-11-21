from agents.orchestrator import handle_message


def main():
    print("=== E-commerce Customer Service Agent ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye! 👋")
            break

        if not user_input:
            continue

        intent, response = handle_message(user_input)
        print(f"[Intent detected: {intent}]")
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
