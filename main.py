# main.py
from chat_agent import BibleChatAgent

if __name__ == "__main__":
    print("\n📖 Welcome to the Offline Biblical Bot CLI! Type 'exit' to quit.\n")
    agent = BibleChatAgent("./bible/bible_books")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye. May your study be blessed.")
            break

        response = agent.ask(user_input)
        print(f"Bot: {response}\n")
