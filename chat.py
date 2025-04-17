# chat.py
from src.inference.chatbot import Chatbot

if __name__ == "__main__":
    chatbot = Chatbot()
    print("Welcome to the Chatbot!")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")
