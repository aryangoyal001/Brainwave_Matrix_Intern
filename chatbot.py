import tkinter as tk
from tkinter import scrolledtext

# Function to get bot response based on user input
def get_bot_response(user_input):
    user_input = user_input.lower().strip()

    if user_input == "hello":
        return "Hi!"
    elif user_input == "how are you":
        return "I'm fine, thanks!"
    elif user_input == "what is your name":
        return "I'm SimpleBot!"
    elif user_input == "tell me a joke":
        return "Why did the computer sneeze? It had a virus! 😂"
    elif user_input == "help":
        return "You can say: hello, how are you, what is your name, tell me a joke, or bye."
    elif user_input == "bye":
        return "Goodbye!"
    else:
        return "Sorry, I didn't understand that."

# Function to send message
def send_message():
    user_input = entry.get()
    if not user_input.strip():
        return

    # Display user message
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_input + "\n")
    chat_area.yview(tk.END)  # Auto-scroll

    # Get and display bot response
    bot_response = get_bot_response(user_input)
    chat_area.insert(tk.END, "Bot: " + bot_response + "\n\n")
    chat_area.yview(tk.END)

    chat_area.config(state=tk.DISABLED)
    entry.delete(0, tk.END)

    # Auto-close on "bye"
    if user_input.lower().strip() == "bye":
        root.after(1500, root.destroy)

# GUI setup
root = tk.Tk()
root.title("SimpleBot - Rule Based Chatbot")
root.geometry("400x500")
root.resizable(False, False)

# Chat area
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
chat_area.config(state=tk.DISABLED)
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Entry and send button
entry_frame = tk.Frame(root)
entry_frame.pack(pady=10)

entry = tk.Entry(entry_frame, font=("Arial", 12), width=28)
entry.pack(side=tk.LEFT, padx=(0, 10))
entry.focus()

send_button = tk.Button(entry_frame, text="Send", font=("Arial", 10), command=send_message)
send_button.pack(side=tk.RIGHT)

# Send message on Enter key
root.bind('<Return>', lambda event: send_message())

# Start with greeting
chat_area.config(state=tk.NORMAL)
chat_area.insert(tk.END, "Bot: Hello! I'm SimpleBot. Type 'help' to see what I can do.\n\n")
chat_area.config(state=tk.DISABLED)

root.mainloop()
