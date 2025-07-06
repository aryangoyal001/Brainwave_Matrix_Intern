import random
import string
import os
import time

# Predefined list of words
word_list = ["apple", "house", "train", "pizza", "chair"]
# Randomly choose one word
word_to_guess = random.choice(word_list)

# Setup
guessed_letters = []
incorrect_guesses = 0
max_incorrect = 6

# Create a display version of the word with underscores
display_word = ['_'] * len(word_to_guess)

# Hangman stages for incorrect attempts
hangman_stages = [
    "\n   +---+\n       |\n       |\n       |\n      ===",
    "\n   +---+\n   O   |\n       |\n       |\n      ===",
    "\n   +---+\n   O   |\n   |   |\n       |\n      ===",
    "\n   +---+\n   O   |\n  /|   |\n       |\n      ===",
    "\n   +---+\n   O   |\n  /|\\  |\n       |\n      ===",
    "\n   +---+\n   O   |\n  /|\\  |\n  /    |\n      ===",
    "\n   +---+\n   O   |\n  /|\\  |\n  / \\  |\n      ==="
]

# Clear screen function
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Stylized display function
def show_status():
    print(hangman_stages[incorrect_guesses])
    print("\n================ HANGMAN ================")
    print("Word:", ' '.join(display_word))
    print(f"Guessed letters: {' '.join(sorted(guessed_letters)) if guessed_letters else 'None'}")
    print(f"Incorrect guesses left: {max_incorrect - incorrect_guesses}")
    print("========================================")

# Welcome screen
clear_screen()
print("🎮 Welcome to HANGMAN!")
print("Guess the word one letter at a time or try to guess the whole word.")
print("You have 6 incorrect guesses. Good luck!")
time.sleep(2)

# Game loop
while incorrect_guesses < max_incorrect and '_' in display_word:
    clear_screen()
    show_status()
    guess = input("\n🔤 Enter a letter or the full word: ").lower().strip()

    # Full word guess
    if len(guess) > 1:
        if guess == word_to_guess:
            display_word = list(word_to_guess)
            break
        else:
            print("\n❌ Incorrect full word guess.")
            incorrect_guesses += 1
            time.sleep(1.5)
            continue

    # Single letter guess validation
    if len(guess) != 1 or guess not in string.ascii_lowercase:
        print("\n❌ Please enter a valid single letter (a-z) or the full word.")
        time.sleep(1.5)
        continue

    if guess in guessed_letters:
        print("\n⚠️ You already guessed that letter.")
        time.sleep(1.5)
        continue

    guessed_letters.append(guess)

    if guess in word_to_guess:
        print("\n✅ Correct!")
        for i in range(len(word_to_guess)):
            if word_to_guess[i] == guess:
                display_word[i] = guess
    else:
        print("\n❌ Incorrect.")
        incorrect_guesses += 1
    time.sleep(1.5)

# Game end
clear_screen()
show_status()
print("\n==================== RESULT ====================")
if '_' not in display_word:
    print("🎉 Congratulations! You guessed the word:", word_to_guess)
else:
    print("😞 Game Over! You've run out of guesses.")
    print("The correct word was:", word_to_guess)
print("================================================")
