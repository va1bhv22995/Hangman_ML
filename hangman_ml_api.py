# hangman_ml_api.py
# Hangman ML API
# ----------------------------
# This script trains/loads an LSTM-based Hangman model for airline domain words/data and provides an API
# to play Hangman either word by word or on a list of test words.
# Flask is used for API endpoints.
# ----------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
import os
import argparse
from flask import Flask, request, jsonify

# ----------------------------
# 1. Load Words from File
# ----------------------------

def load_words(file_paths):
    
    """
    Load words from one or more text files.
    Each word is cleaned to contain only lowercase letters (a-z).
    Returns a sorted list of unique words.
    """
    
    words = set()
    for file_path in file_paths:
        with open(file_path) as f:
            for w in f:
                
                # Remove unwanted characters and lowercase
                w = ''.join(c for c in w.strip().lower() if c in string.ascii_lowercase)
                if w:
                    words.add(w)
    return sorted(words)

# ----------------------------
# 2. Dataset for Training
# ----------------------------

ALL_LETTERS = string.ascii_lowercase
LETTER_TO_IDX = {c: i for i, c in enumerate(ALL_LETTERS)}
IDX_TO_LETTER = {i: c for i, c in enumerate(ALL_LETTERS)}

class HangmanDataset(Dataset):
    
    """
    PyTorch Dataset for Hangman training.
    Each sample is a tuple: (current_state_sequence, guessed_letters_list, target_letter)
    """
    
    def __init__(self, words, max_len=None):
        self.samples = []
        
        # Maximum length of word sequences (for padding)
        self.max_len = max_len or max(len(w) for w in words)

        for word in words:
            guessed = set()
            for letter in word:
                
                # Current state shows guessed letters; others are '_'
                state = [l if l in guessed else "_" for l in word]
                self.samples.append((state, list(guessed), letter))
                guessed.add(letter)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, guessed, target = self.samples[idx]
       
        # Convert letters to indices; 26 is used for padding ('_')
        seq = [LETTER_TO_IDX.get(l, 26) for l in state] + [26] * (self.max_len - len(state))
        target_idx = LETTER_TO_IDX[target]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

# ----------------------------
# 3. LSTM Model Definition
# ----------------------------

class HangmanRNN(nn.Module):
    
    """
    LSTM-based model to predict the next letter in Hangman.
    Input: sequence of letter indices (with padding)
    Output: probability distribution over 26 letters
    """
    
    def __init__(self, vocab_size=27, embed_size=32, hidden_size=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 26)  # 26 letters output

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        
        # Use the output of the last timestep
        return self.fc(out[:, -1, :])

# ----------------------------
# 4. Train Model
# ----------------------------

def train_model(words, epochs=30):
    
    """
    Train the HangmanRNN model on the given word list.
    Saves trained model as 'hangman_model.pth'.
    """
    
    dataset = HangmanDataset(words)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = HangmanRNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "hangman_model.pth")
    print("\nModel saved as hangman_model.pth")
    return model

# ----------------------------
# 5. Get Next Guess
# ----------------------------

def get_next_guess(state_str, guessed_letters, guesses_remaining, WORDS, model):
    
    """
    Given the current state of the word and already guessed letters,
    predict the next letter using the trained model.
    """
    
    seq = [LETTER_TO_IDX.get(l, 26) for l in state_str.replace(" ", "")]
    seq += [26] * (max(len(w) for w in WORDS) - len(seq))  # padding
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor([seq], dtype=torch.long)), dim=1).squeeze().numpy()
    
    # Mask out already guessed letters
    for l in guessed_letters:
        probs[LETTER_TO_IDX[l]] = 0
    return {"nextGuess": IDX_TO_LETTER[probs.argmax()]}

# ----------------------------
# 6. Play Game Helper
# ----------------------------

def play_game(secret, WORDS, model, max_mistakes=6):
    
    """
    Simulate a full game of Hangman for a single word.
    Returns:
        - won (bool): True if the word was guessed
        - mistakes (int): number of incorrect guesses
    """
    
    guessed = []
    state = "_" * len(secret)
    mistakes = 0

    while mistakes < max_mistakes and "_" in state:
        guess = get_next_guess(" ".join(state), guessed, max_mistakes - mistakes, WORDS, model)["nextGuess"]
        guessed.append(guess)
        if guess in secret:
            state = "".join([c if c in guessed else "_" for c in secret])
        else:
            mistakes += 1

    return "_" not in state, mistakes

# ----------------------------
# 7. Evaluate Accuracy
# ----------------------------

def evaluate_accuracy(words_list, WORDS, model):
    
    """
    Evaluate the model on a list of words.
    Returns accuracy and average mistakes.
    """
    
    solved, total_mistakes = 0, 0
    for word in words_list:
        won, mistakes = play_game(word, WORDS, model)
        solved += int(won)
        total_mistakes += mistakes
    return solved / len(words_list) * 100, total_mistakes / len(words_list)

# ----------------------------
# 8. Main + API
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Run API server")
    args = parser.parse_args()

    # Load training and test words
    train_words = load_words(["traindata.txt"])
    test_words = load_words(["testdata.txt"])

    # Load or train model
    model = HangmanRNN()
    if os.path.exists("hangman_model.pth"):
        print("Loading saved model...")
        model.load_state_dict(torch.load("hangman_model.pth"))
        model.eval()
    else:
        print("Training model...")
        model = train_model(train_words, epochs=40)
        model.eval()

    if args.api:
        app = Flask(__name__)

        # ----------------------------
        # Endpoint 1: /next_guess
        # ----------------------------
        
        @app.route("/next_guess", methods=["POST"])
        def next_guess_api():
            
            """
            Input JSON: {"currentWordState": "_ _ e _ a n", "guessedLetters": ["e","a","n"], "guessesRemaining": 4}
            Output JSON: {"nextGuess": "t"}
            """
            
            data = request.get_json()
            return jsonify(get_next_guess(
                data["currentWordState"],
                data["guessedLetters"],
                data["guessesRemaining"],
                train_words,
                model
            ))

        # ----------------------------
        # Endpoint 2: /play_all_tests
        # ----------------------------
        
        @app.route("/play_all_tests", methods=["GET"])
        def play_all_tests():
            
            """
            Plays all words in testdata.txt and returns full results.
            """
            
            results, solved, total_mistakes = [], 0, 0
            for secret in test_words:
                guessed, state, mistakes = [], "_"*len(secret), 0
                while mistakes < 6 and "_" in state:
                    guess = get_next_guess(" ".join(state), guessed, 6-mistakes, train_words, model)["nextGuess"]
                    guessed.append(guess)
                    if guess in secret:
                        state = "".join([c if c in guessed else "_" for c in secret])
                    else:
                        mistakes += 1
                won = "_" not in state
                solved += int(won)
                total_mistakes += mistakes
                results.append({"word": secret, "finalState": state, "mistakes": mistakes, "result": "WIN" if won else "LOSS"})
            return jsonify({
                "totalTestWords": len(test_words),
                "solvedWords": solved,
                "accuracy": solved / len(test_words) * 100,
                "averageMistakes": total_mistakes / len(test_words),
                "details": results
            })

        # ----------------------------
        # Endpoint 3: /play_test_words
        # ----------------------------
        
        @app.route("/play_test_words", methods=["POST"])
        def play_test_words():
           
            """
            Accepts a JSON file: {"words": ["airplane", "airport", ...]}
            Returns full game results for all words.
            """
            
            data = request.get_json()
            words = data.get("words", [])
            results, solved, total_mistakes = [], 0, 0
            for secret in words:
                guessed, state, mistakes = [], "_"*len(secret), 0
                while mistakes < 6 and "_" in state:
                    guess = get_next_guess(" ".join(state), guessed, 6-mistakes, train_words, model)["nextGuess"]
                    guessed.append(guess)
                    if guess in secret:
                        state = "".join([c if c in guessed else "_" for c in secret])
                    else:
                        mistakes += 1
                won = "_" not in state
                solved += int(won)
                total_mistakes += mistakes
                results.append({
                    "word": secret,
                    "finalState": state,
                    "mistakes": mistakes,
                    "result": "WIN" if won else "LOSS"
                })
            return jsonify({
                "totalTestWords": len(words),
                "solvedWords": solved,
                "accuracy": solved / len(words) * 100,
                "averageMistakes": total_mistakes / len(words),
                "details": results
            })

        print("API running at http://127.0.0.1:5000")
        app.run(debug=True)

    else:
        
        # Command-line evaluation
        print("===== Training Data Evaluation =====")
        acc, avg = evaluate_accuracy(train_words, train_words, model)
        print(f"Accuracy: {acc:.2f}%, Avg mistakes: {avg:.2f}")

        print("\n===== Testing Data Evaluation =====")
        acc, avg = evaluate_accuracy(test_words, train_words, model)
        print(f"Accuracy: {acc:.2f}%, Avg mistakes: {avg:.2f}")