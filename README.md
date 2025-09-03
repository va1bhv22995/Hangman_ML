# Hangman ML API

## Overview

This project implements a **Hangman game AI** using an **LSTM-based neural network** built with PyTorch for airline domain words.  
The model predicts the next letter in a Hangman word based on the current state of guessed letters.  
A **Flask API** is provided to interact with the model programmatically.  

The code is self-contained: all scripts, datasets, and dependencies are included.  
The API can run with minimal setup on any system terminal.

---

## Approach / Strategy

1. **Data Preparation**
   - Words are loaded from text files (`traindata.txt` for training, `testdata.txt` for testing).  
   - Words are cleaned to include only lowercase English letters (a-z).  

2. **Machine Learning Model**
   - An **LSTM-based model** predicts the next letter.
   - Input: sequence of letters (current word state + padding for unknown letters `_`)  
   - Output: probability distribution over 26 English letters  
   - Trained using **CrossEntropyLoss** and **Adam optimizer**.

3. **Gameplay Logic**
   - The model receives the **current word state** and **guessed letters**.  
   - Predicts the **next best guess** letter.  
   - Continues until the word is solved or **6 mistakes** are made.

4. **API Endpoints**
   - `/next_guess` – Predicts the next letter given the current state.
   - `/play_all_tests` – Plays all words in `testdata.txt` and returns results.
   - `/play_test_words` – Accepts a user-provided JSON file of words and returns results.

---

## Installation & Setup

1. **Install Dependencies**

    -Python 3.8 or higher (https://www.python.org/downloads/)
    -Donwload/Install Git (https://git-scm.com/downloads/)
    -Download/Install VSCode (https://code.visualstudio.com/) (Optional but recommended)
    -Download/Install Postman Desktop (https://www.postman.com/downloads/)

2. **Clone the repository:**

    -Open the folder in your command promt/terminal where you want to clone the repository and post the following:

        git clone https://github.com/va1bhv22995/Hangman_ML.git
        cd Hangman_ML
        
3. **Running & Testing Instructions**

    -In terminal post the following to install requiremets:
        pip install -r requirements.txt

    -In terminal post the following to run the python script/API:
        python hangman_ml_api.py --api

    -Open Postman Destop to test the model/access the APIs on your required data:
        API Usage Examples:
            - http://127.0.0.1:5000/next_guess         # For POST request with JSON input
            - http://127.0.0.1:5000/play_all_tests     # For GET request – plays all words in testdata.txt
            - http://127.0.0.1:5000/play_test_words    # POST request with a JSON file of words
        Note: For post request go to body>raw and paste the words/data in json format(example given in the hangman_ml_api.py file)
        
        OR JUST GO TO THE FOLLOWING LINK TO SEE ALL OF THE ABOVE APIs WORKSPACE IN POSTMAN:
        https://www.postman.com/dark-satellite-3798579/workspace/hangman-ml/request/48139795-5d6876aa-521a-46ad-960a-840e618bf3ef?action=share&source=copy-link&creator=48139795

5. **Libraries Used**
    This project uses the following well-known Python libraries:

    - **torch, torch.nn, torch.optim, torch.utils.data** – PyTorch library used to define, train, and run the LSTM neural network for Hangman prediction.
    - **flask** – Web framework to create API endpoints for interacting with the model.
    - **argparse** – Standard Python library to handle command-line arguments.
    - **string, os** – Standard Python libraries for text processing, file handling, and general utilities.

