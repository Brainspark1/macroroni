import os
import re
import threading

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import speech_recognition as sr
import torch
import whisper
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

whisper_model = whisper.load_model("tiny") # lowers latency

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

df = pd.read_csv("actual_dataset.csv")

texts = [preprocess_text(text) for text in df.iloc[:, 0]] # preprocessing text from each text in the first column/examples

labels_list = df.iloc[:, 1].tolist() # adding labels in second column to labels list

# using vectorizer from sklearn to fit and transform examples
vectorizer = TfidfVectorizer(analyzer=lambda x:x)
X = vectorizer.fit_transform(texts)

X_train, X_temp, y_train, y_temp = train_test_split(X, labels_list, test_size=0.3, random_state=42, stratify=labels_list)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# intiailizing variable to store best model
best_model = None
best_accuracy = -1 # initialized to negative 1 as accuracy cannot be negative, so good starting place

# hyperparameter tuning for c value - found in existing LiveNLP.py file
for c in np.logspace(-4, 2, 7): # testing from 10^-4 to 10^2
    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_val,y_val)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("Validation accuracy:", best_accuracy)
print("Test accuracy:", best_model.score(X_test,y_test))
print(classification_report(y_test, best_model.predict(X_test)))

# method to predict using LogisticRegression model taken from existing file
def predict(text):
    processed_text = preprocess_text(text)
    x = vectorizer.transform([processed_text])

    probs = model.predict_proba(x)[0]
    best_idx = probs.argmax()

    label = model.classes_[best_idx]
    confidence = probs[best_idx]

    return label, confidence

# taken from Joshua's file
env = gym.make("SuperMarioBros-v0", render_mode="rgb_array")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
obs,info = env.reset()

cv2.namedWindow("Mario", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mario", 1600, 800)

mario_action = 0
lock = threading.Lock() # prevents two parts of code from changing same variable/action at same time

# custom method to set hyperparameter durations on actions
def set_action(action, duration):
    global mario_action

    mario_action = action

    # reset method sets the current action back to default value/0
    def reset():
        global mario_action
        mario_action = 0 

    # performs action for duration amount of time before resetting
    threading.Timer(duration, reset).start()

# holds currently grasped intents
intent_map = {}

for label in best_model.classes_:
    check_label = label.lower()
    
    # complex movement logic adapted from Joshua's file
    if "left" in check_label:
        intent_map[label] = COMPLEX_MOVEMENT.index(["left"])
    elif "right" in check_label:
        intent_map[label] = COMPLEX_MOVEMENT.index(["right"])
    elif "jump" in check_label:
        intent_map[label] = COMPLEX_MOVEMENT.index(["A"])
    elif "run_left" in check_label:
        intent_map[label] = COMPLEX_MOVEMENT.index(["left", "B"])
    elif "run_right" in check_label:
        intent_map[label] = COMPLEX_MOVEMENT.index(["right", "B"])
    elif "duck" in check_label or "down" in check_label:
        intent_map[label] = COMPLEX_MOVEMENT.index(["down"])
    else:
        intent_map[label] = 0 # default value if nothing

recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)

with mic as source:
    print("Calibrating microphone...")
    recognizer.adjust_for_ambient_noise(source,duration=5) # can be tuned for best results

def audio_callback(recognizer, audio):
    global mario_action

    try:
        with open("temp_chunk.wav", "wb") as f:
            f.write(audio.get_wav_data())

        result = whisper_model.transcribe("temp_chunk.wav", fp16=torch.cuda.is_available())

        os.remove("temp_chunk.wav")

        transcribed_text = result["text"].strip()
        if not transcribed_text:
            return # if nothing in transcribed text, do nothing
        
        intent, confidence = predict(transcribed_text)
        print(f"\nTranscript: {transcribed_text}")
        print(f"Intent: {intent} ({confidence:.2f})")

        # if more than 60% confidence of command
        if confidence > 0.6:
            with lock:
                action = intent_map.get(intent, 0)

                if "jump" in intent.lower():
                    set_action(action, 0.25)
                elif "run" in intent.lower():
                    set_action(action, 1.0)
                elif "left" in intent.lower() or "right" in intent.lower():
                    set_action(action, 1.0)
                elif "duck" in intent.lower():
                    set_action(action, 1.5)
                else:
                    set_action(action, 0.5)
    except Exception as e:
        print(e)

recognizer.listen_in_background(mic, audio_callback, phrase_time_limit=3)

print("Press q to end voice control.")

while True:
    with lock:
        action = mario_action
    
    # from Joshua's mario.py
    obs, reward, terminated, truncated, info = env.step(action)
    cv2.imshow("Mario", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # stopping listening if q is pressed
    if (cv2.waitKey(1)) and (0xFF == ord("q")):
        break
    if terminated or truncated:
        obs, info = env.reset()

# closing environment and game/all windows
env.close()
cv2.destroyAllWindows()