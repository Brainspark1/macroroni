import re
import threading
import time
import wave
import io

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import speech_recognition as sr
import mlx_whisper

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", "", text) # updating to include apostrophes in the case of contraction words like "don't"
    return text.split()

df = pd.read_csv("actual_dataset.csv")

texts = [preprocess_text(text) for text in df.iloc[:, 0]] # preprocessing text from each text in the first column/examples

labels_list = df.iloc[:, 1].tolist() # adding labels in second column to labels list

# using vectorizer from sklearn to fit and transform examples
vectorizer = TfidfVectorizer(analyzer=lambda x: x, ngram_range=(1, 3))
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

    probs = best_model.predict_proba(x)[0] # updating to better found model
    best_idx = probs.argmax()

    label = best_model.classes_[best_idx]
    confidence = probs[best_idx]

    return label, confidence

# taken from Joshua's file
env = gym.make("SuperMarioBros-v0", render_mode="rgb_array")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
obs,info = env.reset()

cv2.namedWindow("Mario", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mario", 1600, 800)

# currently held buttons
current_direction = None # could be left, right or nothing
running = False # is running or not

# temporary actions
jump_until = 0
duck_until = 0
fire_until = 0

lock = threading.Lock()

# method to update the state of what mario should be doing
def update_state(intent):
    global current_direction
    global running
    global jump_until
    global duck_until
    global fire_until

    with lock:
        now_moment = time.time()

        if intent == "MOVE_LEFT":
            current_direction = "left"

        elif intent == "MOVE_RIGHT":
            current_direction = "right"

        elif intent == "RUN":
            running = True

        elif intent == "JUMP":
            jump_until = now_moment + 0.25

        elif intent == "DUCK":
            duck_until = now_moment + 1.5

        elif intent == "FIRE":
            fire_until = now_moment + 0.2

        elif intent == "STOP":
            current_direction = None
            running = False

# replacing saving audio as separate temp file instead to a numpy array that can be processed more directly
def audio_to_numpy(audio):
    wav_data = audio.get_wav_data()

    with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())

        audio_np = np.frombuffer(
            frames,
            dtype=np.int16
        ).astype(np.float32)

        # normalizing into an int16 range
        audio_np = audio_np / 32768.0

    return audio_np

recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)

def correct_transcription(text):
    text = text.lower()

    replacements = {
        "dump": "jump",
        "jum": "jump",
        "jumps": "jump",
        "jumping": "jump",
        "chump": "jump",
        "jumpp": "jump",
        "John": "jump"
    }

    for wrong, correct in replacements.items():
        if wrong in text:
            text = text.replace(wrong, correct)

    return text

with mic as source:
    print("Calibrating microphone...")
    recognizer.adjust_for_ambient_noise(source, duration=1) # can be tuned for best results

def audio_callback(recognizer, audio):
    try:
        start_time = time.time()

        audio_np = audio_to_numpy(audio)

        result = mlx_whisper.transcribe(
            audio_np,
            path_or_hf_repo="mlx-community/whisper-tiny.en-mlx",
            language="en",
            initial_prompt=(
                "This is a Mario voice controller. "
                "Commands are: jump, run, left, right, duck, fire, stop, pause. "
                "The player frequently says jump."
            )
        )

        transcribed_text = correct_transcription(result["text"].strip())

        if not transcribed_text:
            print("\nNothing transcribed.") # if nothing in transcribed text, do nothing
            return
        
        words = transcribed_text.split()

        # if more than 15 words of only one type, ignore
        if len(words) > 15 and len(set(words)) == 1:
            print("Ignoring unusually long repeated transcript.")
            return
        elif len(words) > 15:
            print ("Ignoring unusually long transcript.")
            return
                
        intent, confidence = predict(transcribed_text)
        print(f"\nTranscript: {transcribed_text}")
        print(f"Intent: {intent} ({confidence:.2f})")

        # if more than 50% confidence of command
        if confidence > 0.5:
            update_state(intent)

        print("Latency:", time.time() - start_time)

    except Exception as e:
        print(e)

recognizer.listen_in_background(mic, audio_callback, phrase_time_limit=0.7)

print("Press q to end voice control.")

# method to get what buttons should be pressed by adding them to a list based on states of global action variables
def get_current_action():
    buttons = []

    with lock:
        now = time.time()

        if current_direction == "left":
            buttons.append("left")

        elif current_direction == "right":
            buttons.append("right")

        if running:
            buttons.append("B")

        # if haven't finished jumping, press jump button
        if now < jump_until:
            buttons.append("A")

        # if haven't finished ducking, press duck button
        if now < duck_until:
            buttons.append("down")

        # if haven't finished throwing fireball, press fire button
        if now < fire_until:
            buttons.append("A")

    # if no more buttons left to press ...
    if len(buttons) == 0:
        buttons = ["NOOP"] # default to doing nothing

    # return mapping those buttons to mario's movement
    return COMPLEX_MOVEMENT.index(buttons)

# gameplay loop taken from Joshua's file
while True:
    action = get_current_action()

    obs, reward, terminated, truncated, info = env.step(action)

    cv2.imshow("Mario", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if terminated or truncated:
        obs, info = env.reset()

# closing environment and game/all windows
env.close()
cv2.destroyAllWindows()
