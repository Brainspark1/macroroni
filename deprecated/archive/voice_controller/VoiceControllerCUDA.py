import io
import re
import threading
import time
import wave

import cv2
import gymnasium as gym
from faster_whisper import WhisperModel
import numpy as np
import pandas as pd
import speech_recognition as sr
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class VoiceControllerCUDA:
    def __init__(self, dataset_path="actual_dataset.csv"):
        self.lock = threading.Lock()

        # movement states
        self.current_direction = None
        self.running = False
        self.jump_until = 0.0
        self.duck_until = 0.0
        self.fire_until = 0.0

        # vectorizer training
        self.vectorizer = TfidfVectorizer(analyzer=lambda x: x, ngram_range=(1, 3))
        self.best_model = None
        self._train_model(dataset_path)

        # faster whisper model setup
        self.whisper_model = WhisperModel(
            model_size_or_path="tiny.en", 
            device="cuda", 
            compute_type="float16"
        )

        # speech recognition setup
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(sample_rate=16000)
        self._calibrate_mic()

    # method to clean text and split it into tokens
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z'\s]", "", text)  # including apostrophes for contractions like "don't"
        tokens = text.split()

        return tokens

    # protected method to train the model, can only be accessed by this and sub-classes created
    def _train_model(self, dataset_path):
        # loading dataset
        df = pd.read_csv(dataset_path)

        texts = [self.preprocess_text(text) for text in df.iloc[:, 0]]
        labels_list = df.iloc[:, 1].tolist()

        X = self.vectorizer.fit_transform(texts)
        X_train, X_temp, y_train, y_temp = train_test_split(X, labels_list, test_size=0.3, random_state=42, stratify=labels_list)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        best_accuracy = -1 # started at -1 as accuracy cannot be less than 0, so good placeholder value 

        # from c = 10^-4 to 10^2
        for c in np.logspace(-4, 2, 7):
            model = LogisticRegression(C=c, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)

            accuracy = model.score(X_val, y_val)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model

        print(f"Validation accuracy: {best_accuracy:.4f}")
        print(f"Test accuracy: {self.best_model.score(X_test, y_test):.4f}")

    # method to use the logistic regression model to predict the intent
    def predict(self, text):
        processed_text = self.preprocess_text(text)
        x = self.vectorizer.transform([processed_text])
        probs = self.best_model.predict_proba(x)[0]
        best_idx = probs.argmax()
        return self.best_model.classes_[best_idx], probs[best_idx]

    # method can be OVERRIDDEN by subclasses if they also have issues with transcription
    def correct_transcription(self, text):
        text = text.lower()

        replacements = {
            "dump": "jump",
            "jum": "jump",
            "jumps": "jump",
            "jumping": "jump",
            "chump": "jump",
            "jumpp": "jump",
            "john": "jump",
        }

        for wrong, correct in replacements.items():
            if wrong in text:
                text = text.replace(wrong, correct)

        return text

    # method to convert audio to matrix of numpy values for less latency storage of audio
    def audio_to_numpy(self, audio):
        wav_data = audio.get_wav_data()

        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0 # getting values into int16 range

        return audio_np

    # protected method to calibrate the microphone for a custom duration
    def _calibrate_mic(self, duration=1):
        with self.mic as source:
            print("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)

    # method which can be OVERRIDDEN to update what state the relevant object should be executing
    def update_state(self, intent):
        with self.lock:
            now_moment = time.time()
            if intent == "MOVE_LEFT":
                self.current_direction = "left"
            elif intent == "MOVE_RIGHT":
                self.current_direction = "right"
            elif intent == "RUN":
                self.running = True
            elif intent == "JUMP":
                self.jump_until = now_moment + 0.25
            elif intent == "DUCK":
                self.duck_until = now_moment + 1.5
            elif intent == "FIRE":
                self.fire_until = now_moment + 0.2
            elif intent == "STOP":
                self.current_direction = None
                self.running = False

    def audio_callback(self, recognizer, audio, beam_size=1, initial_prompt="This is a Mario voice controller. "
                                                                "Commands are: jump, run, left, right, duck, fire, stop, pause. "
                                                                "The player frequently says jump."):
        try:
            start_time = time.time()
            audio_np = self.audio_to_numpy(audio)

            segments, info = self.whisper_model.transcribe(
                audio_np,
                language="en",
                initial_prompt=initial_prompt,
                beam_size=beam_size  # greedy decoding, only one word
            )
            
            # combining segments into single transcribed string
            text_result = "".join([segment.text for segment in segments])
            transcribed_text = self.correct_transcription(text_result.strip())

            if not transcribed_text:
                return

            words = transcribed_text.split()

            if len(words) > 15:
                print("Ignoring unusually long transcript.")
                return

            intent, confidence = self.predict(transcribed_text)
            print(f"\nTranscript: {transcribed_text} | Intent: {intent} ({confidence:.2f})")

            if confidence > 0.5:
                self.update_state(intent)

            print(f"Latency: {time.time() - start_time:.4f}s")
        except Exception as e:
            print(f"Audio Callback Error: {e}")

    def start_listening(self, phrase_time_limit=0.7):
        print("Press 'q' in the emulator window to stop.")

        return self.recognizer.listen_in_background(
            self.mic, self.audio_callback, phrase_time_limit=phrase_time_limit
        )

    # method can be OVERRIDDEN to get the current array of actions that map to buttons which should be pressed at the same time
    def get_current_action(self):
        buttons = []
        with self.lock:
            now = time.time()
            if self.current_direction == "left":
                buttons.append("left")
            elif self.current_direction == "right":
                buttons.append("right")

            if self.running:
                buttons.append("B")

            if now < self.jump_until:
                buttons.append("A")

            if now < self.duck_until:
                buttons.append("down")

            if now < self.fire_until:
                buttons.append("A")

        if not buttons:
            buttons = ["NOOP"]

        return COMPLEX_MOVEMENT.index(buttons)

# main method to be called by user in their own other file
if __name__ == "__main__":
    controller = VoiceControllerCUDA(dataset_path="actual_dataset.csv")

    env = gym.make("SuperMarioBros-v0", render_mode="rgb_array")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    obs, info = env.reset()

    cv2.namedWindow("Mario", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mario", 1600, 800)

    stop_listening_fn = controller.start_listening()
    print("Voice active. Press 'q' in the gameplay window to exit.")

    try:
        # gameplay loop taken from Joshua's file
        while True:
            action = controller.get_current_action()

            obs, reward, terminated, truncated, info = env.step(action)

            cv2.imshow("Mario", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            # exit if q is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        # closing environment and game/all windows
        env.close()
        cv2.destroyAllWindows()