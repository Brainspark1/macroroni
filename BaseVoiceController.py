import io
import re
import threading
import time
import wave
import numpy as np
import pandas as pd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
import gymnasium as gym

class BaseVoiceController:
    def __init__(self, dataset_path, device_backend="mps", model_size="tiny.en", initial_prompt=None):
        self.lock = threading.Lock()
        self.device_backend = device_backend.lower()
        self.initial_prompt = initial_prompt or "Voice controller listening for commands."
        
        self._init_transcription_engine(model_size)
        
        self.vectorizer = TfidfVectorizer(analyzer=lambda x: x, ngram_range=(1, 3))
        self.best_model = None
        self._train_model(dataset_path)
        
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(sample_rate=16000)
        self._calibrate_mic()

    def _init_transcription_engine(self, model_size):
        self.device_backend = self.device_backend.lower()

        if self.device_backend == "mps":
            try:
                import mlx_whisper
                self.mlx_whisper = mlx_whisper
                
                self.model_path = f"mlx-community/whisper-{model_size}-mlx"
                print(f"[Core] Initialized Apple Silicon (MPS) backend using MLX with: {self.model_path}")
            except ImportError:
                raise ImportError("mlx_whisper is not installed. Run: pip install mlx-whisper")
                
        elif self.device_backend == "cuda":
            try:
                from faster_whisper import WhisperModel

                self.whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
                print(f"[Core] Initialized NVIDIA GPU (CUDA) backend with size: {model_size}")
            except ImportError:
                raise ImportError("faster_whisper is not installed. Run: pip install faster-whisper")
        else:
            raise ValueError("Unsupported backend! Choose either 'mps' or 'cuda'.")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z'\s]", "", text)  
        return text.split()

    def _train_model(self, dataset_path):
        df = pd.read_csv(dataset_path)
        texts = [self.preprocess_text(text) for text in df.iloc[:, 0]]
        labels_list = df.iloc[:, 1].tolist()
        
        X = self.vectorizer.fit_transform(texts)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, labels_list, test_size=0.3, random_state=42, stratify=labels_list
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        best_accuracy = -1
        for c in np.logspace(-4, 2, 7):
            model = LogisticRegression(C=c, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_val, y_val)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                
        print(f"Validation accuracy: {best_accuracy:.4f} | Test accuracy: {self.best_model.score(X_test, y_test):.4f}")

    def predict(self, text):
        processed_text = self.preprocess_text(text)
        x = self.vectorizer.transform([processed_text])
        probs = self.best_model.predict_proba(x)[0]
        return self.best_model.classes_[probs.argmax()], probs.max()

    def correct_transcription(self, text):
        return text.lower()

    def audio_to_numpy(self, audio):
        wav_data = audio.get_wav_data()
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0
        return audio_np

    def _calibrate_mic(self, duration=1):
        with self.mic as source:
            print("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)

    def transcribe_audio(self, audio_np):
        if self.device_backend == "mps":
            result = self.mlx_whisper.transcribe(
                audio_np,
                path_or_hf_repo=self.model_path,
                language="en",
                initial_prompt=self.initial_prompt
            )
            return result["text"].strip()
            
        elif self.device_backend == "cuda":
            segments, _ = self.whisper_model.transcribe(
                audio_np, 
                language="en", 
                initial_prompt=self.initial_prompt
            )
            return "".join([segment.text for segment in segments]).strip()

    def audio_callback(self, recognizer, audio):
        try:
            start_time = time.time()
            audio_np = self.audio_to_numpy(audio)

            raw_text = self.transcribe_audio(audio_np)
            transcribed_text = self.correct_transcription(raw_text)
            
            if not transcribed_text or len(transcribed_text.split()) > 15:
                return

            intent, confidence = self.predict(transcribed_text)
            print(f"Transcript: {transcribed_text} | Intent: {intent} ({confidence:.2f}) | Latency: {time.time() - start_time:.4f}s")
            
            if confidence > 0.5:
                self.update_state(intent)
        except Exception as e:
            print(f"Audio Callback Error: {e}")

    def start_listening(self, phrase_time_limit=0.7):
        return self.recognizer.listen_in_background(
            self.mic, self.audio_callback, phrase_time_limit=phrase_time_limit
        )

    def update_state(self, intent):
        raise NotImplementedError("Subclasses must implement `update_state(self, intent)`")

    def get_current_action(self):
        raise NotImplementedError("Subclasses must implement `get_current_action(self)`")
    
class MarioVoiceController(BaseVoiceController):
    def __init__(self, dataset_path, device_backend="mps"):

        prompt = "Mario alternative commands: jump, right, left, duck, stop."

        super().__init__(
            dataset_path=dataset_path, 
            device_backend=device_backend, 
            model_size="tiny.en", 
            initial_prompt=prompt
        )
        
        self.current_direction = None
        self.jump_until = 0.0
        self.duck_until = 0.0

    def correct_transcription(self, text):
        replacements = {
            "dump": "jump",
            "jum": "jump",
            "jumps": "jump",
            "jumping": "jump",
            "chump": "jump",
            "jumpp": "jump",
            "john": "jump",
            "joe": "jump",
            "write": "write"
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

    # need to be overridden
    def update_state(self, intent):
        with self.lock:
            now = time.time()
            if intent == "MOVE_LEFT":
                self.current_direction = "left"
            elif intent == "MOVE_RIGHT":
                self.current_direction = "right"
            elif intent == "JUMP":
                self.jump_until = now + 0.4 
            elif intent == "DUCK":
                self.duck_until = now + 1.0
            elif intent == "STOP":
                self.current_direction = None

    # need to be overridden
    def get_current_action(self):
        buttons = []

        with self.lock:
            now = time.time()
            
            if self.current_direction == "left":
                buttons.append("left")
            elif self.current_direction == "right":
                buttons.append("right")
                
            if now < self.jump_until:
                buttons.append("A")
            if now < self.duck_until:
                buttons.append("down")
        
        if not buttons:
            buttons = ["NOOP"]
            
        return COMPLEX_MOVEMENT.index(buttons)

if __name__ == "__main__":
    controller = MarioVoiceController(dataset_path="actual_dataset.csv", device_backend="mps")
    
    env = gym.make("SuperMarioBros-v0", render_mode="rgb_array")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    obs, info = env.reset()
    
    cv2.namedWindow("Mario", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mario", 1600, 800)
    
    stop_listening_fn = controller.start_listening()
    
    print("New Voice Controller active. Press 'q' to exit.")
    try:
        while True:
            action = controller.get_current_action()
            obs, reward, terminated, truncated, info = env.step(action)
            
            cv2.imshow("Mario", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()
        cv2.destroyAllWindows()