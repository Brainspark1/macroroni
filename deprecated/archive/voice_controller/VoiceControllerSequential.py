import io
import re
import threading
import time
import wave

import cv2
import gymnasium as gym
import mlx_whisper
import numpy as np
import pandas as pd
import speech_recognition as sr
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from transformers import pipeline

class VoiceControllerSequential:
    def __init__(self, dataset_path="actual_dataset.csv"):
        self.lock = threading.Lock()

        # movement states
        self.current_direction = None
        self.running = False
        self.jump_until = 0.0
        self.duck_until = 0.0
        self.fire_until = 0.0

        # speech recognition setup
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(sample_rate=16000)
        self._calibrate_mic()
        

text = input('Type something: ')
classifier = pipeline("ner", model="Saggarwal/token_bert")
print(classifier(text))