import io
import json
import logging
import threading
import time
import wave

import cv2
import numpy as np
import pynput
import pyaudio
import sklearn
import transformers

import faster_whisper
import mlx_whisper
import logging

from NESVoiceController import NESVoiceController

logger = logging.getLogger("ActionHandler")

class ManualActionHandler(NESVoiceController):
    def __init__(self, mapping_json_path, auto_tracking_class, device_backend="mps", whisper_model_size="tiny.en", initial_prompt=None):
        super().__init__(mapping_json_path=mapping_json_path, device_backend=device_backend, whisper_model_size=whisper_model_size, initial_prompt=initial_prompt)

        self.data = self.read_json_file(mapping_json_path)
        self.auto_tracking_class = auto_tracking_class

        self.action_duration_lookup = {
            name: info["duration"]
            for name, info in self.data["actions"].items()
            if info["duration"] != None
        }

        self.action_button_lookup = {
            name: info["button"]
            for name, info in self.data["actions"].items()
        }
        
        self.action_array = []
        self.duration_array = [0, 0, 0, 0, 0, 0, 0, 0] # b, a, mode, start, up, down, left, right
        self.button2id = {
            "b": 0,
            "a": 1,
            "mode": 2,
            "start": 3,
            "up": 4,
            "down": 5,
            "left": 6,
            "right": 7
        }

    def process_game_commands(self, entities, actions_array):
        sentence = None

        # combining found entities from NESBERT into single sentence
        target_words = [
            entity.get("word", "").strip().lower()
            for entity in entities
            if entity.get("entity_group") == "TARGET"
        ]

        action_words = [
            entity.get("word", "").strip().lower()
            for entity in entities
            if entity.get("entity_group") == "ACTION"
        ]

        if target_words:
            sentence = " ".join(target_words)
        else:
            if action_words:
                action_sentence = " ".join(action_words)

                if "##" in action_sentence:
                    action_sentence = action_sentence.replace("#", "").replace(" ", "")

                logger.info(f"Passing action to semantic mapper: {action_sentence}")
            
                action_name, score = self.set_action_from_similarity(action_sentence)

                if action_name:
                    if action_name.lower() == "stop":
                        logger.info("Stop command received, cancelling all mode actions")
                        self.auto_tracking_class.deactivate()

                        return

                    action_duration = self.action_duration_lookup[action_name]
                    action_buttons = self.action_button_lookup[action_name]
                    
                    for button in action_buttons:
                        self.duration_array[self.button2id[button]] = action_duration
                        actions_array[self.button2id[button]] = 1

                    print(f"Resolved action: '{action_name}' (confidence: {score:.2f})")
                else:
                    print(f"Could not resolve action from: {action_sentence}")

                self.action_array = actions_array

                return

    def set_action_from_similarity(self, transcript_sentence, min_confidence=0.2):
        name, score = self.semantic_mapper.find_max_action_similarity(transcript_sentence)

        if score < min_confidence:
            return None, score

        return name, score

    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)

        return data

    def get_action_array(self):
        return self.action_array
    
    def get_duration_array(self):
        return self.duration_array