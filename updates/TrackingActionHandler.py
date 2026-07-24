import io
import json
import logging
import threading
import time
import wave
import inspect

import cv2
import numpy as np
import pynput
import pyaudio
import sklearn
import transformers

import faster_whisper
import mlx_whisper
import stable_retro

from AutoTracking import AutoTracking
from ActionFinder import ActionFinder

class TrackingActionHandler:
    def __init__(self, env, json_path):
        self.data = self.read_json_file(json_path)

        self.env = env

        self.track_distance_lookup = {
            name: int(info["tracking"]["track_distance"], 16)
            for name, info in self.data["targets"].items()
            if name != "enemy"
        }

        self.auto_tracking_class = AutoTracking(json_path)

        self.target_profiles = self.auto_tracking_class.get_distances_to_targets(env, self.auto_tracking_class.get_game_positions(env))

        self.function_dictionary = ActionFinder.__get_function_dict()

    def go_to_target(self, target_profiles, left_index, right_index, action):
        target_type = self.target_profiles["enemy_type"]

        candidates = [
            {"name": name, **info}
            for name, info in target_profiles.items()
            if name == target_type
        ]

        if not candidates:
            self.auto_tracking_class.deactivate_tracking()
            action = np.zeros(self.env.action_space.shape, dtype=int)

        target = self.auto_tracking_class.pick_target(candidates)

        horizontal_distance = self.target_profiles["horizontal_distance"]
        tracking_distance = int(self.data[target_type]["tracking"]["track_distance"], 16)

        function_name = self.data[target_type]["tracking"]["action"]

        if horizontal_distance >= 0: # to right
            if abs(horizontal_distance) != tracking_distance:
                action[right_index] = 1
            else:
                ActionFinder.__getattr__(function_name)
                action[right_index] = 0
        else: # to left
            if abs(horizontal_distance) != tracking_distance:
                action[left_index] = 1
            else:
                ActionFinder.__getattr__(function_name)
                action[left_index] = 0

    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)

        return data