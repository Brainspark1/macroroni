"""
nes_voice_control.py
=====================

Everything combined into one file:

  SemanticMapper        - TF-IDF matching of a spoken phrase to a target name
  AutoEnemyTracking     - NES RAM reading + auto-jump control logic
  NESVoiceController    - mic capture -> Whisper transcription -> BERT NER
  MarioVoiceController  - glue class: voice entities -> AutoEnemyTracking
  GameRunner            - the gym_super_mario_bros game loop

Data flow:

    mic audio
       |
       v
    NESVoiceController        (Whisper transcription -> BERT NER entities)
       |
       v
    MarioVoiceController       <- previously missing glue class
       |  turns entities into a phrase, calls AutoEnemyTracking
       v
    AutoEnemyTracking
       |  uses SemanticMapper to resolve phrase -> target name -> memory address
       v
    SemanticMapper
       |
       v
    GameRunner
       runs the gym loop every frame, calling auto_enemy_tracking.get_action()
       to produce NES controller input

Bug fixed vs. the original SemanticMapper: `target_names` used to include
"enemy" while `descriptions` filtered it out, silently misaligning the two
lists' indices unless "enemy" happened to be the last key in the JSON.
Both lists are now built from the same filtered iteration.

Install (pick the whisper backend that matches your hardware):

    pip install numpy opencv-python pynput SpeechRecognition transformers \
                scikit-learn gym-super-mario-bros nes-py
    pip install mlx-whisper       # Apple Silicon (device_backend="mps")
    pip install faster-whisper    # Nvidia GPU   (device_backend="cuda")

Run:

    python nes_voice_control.py --json path/to/json_file.json --device mps
    python nes_voice_control.py --json path/to/json_file.json --no-voice   # manual typed-target mode
"""

import argparse
import io
import json
import logging
import threading
import time
import wave

import cv2
import numpy as np
import speech_recognition as sr
from pynput import keyboard
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

logger = logging.getLogger("nes_voice_control")


# ---------------------------------------------------------------------------
# SemanticMapper
# ---------------------------------------------------------------------------

class SemanticMapper:
    """
    Matches a spoken transcript sentence to the closest target name using
    TF-IDF + cosine similarity against each target's "description" field
    in the shared game-mapping JSON file (the same file AutoEnemyTracking
    reads its memory addresses from).
    """

    def __init__(self, json_path):
        self.json_data = self.read_json_file(json_path)

    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        return data

    def find_max_similarity(self, transcript_sentence):
        targets_data = self.json_data["targets"]

        # FIX: both target_names and descriptions must be built from the
        # SAME filtered iteration, otherwise their indices drift apart
        # whenever "enemy" isn't the last key in the dict.
        target_names = [name for name in targets_data.keys() if name != "enemy"]
        descriptions = [targets_data[name]["description"] for name in target_names]

        if not descriptions:
            return None, 0.0

        all_documents = [transcript_sentence] + descriptions

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_documents)

        target_vector = tfidf_matrix[0:1]
        description_vectors = tfidf_matrix[1:]
        similarity_scores = cosine_similarity(target_vector, description_vectors)[0]

        max_idx = int(np.argmax(similarity_scores))
        max_score = float(similarity_scores[max_idx])
        max_target_name = target_names[max_idx]

        return max_target_name, max_score


# ---------------------------------------------------------------------------
# AutoEnemyTracking
# ---------------------------------------------------------------------------

class AutoEnemyTracking:
    """
    Reads Mario/enemy positions from NES RAM and computes the controller
    action needed to approach and jump on the currently targeted enemy
    type. The target type itself is set externally (e.g. by
    MarioVoiceController) via `activate_set_target()`.
    """

    def __init__(
        self,
        json_path,
        max_x_distance=15,
        max_kill_x=22,
        frames_decide_run=20,
        mario_approach_speed_fallback=2.0,
        min_hold_frames=3,
        max_hold_frames=14,
    ):
        self.active = False
        self.target_slot = None
        self.jump_direction = None
        self.recovery_frames_left = 0
        self.total_recovery_frames = 0
        self.jump_hold_frames = 0
        self.starting_score = None
        self.awaiting_kill_confirmation = False

        self.data = self.read_json_file(json_path)
        self.character_data = self.data["characters"]
        self.enemy_data = self.data["targets"]["enemy"]
        self.item_data = self.data["items"]
        self.env_data = self.data["environment"]

        self.target_type_lookup = {
            name: int(info["address"], 16)
            for name, info in self.data["targets"].items()
            if name != "enemy"
        }

        self.semantic_mapper = SemanticMapper(json_path=json_path)
        self.target_type_address = None
        self.target_type_name = None

        self.max_x_distance = max_x_distance
        self.max_x_kill = max_kill_x
        self.frames_decide_run = frames_decide_run
        self.mario_approach_speed_fallback = mario_approach_speed_fallback
        self.min_hold_frames = min_hold_frames
        self.max_hold_frames = max_hold_frames

        (
            self.character_absolute_page_number,
            self.character_vertical_screen_position,
            self.character_horizontal_position,
        ) = self.initialize_character_variables()

        (
            self.enemy_active_state,
            self.enemy_type_address,
            self.enemy_horizontal_velocity,
            self.enemy_absolute_map_num,
            self.enemy_horizontal_page_position,
            self.enemy_vertical_screen_position,
        ) = self.initialize_enemy_variables()

        self.initialize_item_veriables()

        (
            self.env_time_hundred,
            self.env_time_ten,
            self.env_time_one,
        ) = self.initialize_environment_variables()

    def initialize_character_variables(self):
        character_absolute_page_number = int(self.character_data["absolute_page_number"], 16)
        character_vertical_screen_position = int(self.character_data["vertical_screen_position"], 16)
        character_horizontal_position = int(self.character_data["horizontal_position"], 16)

        return character_absolute_page_number, character_vertical_screen_position, character_horizontal_position

    def initialize_enemy_variables(self):
        enemy_active_state = int(self.enemy_data["active_state"], 16)
        enemy_type_address = int(self.enemy_data["enemy_type"], 16)
        enemy_horizontal_velocity = int(self.enemy_data["horizontal_velocity"], 16)
        enemy_absolute_map_num = int(self.enemy_data["absolute_map_num"], 16)
        enemy_horizontal_page_position = int(self.enemy_data["horizontal_page_position"], 16)
        enemy_vertical_screen_position = int(self.enemy_data["enemy_vertical_screen_position"], 16)

        return (
            enemy_active_state,
            enemy_type_address,
            enemy_horizontal_velocity,
            enemy_absolute_map_num,
            enemy_horizontal_page_position,
            enemy_vertical_screen_position,
        )

    def initialize_item_veriables(self):
        pass

    def initialize_environment_variables(self):
        env_time_hundred = int(self.env_data["time_hundred"], 16)
        env_time_ten = int(self.env_data["time_ten"], 16)
        env_time_one = int(self.env_data["time_one"], 16)

        return env_time_hundred, env_time_ten, env_time_one

    def activate(self):
        self.active = True
        self.target_slot = None
        self.jump_direction = None
        self.recovery_frames_left = 0
        self.total_recovery_frames = 0
        self.jump_hold_frames = 0
        self.starting_score = None
        self.awaiting_kill_confirmation = False
        self.target_type_address = None
        self.target_type_name = None

    def deactivate(self):
        self.active = False
        self.target_slot = None
        self.jump_direction = None
        self.recovery_frames_left = 0
        self.total_recovery_frames = 0
        self.jump_hold_frames = 0
        self.starting_score = None
        self.awaiting_kill_confirmation = False
        self.target_type_address = None
        self.target_type_name = None

    def set_target_from_similarity(self, transcript_sentence, min_confidence=0.2):
        name, score = self.semantic_mapper.find_max_similarity(transcript_sentence)

        if name is None or score < min_confidence:
            self.target_type_address = None
            self.target_type_name = None
            return None, score

        self.target_type_address = self.target_type_lookup.get(name)
        self.target_type_name = name
        return name, score

    def activate_set_target(self, transcript_sentence):
        self.activate()
        return self.set_target_from_similarity(transcript_sentence)

    def pick_target(self, enemy_profiles):
        closest_enemy = enemy_profiles[0]
        closest_time = closest_enemy.get("time_to_collision_frames", float("inf"))

        for enemy in enemy_profiles[1:]:
            enemy_time = enemy.get("time_to_collision_frames", float("inf"))
            if enemy_time < closest_time:
                closest_enemy = enemy
                closest_time = enemy_time

        return closest_enemy

    def calculate_jump_arc(self, horizontal_distance, enemy):
        approach_speed = self.mario_approach_speed_fallback

        if enemy.get("is_coming_towards"):
            approach_speed = approach_speed + enemy.get("speed_per_frame", 0)

        if approach_speed <= 0:
            approach_speed = self.mario_approach_speed_fallback

        time_to_collision = abs(horizontal_distance) / approach_speed

        hold_jump_frames = round(time_to_collision / 2)
        hold_jump_frames = max(self.min_hold_frames, min(self.max_hold_frames, hold_jump_frames))

        total_frames = hold_jump_frames * 2

        return hold_jump_frames, total_frames

    def get_action(self, enemy_profiles, current_score):
        if self.recovery_frames_left > 0 and self.jump_direction:
            self.recovery_frames_left -= 1
            frames_elapsed = self.total_recovery_frames - self.recovery_frames_left

            if frames_elapsed <= self.jump_hold_frames:
                return COMPLEX_MOVEMENT.index([self.jump_direction, "A"])
            else:
                if self.recovery_frames_left == 0:
                    self.awaiting_kill_confirmation = True
                return COMPLEX_MOVEMENT.index([self.jump_direction])

        if not enemy_profiles:
            self.deactivate()
            return COMPLEX_MOVEMENT.index(["NOOP"])

        if self.target_type_address is not None:
            candidates = [enemy for enemy in enemy_profiles if enemy.get("enemy_type") == self.target_type_address]
        else:
            candidates = enemy_profiles

        if not candidates:
            return COMPLEX_MOVEMENT.index(["NOOP"])

        target = self.pick_target(enemy_profiles)
        self.target_slot = target["enemy_slot_number"]
        horizontal_distance = target["horizontal_distance"]
        time_to_collision = target["time_to_collision_frames"]

        if horizontal_distance >= 0:
            desired_run_action = ["right", "B"]
        else:
            desired_run_action = ["left", "B"]

        kill_attempt = (
            abs(horizontal_distance) <= self.max_x_kill
        ) or (
            abs(horizontal_distance) <= self.max_x_kill * 2
            and time_to_collision <= self.frames_decide_run
        )

        if kill_attempt:
            if horizontal_distance >= 0:
                self.jump_direction = "right"
            else:
                self.jump_direction = "left"

            self.jump_hold_frames, self.total_recovery_frames = self.calculate_jump_arc(horizontal_distance, target)
            self.recovery_frames_left = self.total_recovery_frames
            self.starting_score = current_score

            return COMPLEX_MOVEMENT.index([self.jump_direction, "A"])

        if abs(horizontal_distance) > self.max_x_distance:
            return COMPLEX_MOVEMENT.index(desired_run_action)

        if horizontal_distance >= 0:
            return COMPLEX_MOVEMENT.index(["right"])
        else:
            return COMPLEX_MOVEMENT.index(["left"])

    def confirming_stopping_kill(self, current_score):
        if not self.awaiting_kill_confirmation:
            return False

        self.awaiting_kill_confirmation = False

        if self.starting_score is not None and current_score > self.starting_score:
            self.deactivate()
            return True

        return False

    def get_game_positions(self, env):
        ram = env.unwrapped.ram

        mario_x_page = int(ram[self.character_absolute_page_number])
        mario_x_screen = int(ram[self.character_horizontal_position])
        mario_x_pos = (mario_x_page * 256) + mario_x_screen
        mario_y_pos = int(ram[self.character_vertical_screen_position])

        enemy_slots = []

        for i in range(5):
            enemy_active = ram[self.enemy_active_state + i]

            if enemy_active:
                enemy_x_page = int(ram[self.enemy_absolute_map_num + i])
                enemy_x_screen = int(ram[self.enemy_horizontal_page_position + i])
                enemy_x_pos = (enemy_x_page * 256) + enemy_x_screen
                enemy_y_pos = int(ram[self.enemy_vertical_screen_position + i])
                enemy_type = int(ram[self.enemy_type_address + i])

                enemy_slots.append({
                    "slot": i,
                    "x": enemy_x_pos,
                    "y": enemy_y_pos,
                    "type": enemy_type,
                })

        return {
            "mario": {"x": mario_x_pos, "y": mario_y_pos},
            "enemies": enemy_slots,
        }

    def get_distances_to_enemies(self, env, positions):
        ram = env.unwrapped.ram

        enemy_metrics = []

        mario_x = positions["mario"]["x"]
        mario_y = positions["mario"]["y"]

        for enemy in positions["enemies"]:
            horizontal_distance = enemy["x"] - mario_x
            vertical_distance = enemy["y"] - mario_y
            distance = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5

            raw_speed_byte = int(ram[self.enemy_horizontal_velocity + enemy["slot"]])

            if raw_speed_byte > 128:
                enemy_direction = "left"
                enemy_speed = abs(256 - raw_speed_byte)
            else:
                enemy_direction = "right"
                enemy_speed = raw_speed_byte

            if enemy_speed == 0:
                enemy_speed = 1

            is_moving_towards_mario = False

            if horizontal_distance > 0 and enemy_direction == "left":
                is_moving_towards_mario = True
            elif horizontal_distance < 0 and enemy_direction == "right":
                is_moving_towards_mario = True

            if is_moving_towards_mario:
                time_to_collision_frames = abs(horizontal_distance) / enemy_speed
            else:
                time_to_collision_frames = float("inf")

            enemy_metrics.append({
                "enemy_slot_number": enemy["slot"],
                "enemy_type": enemy["type"],
                "horizontal_distance": horizontal_distance,
                "vertical_distance": vertical_distance,
                "distance": round(distance, 2),
                "speed_per_frame": enemy_speed,
                "is_coming_towards": is_moving_towards_mario,
                "time_to_collision_frames": time_to_collision_frames,
            })

        return enemy_metrics

    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)

        return data


# ---------------------------------------------------------------------------
# NESVoiceController
# ---------------------------------------------------------------------------

class NESVoiceController:
    """
    Base class handling: mic capture -> Whisper transcription -> BERT NER.

    This class is intentionally generic and NOT tied to any particular game.
    Subclasses (e.g. MarioVoiceController below) implement
    `process_game_commands` to translate extracted entities into actual
    game-controller behavior.
    """

    def __init__(self, mapping_json_path, device_backend="mps", whisper_model_size="tiny.en", initial_prompt=None):
        self.lock = threading.Lock()
        self.device_backend = device_backend
        self.initial_prompt = initial_prompt or "Make sure to listen for NES gameplay commands like \"jump on that enemy.\""

        self.game_mappings = {}
        self.load_game_mappings(mapping_json_path)

        self._init_transcription_engine(whisper_model_size)
        self._init_ner_pipeline("Saggarwal/token_bert")

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(sample_rate=16000)
        self._calibrate_mic()

    def load_game_mappings(self, json_path):
        try:
            with open(json_path, "r") as f:
                content = f.read()

            self.game_mappings = json.loads(content)
            logger.info(f"Successfully loaded game mappings from {json_path}")

        except Exception as e:
            logger.error(f"Failed to load game mappings: {e}")
            self.game_mappings = {}

    def _init_ner_pipeline(self, model_path):
        logger.info(f"Loading NESBERT Token Classification Model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        if self.device_backend == "cuda":
            device_str = "cuda:0"
        elif self.device_backend == "mps":
            try:
                self.model = self.model.to("mps")
                device_str = "mps"
            except Exception:
                logger.warning("Failed to move NESBERT to MPS. Defaulting to CPU.")
                device_str = "cpu"
        else:
            device_str = "cpu"

        self.nlp_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=device_str,
        )

    def _init_transcription_engine(self, model_size):
        if self.device_backend == "mps":
            try:
                import mlx_whisper

                self.mlx_whisper = mlx_whisper
                self.whisper_model_path = f"mlx-community/whisper-{model_size}-mlx"

                logger.info(f"Initialized MPS Whisper backend: {self.whisper_model_path}")

            except ImportError:
                raise ImportError("mlx_whisper is required for MPS. Run: pip install mlx-whisper")

        elif self.device_backend == "cuda":
            try:
                from faster_whisper import WhisperModel

                self.whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

                logger.info(f"Initialized CUDA Whisper backend (Size: {model_size})")

            except ImportError:
                raise ImportError("faster_whisper is required for CUDA. Run: pip install faster-whisper")
        else:
            raise ValueError("Unsupported backend model, choose either 'mps' or 'cuda'.")

    def _calibrate_mic(self, duration=2):
        with self.mic as source:
            logger.info("Calibrating microphone for ambient background noise...")

            self.recognizer.adjust_for_ambient_noise(source, duration=duration)

    def audio_to_numpy(self, audio):
        wav_data = audio.get_wav_data()
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0

        return audio_np

    def transcribe_audio(self, audio_np):
        if self.device_backend == "mps":
            result = self.mlx_whisper.transcribe(
                audio_np, path_or_hf_repo=self.whisper_model_path,
                language="en",
                initial_prompt=self.initial_prompt,
            )

            return result["text"].strip()

        elif self.device_backend == "cuda":
            segments, _ = self.whisper_model.transcribe(
                audio_np,
                language="en",
                initial_prompt=self.initial_prompt,
            )

            return "".join([segment.text for segment in segments]).strip()

    def extract_entities(self, text):
        if not text:
            return []

        return self.nlp_pipeline(text)

    def audio_callback(self, recognizer, audio):
        try:
            start_time = time.time()
            audio_np = self.audio_to_numpy(audio)
            raw_text = self.transcribe_audio(audio_np)

            if not raw_text or len(raw_text.split()) > 15:
                return

            raw_text = raw_text.lower()
            entities = self.extract_entities(raw_text)

            logger.info(f"Transcript: '{raw_text}' | NER Tags: {entities} | Latency: {time.time() - start_time:.4f}s")

            if entities:
                self.process_game_commands(entities)

        except Exception as e:
            logger.error(f"Audio Callback Error: {e}")

    def start_listening(self, phrase_time_limit=1.5):
        return self.recognizer.listen_in_background(self.mic, self.audio_callback, phrase_time_limit=phrase_time_limit)

    def process_game_commands(self, entities):
        raise NotImplementedError("Subclasses or game wrappers must implement `process_game_commands(self, entities)`")


# ---------------------------------------------------------------------------
# MarioVoiceController  (the previously-missing glue class)
# ---------------------------------------------------------------------------

_ANY_ENEMY_PHRASES = {"any", "anything", "any enemy", "enemies"}


class MarioVoiceController(NESVoiceController):
    """
    Connects voice to actual game action. NESVoiceController on its own only
    transcribes speech and tags entities; this subclass reassembles those
    entities into a phrase and hands it to AutoEnemyTracking (via
    SemanticMapper) to arm the auto-jump logic.
    """

    def __init__(
        self,
        mapping_json_path,
        auto_enemy_tracking,
        device_backend="mps",
        whisper_model_size="tiny.en",
        initial_prompt=None,
    ):
        super().__init__(
            mapping_json_path=mapping_json_path,
            device_backend=device_backend,
            whisper_model_size=whisper_model_size,
            initial_prompt=initial_prompt,
        )
        self.auto_enemy_tracking = auto_enemy_tracking

    def process_game_commands(self, entities):
        with self.lock:
            phrase = " ".join(ent.get("word", "") for ent in entities).strip().lower()

            if not phrase:
                return

            if phrase in _ANY_ENEMY_PHRASES:
                self.auto_enemy_tracking.activate()
                self.auto_enemy_tracking.target_type_address = None
                self.auto_enemy_tracking.target_type_name = None
                logger.info("Auto tracker enabled for any enemy")
                return

            name, score = self.auto_enemy_tracking.activate_set_target(phrase)

            if name:
                logger.info(f"Auto tracking started on '{name}' (confidence={score:.2f})")
            else:
                logger.info(f"No recognized target for phrase '{phrase}'; deactivating.")
                self.auto_enemy_tracking.deactivate()


# ---------------------------------------------------------------------------
# GameRunner
# ---------------------------------------------------------------------------

class GameRunner:
    """
    Runs the emulator loop, tying together manual keyboard movement,
    AutoEnemyTracking, and MarioVoiceController (running in the background
    via NESVoiceController.start_listening()).

    Set use_voice=False to fall back to typing a target name at the
    terminal (press 'e' then type) instead of using the microphone.
    """

    def __init__(
        self,
        mapping_json_path,
        device_backend="mps",
        whisper_model_size="tiny.en",
        use_voice=True,
        window_size=(1920, 1080),
    ):
        self.mapping_json_path = mapping_json_path
        self.use_voice = use_voice

        self.env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="rgb_array")
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        self.obs, self.info = self.env.reset()

        self.window_name = "Super Mario Bros"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *window_size)

        self.mario_action = 0
        self.move_left = False
        self.move_right = False
        self.move_down = False
        self.running = False
        self.jumping = False
        self.jump_start = 0
        self.awaiting_target_input = False

        self.auto_enemy_tracking = AutoEnemyTracking(json_path=mapping_json_path)

        self.voice_controller = None
        if self.use_voice:
            self.voice_controller = MarioVoiceController(
                mapping_json_path=mapping_json_path,
                auto_enemy_tracking=self.auto_enemy_tracking,
                device_backend=device_backend,
                whisper_model_size=whisper_model_size,
            )

        self._keyboard_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._keyboard_listener.daemon = True

    def _handle_manual_transcript(self, sentence):
        sentence = sentence.strip().lower()

        if not sentence:
            logger.info("No transcript, not starting auto tracking.")
            return

        if sentence == "any":
            self.auto_enemy_tracking.activate()
            self.auto_enemy_tracking.target_type_address = None
            self.auto_enemy_tracking.target_type_name = None
            logger.info("Auto tracker enabled, set to any enemy")
            return

        name, _ = self.auto_enemy_tracking.activate_set_target(sentence)

        if name:
            logger.info(f"Auto tracking started on {name}")
        else:
            logger.info("No recognized target to track.")
            self.auto_enemy_tracking.deactivate()

    def _manual_input_listener(self):
        while True:
            if self.awaiting_target_input:
                target_input = input("Enter target: ")
                self.awaiting_target_input = False
                self._handle_manual_transcript(target_input)
            else:
                time.sleep(0.3)

    def _on_press(self, key):
        try:
            if key.char.lower() == "a":
                self.move_left = True
            elif key.char.lower() == "d":
                self.move_right = True
            elif key.char.lower() == "f":
                self.running = True
            elif key.char.lower() == "s":
                self.move_down = True
            elif key.char.lower() == "e" and not self.use_voice:
                self.awaiting_target_input = True
        except AttributeError:
            pass

        if key == keyboard.Key.space and not self.jumping:
            self.jumping = True
            self.jump_start = time.time()

    def _on_release(self, key):
        try:
            if key.char.lower() == "a":
                self.move_left = False
            elif key.char.lower() == "d":
                self.move_right = False
            elif key.char.lower() == "f":
                self.running = False
            elif key.char.lower() == "s":
                self.move_down = False
        except AttributeError:
            pass

        if key == keyboard.Key.space:
            self.jumping = False

    def _compute_manual_action(self):
        if self.move_down:
            mario_action = COMPLEX_MOVEMENT.index(["down"])
        elif self.move_right and self.move_left:
            mario_action = COMPLEX_MOVEMENT.index(["NOOP"])
        elif self.move_right:
            mario_action = COMPLEX_MOVEMENT.index(["right", "B"] if self.running else ["right"])
        elif self.move_left:
            mario_action = COMPLEX_MOVEMENT.index(["left", "B"] if self.running else ["left"])
        else:
            mario_action = COMPLEX_MOVEMENT.index(["NOOP"])

        if self.jumping and (time.time() - self.jump_start) < 0.8:
            if mario_action == COMPLEX_MOVEMENT.index(["NOOP"]):
                mario_action = COMPLEX_MOVEMENT.index(["A"])
            elif mario_action == COMPLEX_MOVEMENT.index(["down"]):
                pass
            else:
                try:
                    base = ["right"] if self.move_right else ["left"]
                    if self.running:
                        base.append("B")
                    base.append("A")
                    mario_action = COMPLEX_MOVEMENT.index(base)
                except ValueError:
                    mario_action = COMPLEX_MOVEMENT.index(["A"])

        return mario_action

    def run(self):
        if self.use_voice:
            self.voice_controller.start_listening()
            logger.info("Voice control active. Say a target name (e.g. 'jump on that goomba') or 'any'.")
        else:
            transcript_thread = threading.Thread(target=self._manual_input_listener)
            transcript_thread.daemon = True
            transcript_thread.start()
            logger.info("Manual mode active. Press 'e' then type a target name at the terminal.")

        self._keyboard_listener.start()

        try:
            while True:
                key_cv2 = cv2.waitKey(1) & 0xFF
                if key_cv2 == ord("q"):
                    break

                if self.auto_enemy_tracking.active:
                    positions = self.auto_enemy_tracking.get_game_positions(self.env)
                    enemy_profiles = self.auto_enemy_tracking.get_distances_to_enemies(self.env, positions)
                    self.mario_action = self.auto_enemy_tracking.get_action(enemy_profiles, self.info.get("score", 0))
                else:
                    self.mario_action = self._compute_manual_action()

                self.obs, reward, terminated, truncated, self.info = self.env.step(self.mario_action)
                done = terminated or truncated

                if self.auto_enemy_tracking.active:
                    if self.auto_enemy_tracking.confirming_stopping_kill(self.info.get("score", 0)):
                        logger.info("Stopping auto tracking.")

                cv2.imshow(self.window_name, cv2.cvtColor(self.obs, cv2.COLOR_RGB2BGR))

                if done:
                    self.obs, self.info = self.env.reset()
                    self.move_left = False
                    self.move_right = False
                    self.move_down = False
                    self.running = False
                    self.jumping = False
                    self.auto_enemy_tracking.deactivate()

                # increase max time per level to 999
                self.env.unwrapped.ram[self.auto_enemy_tracking.env_time_hundred] = 0x09
                self.env.unwrapped.ram[self.auto_enemy_tracking.env_time_ten] = 0x09
                self.env.unwrapped.ram[self.auto_enemy_tracking.env_time_one] = 0x09
        finally:
            self.env.close()
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Voice-controlled NES auto-enemy-tracking")
    parser.add_argument("--json", required=True, help="Path to the game mapping json_file.json")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda"], help="Whisper/BERT hardware backend")
    parser.add_argument("--whisper-size", default="tiny.en", help="Whisper model size, e.g. tiny.en, base.en")
    parser.add_argument("--no-voice", action="store_true", help="Use typed-target mode instead of the microphone")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    runner = GameRunner(
        mapping_json_path=args.json,
        device_backend=args.device,
        whisper_model_size=args.whisper_size,
        use_voice=not args.no_voice,
    )
    runner.run()


if __name__ == "__main__":
    main()