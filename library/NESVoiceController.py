import io
import json
import logging
import threading
import time
import wave
import numpy as np
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logger = logging.getLogger("nes_voice")

class NESVoiceController:
    def __init__(self, mapping_json_path, device_backend="mps", model_size="tiny.en", initial_prompt=None):
        """        
        :param mapping_json_path: path to the game-specific json file containing character/item/target memory addresses
        :param device_backend: hardware backend currently found in user's computer (write either "mps" for MacOS, "cuda" for Nvidia, or "cpu")
        :param model_size: size variant for OpenAI Whisper core (defaulted to tiny English model for minimal latency issues)
        :param initial_prompt: context prompt to guide Whisper transcriptions
        """

        self.lock = threading.Lock()
        self.device_backend = device_backend
        self.initial_prompt = initial_prompt or "Make sure to listen for NES gameplay commands like \"jump on that enemy.\""
        
        # loading game-specific memory mappings from json file
        self.game_mappings = {}
        self.load_game_mappings(mapping_json_path)

        # initializing transcription and ner pipelines
        self._init_transcription_engine(model_size)
        self._init_ner_pipeline("Saggarwal/token_bert") # passing in bert model
        
        # setting up classes to capture audio
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(sample_rate=16000) # fixed recording rate at 16 kHz
        self._calibrate_mic()

    # method to load and parse the json file that maps objects to memory addresses
    def load_game_mappings(self, json_path):
        try:
            with open(json_path, 'r') as f:
                content = f.read()

            self.game_mappings = json.loads(content)
            logger.info(f"Successfully loaded game mappings from {json_path}")

        except Exception as e:
            logger.error(f"Failed to load game mappings: {e}")
            self.game_mappings = {}

    # method to initialize the tokenizer and model pipeline with the passed in model path (Saggarwal/token_bert before sarthak builds the next bert)
    def _init_ner_pipeline(self, model_path):
        logger.info(f"Loading NESBERT Token Classification Model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # checking hardware components found
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

        # loading the pipeline
        self.nlp_pipeline = pipeline(
            "token-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple",
            device=device_str
        )

    # method to set up whisper engine based on hardware availability
    def _init_transcription_engine(self, model_size):
        if self.device_backend == "mps":
            try:
                import mlx_whisper

                self.mlx_whisper = mlx_whisper
                self.whisper_model_path = f"mlx-community/whisper-{model_size}-mlx"

                logger.info(f"Initialized MPS Whisper backend: {self.whisper_model_path}")

            # error if mlx_whisper not imported
            except ImportError:
                raise ImportError("mlx_whisper is required for MPS. Run: pip install mlx-whisper")
        
        elif self.device_backend == "cuda":
            try:
                from faster_whisper import WhisperModel

                self.whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

                logger.info(f"Initialized CUDA Whisper backend (Size: {model_size})")

            # error if faster_whisper not imported
            except ImportError:
                raise ImportError("faster_whisper is required for CUDA. Run: pip install faster-whisper")
        else:
            raise ValueError("Unsupported backend model, choose either 'mps' or 'cuda'.")

    # method to calibrate microphone based on background noise
    def _calibrate_mic(self, duration=2):
        with self.mic as source:
            logger.info("Calibrating microphone for ambient background noise...")

            self.recognizer.adjust_for_ambient_noise(source, duration=duration)

    # method to store temporary wav file in numpy array to minimize latency
    def audio_to_numpy(self, audio):
        wav_data = audio.get_wav_data()
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0

        return audio_np

    # method to transcribe audio using loaded whisper model
    def transcribe_audio(self, audio_np):
        if self.device_backend == "mps":
            result = self.mlx_whisper.transcribe(
                audio_np, path_or_hf_repo=self.whisper_model_path, 
                language="en", 
                initial_prompt=self.initial_prompt
            )

            # returning text part of transcription
            return result["text"].strip()
        
        elif self.device_backend == "cuda":
            segments, _ = self.whisper_model.transcribe( # only parsing the segments, discarding everything else using _ operator in explosion
                audio_np, 
                language="en",
                initial_prompt=self.initial_prompt
            )

            # joining collected segments together into transcribed sentence
            return "".join([segment.text for segment in segments]).strip()

    # method to pass text through NESBERT pipeline to extract tags/entity names
    def extract_entities(self, text):

        # return nothing if no text to pass through the bert
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
            
            # if entities have been recognized, reference them with json mapping into memory adddresses for bert to use
            if entities:
                self.process_game_commands(entities)

        except Exception as e:
            logger.error(f"Audio Callback Error: {e}")

    def start_listening(self, phrase_time_limit=1.5): 
        return self.recognizer.listen_in_background(self.mic, self.audio_callback, phrase_time_limit=phrase_time_limit)

    # method must be overriden by developers for emulator loop to translate extracted entities into use case/game controls
    def process_game_commands(self, entities):

        # raise error if not implemented
        raise NotImplementedError("Subclasses or game wrappers must implement `process_game_commands(self, entities)`")
