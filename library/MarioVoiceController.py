from NESVoiceController import NESVoiceController
import logging

logger = logging.getLogger("MarioVoiceController")

class MarioVoiceController(NESVoiceController):
    def __init__(self, mapping_json_path, auto_tracking_class, device_backend="mps", whisper_model_size="tiny.en", initial_prompt=None):
        super().__init__(mapping_json_path=mapping_json_path, device_backend=device_backend, whisper_model_size=whisper_model_size, initial_prompt=initial_prompt)

        self.auto_tracking_class = auto_tracking_class

    def process_game_commands(self, entities):
        with self.lock:

            # combining found entities from NESBERT into single sentence
            sentence = " "

            for entity in entities:
                word = entity.get("word")
                cleaned_word = word.strip().lower()

                sentence.join(cleaned_word)

            # if no entities recognized/no sentence, do nothing
            if not sentence:
                return 
            
            if sentence == "any" or sentence == "anything" or sentence == "any enemy" or sentence == "closest":
                self.auto_tracking_class.activate()
                self.auto_tracking_class.target_type_address = None
                self.auto_tracking_class.target_type_name = None
                logger.info("Auto tracker enabled for any enemy")
                
                return

            name, score = self.auto_tracking_class.activate_set_target(sentence)

            if name:
                logger.info(f"Auto tracking started on {name} with a confidence of {score:.2f})")
            else:
                logger.info(f"Deactivating auto tracking, no recognized target in: {sentence}")
                self.auto_tracking_class.deactivate()