from NESVoiceController import NESVoiceController
import logging

logger = logging.getLogger("MarioVoiceController")


class MarioVoiceController(NESVoiceController):
    def __init__(
        self,
        mapping_json_path,
        auto_tracking_class,
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

        self.auto_tracking_class = auto_tracking_class

        self.pass_movement = False
        self.current_action = None

    def process_game_commands(self, entities):
        with self.lock:
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

            sentence = None

            if target_words:
                sentence = " ".join(target_words)
            else:
                if action_words:
                    action_sentence = " ".join(action_words)

                    if "##" in action_sentence:
                        action_sentence = action_sentence.replace("#", "").replace(
                            " ", ""
                        )

                    print(f"Passing action to semantic mapper: {action_sentence}")

                    action_name, score = (
                        self.auto_tracking_class.set_action_from_similarity(
                            action_sentence
                        )
                    )

                    if action_name:
                        # getting action mode from JSON file
                        action_info = self.game_mappings.get("actions", {}).get(
                            action_name, {}
                        )  # default getting empty action dictionary
                        mode = action_info.get("mode", "once")  # default to one press

                        # if user wants to stop the mode action,
                        if mode == "stop":
                            print("Stop command received, cancelling all mode actions")
                            self.auto_tracking_class.deactivate()
                            self.auto_tracking_class.passing_action = False
                            self.auto_tracking_class.action_type_name = None
                            self.auto_tracking_class.action_mode = "once"
                            self.auto_tracking_class.action_hold_frames = 0
                            return

                        self.auto_tracking_class.set_action_mode(mode)

                        print(
                            f"Resolved action: '{action_name}' (confidence: {score:.2f}) with mode: {mode}"
                        )
                    else:
                        print(f"Could not resolve action from: {action_sentence}")

                    return

            # if no entities recognized/no sentence, do nothing - moving up to fix Audio Callback Error bug
            if sentence is None or not sentence or not target_words:
                return

            if "##" in sentence:
                sentence = sentence.replace("#", "").replace(" ", "")

            print(f"DEBUG raw entities: {entities}")
            print(f"DEBUG phrase passed to mapper: '{sentence}'")

            if (
                sentence == "any"
                or sentence == "anything"
                or sentence == "any enemy"
                or sentence == "closest"
            ):
                self.auto_tracking_class.activate()
                self.auto_tracking_class.target_type_address = None
                self.auto_tracking_class.target_type_name = None
                logger.info("Auto tracker enabled for any enemy")

                return

            name, score = self.auto_tracking_class.activate_set_target(sentence)

            if name:
                # if name contains powerup ending or is a coin (is a powerup)
                if "_powerup" in name or name == "coin":
                    self.auto_tracking_class.powerup_active = True
                    logger.info(f"Powerup tracking started on {name}")
                else:
                    self.auto_tracking_class.powerup_active = False
                    logger.info(f"Enemy tracking started on {name}")
            else:
                logger.info(
                    f"Deactivating auto tracking, no recognized target in: {sentence}"
                )
                self.auto_tracking_class.deactivate()
