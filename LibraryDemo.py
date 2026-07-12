from macroroni import BaseVoiceController

import time
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
import gymnasium as gym

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