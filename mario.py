import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2

import gym_super_mario_bros  # registers environments

env = gym.make('SuperMarioBros-v0', render_mode='rgb_array')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs, info = env.reset()


WINDOW_SCALE = 5
cv2.namedWindow('Super Mario Bros', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Super Mario Bros', 256 * WINDOW_SCALE, 240 * WINDOW_SCALE)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):          # Quit
        break
    elif key == ord(' '):        # Space = Jump
        action = COMPLEX_MOVEMENT.index(['A'])          # Jump
    elif key == 0xFF00 + 81:     # Right arrow
        action = COMPLEX_MOVEMENT.index(['RIGHT'])
    elif key == 0xFF00 + 82:     # Left arrow  (adjust if needed)
        action = COMPLEX_MOVEMENT.index(['LEFT'])
    elif key == 0xFF00 + 84:     # Down
        action = COMPLEX_MOVEMENT.index(['DOWN'])
    elif key == ord('z') or key == ord('Z'):
        action = COMPLEX_MOVEMENT.index(['B'])          # Run / Attack
    else:
        action = 0  # NOOP (do nothing)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    cv2.imshow('Super Mario Bros - Keyboard Control', 
               cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    
    if done:
        obs, info = env.reset()
        print("Level restarted!")

env.close()
cv2.destroyAllWindows()