import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2   # for the manual rendering of the game environment to see it
from pynput import keyboard    # For better keyboard controls
import threading
import time


env = gym.make('SuperMarioBros-v0', render_mode='rgb_array')  # For the Emulator Environment 
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs, info = env.reset()

window_name = 'Super Mario Bros'  # For Display Size
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.resizeWindow(window_name, 1600, 800)

mario_action = 0       # temporary Key controls 
jumping = False        # better adjustment for jumping height 
jump_start = 0 

def on_press(key):
    global mario_action, jumping, jump_start
    try:
        if key.char == 'a' or key.char == 'A':
            mario_action = COMPLEX_MOVEMENT.index(['left'])
        elif key.char == 'd' or key.char == 'D':
            mario_action = COMPLEX_MOVEMENT.index(['right'])
        elif key.char == 's' or key.char == 'S':
            mario_action = COMPLEX_MOVEMENT.index(['down'])
        elif key.char == 'z' or key.char == 'Z':
            mario_action = COMPLEX_MOVEMENT.index(['B'])
    except:
        pass

    if key == keyboard.Key.space:
        if not jumping:
            jumping = True
            jump_start = time.time()
            mario_action = COMPLEX_MOVEMENT.index(['A'])
            

def key_release(key):
    global mario_action, jumping
    if key == keyboard.Key.esc:
        return False
    if key == keyboard.Key.space:
        jumping = False
    if hasattr(key, 'char') and key.char in ['a', 'A', 'd', 'D']:
        mario_action = 0

listener = keyboard.Listener(on_press=on_press, on_release=key_release)
listener.daemon = True
listener.start()

while True:
    key = cv2.waitKey(1) & 0xFF
    action = mario_action

    # this lets you hold space to jump higher but its not really smooth
    if jumping and (time.time() - jump_start) < 0.8:  #jump duration
        action = COMPLEX_MOVEMENT.index(['A'])

    if key == ord('q'):
        break
    elif key == ord(' '):
        action = COMPLEX_MOVEMENT.index(['A'])

    movement = COMPLEX_MOVEMENT[action]
    print(f"Action: {action} | Movement: {movement}", end="\r")

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    cv2.imshow(window_name, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if done:
        obs, info = env.reset()
env.close()
cv2.destroyAllWindows()
