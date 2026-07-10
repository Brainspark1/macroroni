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

cv2.resizeWindow(window_name, 1920, 1080)

# control additions to fix jumping and running 
mario_action = 0        
move_left = False
move_right = False
running = False
jumping = False        # better adjustment for jumping height 
jump_start = 0 

def on_press(key):
    global move_left, move_right, running, jumping, jump_start, mario_action
    try:
        if key.char.lower() == 'a':
            move_left = True
        elif key.char.lower() == 'd':
            move_right = True
        elif key.char.lower() == 'z':
            running = True
        elif key.char.lower() == 's':
            mario_action = COMPLEX_MOVEMENT.index(['down'])
    except:
        pass

    if key == keyboard.Key.space and not jumping:
        jumping = True
        jump_start = time.time()
            
            

def on_release(key):
    global move_left, move_right, running, jumping, mario_action
    try:
        if key.char.lower() == 'a':
            move_left = False
        elif key.char.lower() == 'd':
            move_right = False
        elif key.char.lower() == 'z':
            running = False
        elif key.char.lower() == 's':
            mario_action = 0
    except:
        pass
    if key == keyboard.Key.space:
        jumping = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

while True:
    key_cv2 = cv2.waitKey(1) & 0xFF
    if key_cv2 == ord('q'):
        break
    
    # helps mario's movement while jumping
    if move_right and move_left:
        mario_action = COMPLEX_MOVEMENT.index(['NOOP'])
    elif move_right:
        if running:
            mario_action = COMPLEX_MOVEMENT.index(['right', 'B'])
        else:
            mario_action = COMPLEX_MOVEMENT.index(['right'])
    elif move_left:
        if running:
            mario_action = COMPLEX_MOVEMENT.index(['left', 'B'])
        else:
            mario_action = COMPLEX_MOVEMENT.index(['left'])
    else:
        mario_action = COMPLEX_MOVEMENT.index(['NOOP'])

    # jump height adjustment
    if jumping and (time.time() - jump_start) < 0.8:
        if mario_action == COMPLEX_MOVEMENT.index(['NOOP']):
            mario_action = COMPLEX_MOVEMENT.index(['A'])
        else:
            try:
                base = ['right'] if move_right else ['left']
                if running:
                    base.append('B')
                base.append('A')
                mario_action = COMPLEX_MOVEMENT.index(base)
            except ValueError:
                mario_action = COMPLEX_MOVEMENT.index(['A'])

    movement = COMPLEX_MOVEMENT[mario_action]

    obs, reward, terminated, truncated, info = env.step(mario_action)
    done = terminated or truncated

    cv2.imshow(window_name, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # resets the game when you die a certain amount of times
    if done:
        obs, info = env.reset()
        move_left = False
        move_right = False
        running = False
        jumping = False

env.close()
cv2.destroyAllWindows()
