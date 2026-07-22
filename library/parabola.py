from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2  # for the manual rendering of the game environment to see it
from pynput import keyboard  # For better keyboard controls
import time
import threading

from AutoEnemyTracking import AutoEnemyTracking

# most code taken from Joshua's existing file to run the environment/emulator
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", render_mode="rgb_array"
)  # For the Emulator Environment
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs, info = env.reset()

window_name = "Super Mario Bros"  # For Display Size
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1920, 1080)

# control additions to fix jumping and running
mario_action = 0
move_left = False
move_right = False
move_down = False
running = False
jumping = False  # better adjustment for jumping height
jump_start = 0
awaiting_target_input = False

# initializing auto tracking controller for goombas
auto_enemy_tracking = AutoEnemyTracking()


# method to handle incoming user transcripts into emulator, connect to whisper
def handle_transcript(sentence):
    sentence = sentence.strip().lower()  # cleaning up sentence

    # if no transcript, say so
    if not sentence:
        print("No transcript, not starting auto tracking.")
        return

    if sentence == "any":
        auto_enemy_tracking.activate()

        auto_enemy_tracking.target_type_address = None
        auto_enemy_tracking.target_type_name = None
        print("Auto tracker enabled, set to any enemy")
        return

    name, _ = auto_enemy_tracking.activate_set_target(sentence)

    if name:
        print(f"Auto tracking started on {name}")
    else:
        print("No recognized target to track.")
        auto_enemy_tracking.deactivate()


def input_listener():
    global awaiting_target_input

    print("Input listener is active.")

    while True:
        if awaiting_target_input:
            target_input = input("Enter target: ")
            awaiting_target_input = False

            handle_transcript(target_input)
        else:
            time.sleep(0.3)


def on_press(key):
    global move_left, move_right, move_down, running, jumping, jump_start, awaiting_target_input

    try:
        if key.char.lower() == "a":
            move_left = True
        elif key.char.lower() == "d":
            move_right = True
        elif key.char.lower() == "f":
            running = True
        elif key.char.lower() == "s":
            move_down = True
        elif key.char.lower() == "e":
            awaiting_target_input = True
    except AttributeError:
        pass

    if key == keyboard.Key.space and not jumping:
        jumping = True
        jump_start = time.time()


def on_release(key):
    global move_left, move_right, move_down, running, jumping
    try:
        if key.char.lower() == "a":
            move_left = False
        elif key.char.lower() == "d":
            move_right = False
        elif key.char.lower() == "f":
            running = False
        elif key.char.lower() == "s":
            move_down = False
    except AttributeError:
        pass
    if key == keyboard.Key.space:
        jumping = False


transcript_thread = threading.Thread(target=input_listener)
transcript_thread.daemon = True
transcript_thread.start()

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

while True:
    start_time = time.time()

    key_cv2 = cv2.waitKey(1) & 0xFF
    if key_cv2 == ord("q"):
        break

    # if auto tracking is active,
    if auto_enemy_tracking.active:
        positions = auto_enemy_tracking.get_game_positions(
            env
        )  # get positions of enemies
        enemy_profiles = auto_enemy_tracking.get_distances_to_enemies(
            env, positions
        )  # get metrics of enemies
        mario_action = auto_enemy_tracking.get_action(
            enemy_profiles, info.get("score", 0)
        )  # and determine which action mario should go for
    else:
        # helps mario's movement while jumping
        if move_down:
            mario_action = COMPLEX_MOVEMENT.index(["down"])
        elif move_right and move_left:
            mario_action = COMPLEX_MOVEMENT.index(["NOOP"])
        elif move_right:
            if running:
                mario_action = COMPLEX_MOVEMENT.index(["right", "B"])
            else:
                mario_action = COMPLEX_MOVEMENT.index(["right"])
        elif move_left:
            if running:
                mario_action = COMPLEX_MOVEMENT.index(["left", "B"])
            else:
                mario_action = COMPLEX_MOVEMENT.index(["left"])
        else:
            mario_action = COMPLEX_MOVEMENT.index(["NOOP"])

        # jump height adjustment
        if jumping and (time.time() - jump_start) < 0.8:
            if mario_action == COMPLEX_MOVEMENT.index(["NOOP"]):
                mario_action = COMPLEX_MOVEMENT.index(["A"])
            elif mario_action == COMPLEX_MOVEMENT.index(["down"]):
                pass
            else:
                try:
                    base = ["right"] if move_right else ["left"]
                    if running:
                        base.append("B")
                    base.append("A")
                    mario_action = COMPLEX_MOVEMENT.index(base)
                except ValueError:
                    mario_action = COMPLEX_MOVEMENT.index(["A"])

    movement = COMPLEX_MOVEMENT[mario_action]

    # stepping the emulator forward with chosen action
    obs, reward, terminated, truncated, info = env.step(mario_action)
    done = terminated or truncated

    if auto_enemy_tracking.active:
        if auto_enemy_tracking.confirming_stopping_kill(info.get("score", 0)):
            print("Stopping auto tracking.")

    # fetching core absolute coordinates for player
    positions = auto_enemy_tracking.get_game_positions(env)

    # calculating distance and time until collision for enemies
    enemy_profiles = auto_enemy_tracking.get_distances_to_enemies(env, positions)

    cv2.imshow(window_name, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    if done:
        obs, info = env.reset()
        move_left = False
        move_right = False
        move_down = False
        running = False
        jumping = False
        auto_enemy_tracking.deactivate()

    # increase max time per level to 999
    env.unwrapped.ram[auto_enemy_tracking.env_time_hundred] = 0x09  # hundreds
    env.unwrapped.ram[auto_enemy_tracking.env_time_ten] = 0x09  # tens
    env.unwrapped.ram[auto_enemy_tracking.env_time_one] = 0x09  # ones

env.close()
cv2.destroyAllWindows()
