from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2  # for the manual rendering of the game environment to see it
from pynput import keyboard  # For better keyboard controls
import time
import json

from passing_actions.macroroni.library.AutoEnemyPowerupTracking import (
    AutoEnemyPowerupTracking,
)
from MarioVoiceController import MarioVoiceController

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

with open(
    "/Users/nihaalgarud/UTD_nes_voice/passing_actions/macroroni/library/json_file.json",
    "r",
) as f:
    json_data = json.load(f)

# # action : list of buttons to be pressed for action to occur
# ACTION_BUTTON_MAP = {}
# for action_name, action_info in json_data["actions"].items():
#     button = action_info.get("button", "").lower()
#     ACTION_BUTTON_MAP[action_name] = button

# action : list of buttons to be pressed for action to occur with modifier
ACTION_INFO_MAP = {}
for action_name, action_info in json_data["actions"].items():
    ACTION_INFO_MAP[action_name] = {
        "button": action_info.get("button", "").lower(),
        "modifier": action_info.get("modifier", "").lower(),
        "mode": action_info.get("mode", "once"),  # default to one execution
    }

# initializing auto tracking controller for goombas
auto_enemy_tracking = AutoEnemyPowerupTracking(
    json_path="/Users/nihaalgarud/UTD_nes_voice/passing_actions/macroroni/library/json_file.json"
)
voice_controller = MarioVoiceController(
    mapping_json_path="/Users/nihaalgarud/UTD_nes_voice/passing_actions/macroroni/library/json_file.json",
    auto_tracking_class=auto_enemy_tracking,
    device_backend="mps",
    initial_prompt="Look out for words such as goomba, koopa, mash, spam and 'run right'",
)


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


def on_press(key):
    global move_left, move_right, move_down, running, jumping, jump_start

    try:
        if key.char.lower() == "a":
            move_left = True
        elif key.char.lower() == "d":
            move_right = True
        elif key.char.lower() == "f":
            running = True
        elif key.char.lower() == "s":
            move_down = True
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


voice_controller.start_listening()

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

while True:

    key_cv2 = cv2.waitKey(1) & 0xFF
    if key_cv2 == ord("q"):
        break

    if auto_enemy_tracking.passing_action and auto_enemy_tracking.action_type_name:
        action_name = auto_enemy_tracking.action_type_name
        # button = ACTION_INFO_MAP.get(action_name)

        # action_info = json_data.get("actions", {}).get(action_name, {})
        # modifier = action_info.get(
        #     "modifier", ""
        # )  # getting modifier if there, for example B key when run direction commands

        action_info = ACTION_INFO_MAP.get(action_name, {})

        if action_info:
            button = action_info.get("button", "")
            modifier = action_info.get("modifier", "")
        else:
            button = ""
            modifier = ""

        if button:
            try:
                # Handle both single button and multi-button combos
                if button == "left":
                    if modifier == "b":  # if run modifier,
                        mario_action = COMPLEX_MOVEMENT.index(["left", "B"])
                    else:
                        mario_action = COMPLEX_MOVEMENT.index(["left"])
                elif button == "right":
                    if modifier == "b":
                        mario_action = COMPLEX_MOVEMENT.index(["right", "B"])
                    else:
                        mario_action = COMPLEX_MOVEMENT.index(["right"])
                elif button == "down":
                    mario_action = COMPLEX_MOVEMENT.index(["down"])
                elif button == "a":
                    mario_action = COMPLEX_MOVEMENT.index(["A"])
                elif button == "b":
                    mario_action = COMPLEX_MOVEMENT.index(["B"])
                else:
                    mario_action = COMPLEX_MOVEMENT.index(["NOOP"])
            except ValueError:
                mario_action = COMPLEX_MOVEMENT.index(["NOOP"])
        else:
            mario_action = COMPLEX_MOVEMENT.index(["NOOP"])

        # executing logic baesd on mode
        mode = auto_enemy_tracking.action_mode

        # holding mode shouldn't get encapsulated into else statement, so separated
        if mode == "hold":
            pass

        # mashing in 4-frame cycle - 2 frames pressing button, 2 frames releasing button
        elif mode == "mash":
            auto_enemy_tracking.mash_frame_counter += 1

            # if in frames 3-4, release button to mash
            if auto_enemy_tracking.mash_frame_counter % 4 > 2:
                mario_action = COMPLEX_MOVEMENT.index(["NOOP"])

        else:  # mode = once
            if auto_enemy_tracking.action_hold_frames <= 0:
                auto_enemy_tracking.passing_action = False
                auto_enemy_tracking.action_type_name = None
                auto_enemy_tracking.action_mode = "once"
            else:
                auto_enemy_tracking.action_hold_frames = (
                    auto_enemy_tracking.action_hold_frames - 1
                )

    # elif auto_enemy_tracking.active:
    #     positions = auto_enemy_tracking.get_game_positions(env)  # get positions of enemies
    #     enemy_profiles = auto_enemy_tracking.get_distances_to_enemies(env, positions)  # get metrics of enemies
    #     mario_action = auto_enemy_tracking.get_action(enemy_profiles, info.get("score", 0))  # and determine which action mario should go for
    elif auto_enemy_tracking.active:
        positions = auto_enemy_tracking.get_game_positions(
            env
        )  # get positions of objects in game

        if auto_enemy_tracking.powerup_active:
            powerup_positions = auto_enemy_tracking.get_powerup_positions(env)
            powerup_profiles = auto_enemy_tracking.get_distances_to_powerups(
                positions, powerup_positions
            )
            mario_action = auto_enemy_tracking.get_powerup_action(powerup_profiles)
        else:
            enemy_profiles = auto_enemy_tracking.get_distances_to_enemies(
                env, positions
            )
            mario_action = auto_enemy_tracking.get_action(
                enemy_profiles, info.get("score", 0)
            )
    # manual mario control for now
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
