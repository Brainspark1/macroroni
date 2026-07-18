import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2   # for the manual rendering of the game environment to see it
from pynput import keyboard    # For better keyboard controls
import time

class AutoEnemyTracking:
    # defining constants
    MAX_X_DISTANCE = 15  # how far mario should be at most from a goomba before trying to jump on it
    MAX_KILL_X = 22 # the maximum distance between mario and goomba that initiates mario's actions to jump on the goomba
    FRAMES_DECIDE_RUN = 20 # no need for mario to run up to goomba if frames/time between goomba and mario less than this value
    MARIO_APPROACH_SPEED_FALLBACK = 2.0 # in pixels per frame, used if we can't read mario's real speed yet
    MIN_HOLD_FRAMES = 3 # clamping minimum hold jump frames to prevent mario from doing too short hop
    MAX_HOLD_FRAMES = 14 # clamping maximum hold jump frames to prevent mario from doing too long leap

    def __init__(self):
        self.active = False
        self.target_slot = None
        self.jump_direction = None
        self.recovery_frames_left = 0    
        self.total_recovery_frames = 0
        self.jump_hold_frames = 0 
        self.starting_score = None
        self.awaiting_kill_confirmation = False

    # method to activate auto tracking
    def activate(self):
        self.active = True
        self.target_slot = None
        self.jump_direction = None
        self.recovery_frames_left = 0
        self.total_recovery_frames = 0
        self.jump_hold_frames = 0
        self.starting_score = None
        self.awaiting_kill_confirmation = False

    # method to stop auto tracking/set everything to default value
    def deactivate(self):
        self.active = False
        self.target_slot = None
        self.jump_direction = None
        self.recovery_frames_left = 0
        self.total_recovery_frames = 0
        self.jump_hold_frames = 0
        self.starting_score = None
        self.awaiting_kill_confirmation = False

    # method to find closest enemy target to get
    def pick_target(self, enemy_profiles):
        # starting point by assuming first enemy in enemy list is the closest to mario
        closest_enemy = enemy_profiles[0]
        # getting amount of time until collision
        closest_time = closest_enemy.get("time_to_collision_frames", float('inf'))

        # checking every other enemy from index 1 to the end of the list
        for enemy in enemy_profiles[1:]:
            # getting corresponding time to collision
            enemy_time = enemy.get("time_to_collision_frames", float('inf'))

            # if new enemy time is less than closest found time yet, set enemy and time values accordingly
            if enemy_time < closest_time:
                closest_enemy = enemy
                closest_time = enemy_time

        # return which enemy is the closest one
        return closest_enemy

    # method to solve quadratic jump arc model for the current jump to return how many frames the jump button should be held
    def calculate_jump_arc(self, horizontal_distance, enemy):
        approach_speed = self.MARIO_APPROACH_SPEED_FALLBACK

        # if enemy is coming towards mario, add the speed of the enemy to the approach speed
        if enemy.get("is_coming_towards"):
            approach_speed = approach_speed + enemy.get("speed_per_frame", 0)

        # preventing speed from being negative or 0 as fallback in case this happens
        if approach_speed <= 0:
            approach_speed = self.MARIO_APPROACH_SPEED_FALLBACK

        # holds how many frames until mario and goomba are at same x position (time = distance between them / speed goomba is coming at)
        time_to_collision = abs(horizontal_distance) / approach_speed

        # hold jump button for around half the total time to get to vertex of arc
        hold_jump_frames = round(time_to_collision / 2)

        # produces hold frame value in middle to prevent overshooting or undershooting
        hold_jump_frames = max(self.MIN_HOLD_FRAMES, min(self.MAX_HOLD_FRAMES, hold_jump_frames))

        # total frames is just double number of frames to hold jump, since the jump button needs to be held for half of the arc to get to vertex
        total_frames = hold_jump_frames * 2

        return hold_jump_frames, total_frames

    # method for mario to get the action/combination of buttons he needs to get the closest enemy/goomba
    def get_action(self, enemy_profiles, current_score):
        # commiting to jump mario has started mid-jump if still have frames to jump and set direction
        if self.recovery_frames_left > 0 and self.jump_direction:
            # decreasing number of frames left for mario to continue jumping action
            self.recovery_frames_left -= 1

            # calculating number of frames passed by to determine for how much longer the jump button needs to be held
            frames_elapsed = self.total_recovery_frames - self.recovery_frames_left

            # if number of frames elapsed since start haven't reached this jump's apex time,
            if frames_elapsed <= self.jump_hold_frames:
                # keep holding jump button (still on the way up the parabola)
                return COMPLEX_MOVEMENT.index([self.jump_direction, 'A'])
            else:
                if self.recovery_frames_left == 0:
                    self.awaiting_kill_confirmation = True  # if has landed, check if goomba has been killed
                # release jump button and let gravity take over for the descending half
                return COMPLEX_MOVEMENT.index([self.jump_direction])

        # if no enemies in sight, deactivate and make mario not do anything
        if not enemy_profiles:
            self.deactivate()
            return COMPLEX_MOVEMENT.index(['NOOP'])

        # PIPELINE - picking a target and deciding to kill it or get closer
        target = self.pick_target(enemy_profiles)
        self.target_slot = target["enemy_slot_number"]
        horizontal_distance = target["horizontal_distance"]
        time_to_collision = target["time_to_collision_frames"]

        # if move right (positive horizontal distance), press right button and run, otherwise left button and run
        if horizontal_distance >= 0:
            desired_run_action = ['right', 'B']
        else:
            desired_run_action = ['left', 'B']

        # determine to kill goomba if ...
        kill_attempt = (
            # x distance between goomba and mario is in range of set kill distance
            abs(horizontal_distance) <= self.MAX_KILL_X
        ) or (
            # OR x distance is around two times kill distance (approximation taking into account goomba's velocity) and time to collision is within number of frames to count enemy as target
            abs(horizontal_distance) <= self.MAX_KILL_X * 2
            and time_to_collision <= self.FRAMES_DECIDE_RUN
        )

        # if mario should try to kill the goomba ...
        if kill_attempt:
            # locking jump direction to prevent wobbling mid-air in case
            if horizontal_distance >= 0:
                self.jump_direction = 'right'
            else:
                self.jump_direction = 'left'

            self.jump_hold_frames, self.total_recovery_frames = self.calculate_jump_arc(horizontal_distance, target)
            self.recovery_frames_left = self.total_recovery_frames
            self.starting_score = current_score

            # walking jump for more consistent execution (no run button/B pressed)
            return COMPLEX_MOVEMENT.index([self.jump_direction, 'A'])

        # mario approaching the goomba if still greater distance than max distance between mario and goomba to execute killing action
        if abs(horizontal_distance) > self.MAX_X_DISTANCE:
            return COMPLEX_MOVEMENT.index(desired_run_action)  # executing sequence of run actions defined above to get to goomba position

        # keep walking until in killing range if close-by/needs little more time
        if horizontal_distance >= 0:
            return COMPLEX_MOVEMENT.index(['right'])
        else:
            return COMPLEX_MOVEMENT.index(['left'])

    # method to confirm whether or not to kill the goomba
    def confirming_stopping_kill(self, current_score):
        if not self.awaiting_kill_confirmation:
            return False

        # setting kill confirmation variable to false as can only get checked once per landing
        self.awaiting_kill_confirmation = False

        # if current score is greater than the score at jump-start (points went up during auto tracking)
        if self.starting_score is not None and current_score > self.starting_score:
            self.deactivate()
            return True

        # return false by default as awaiting kill confirmation set to false
        return False

# method to get positions of mario and enemy from memory
def get_game_positions(env):
    # accessing raw ram bytes for environment (2048 bytes array for nes games)
    ram = env.unwrapped.ram

    # page on which mario currently is at (width of 256 pixels)
    mario_x_page = int(ram[0x006D])

    # horizontal position of mario in current page
    mario_x_screen = int(ram[0x0086])

    # gets position of mario from very start of level, since each page = 256 pixels long,
    # so multiplying page number by page width then adding mario x position on current page produces distance from start of level
    mario_x_pos = (mario_x_page * 256) + mario_x_screen

    # mario y position
    mario_y_pos = int(ram[0x00CE])

    # initializing empty list to hold data for any enemies currently on screen
    enemy_slots = []

    # looping through the 5 available hardware slots that the nes uses to track active enemies (from data crystal)
    for i in range(5):

        # checking to see if current enemy slot contains a live enemy (note that 00 means empty/dead)
        enemy_active = ram[0x000F + i]  # + i allows active enemies to be checked for in all 5 possible enemy ram slots

        # if there is an active enemy
        if enemy_active:

            # tracking absolute positions of enemies in each slot
            enemy_x_page = int(ram[0x006E + i])
            enemy_x_screen = int(ram[0x0087 + i])

            # getting absolute position of enemy from start of level to properly match mario's coordinate scale
            enemy_x_pos = (enemy_x_page * 256) + enemy_x_screen

            # enemy y position
            enemy_y_pos = int(ram[0x00CF + i])

            # adding to currently looped over slot the x and y positions of an enemy if one is there
            enemy_slots.append({
                "slot": i,
                "x": enemy_x_pos,
                "y": enemy_y_pos
            })

    # returning dictionary containing mario's x and y positions, and the positions of enemies in each slot
    return {
        "mario": {"x": mario_x_pos, "y": mario_y_pos},
        "enemies": enemy_slots
    }

# method to get the horizontal and vertical distances between mario and enemies
def get_distances_to_enemies(env, positions):
    ram = env.unwrapped.ram

    # dictionary to hold distance metrics for every active enemy found on screen
    enemy_metrics = []

    # extracting mario coordinates
    mario_x = positions["mario"]["x"]
    mario_y = positions["mario"]["y"]

    # looping through each active enemy dictionary stored in the provided list
    for enemy in positions["enemies"]:
        # calculating horizontal distance (note that positive means enemy is right, negative means enemy is left)
        horizontal_distance = enemy["x"] - mario_x

        # calculating vertical distance (positive means enemy is below mario, negative means above mario)
        vertical_distance = enemy["y"] - mario_y

        # finding distance using pythagorean theorem
        distance = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5

        raw_speed_byte = int(ram[0x0058 + enemy["slot"]])  # cast to a standard python int right away

        # determining actual direction of enemy by checking if the memory byte is signed as negative (going left) or as positive (going right)
        # if byte raw value is above 128/represents negative values/moving left in nes ...
        if raw_speed_byte > 128:
            enemy_direction = "left"
            enemy_speed = abs(256 - raw_speed_byte)  # representing negative values as difference between actual negative value and 256 (unable to represent negative values without taking up important ram space)
        # if byte raw value is below 128/represents positive values/moving right in nes ...
        else:
            enemy_direction = "right"
            enemy_speed = raw_speed_byte

        # fallback safety check to avoid any zero division errors if an enemy is momentarily stationary
        if enemy_speed == 0:
            enemy_speed = 1

        # block determining if velocity direction means the gap is actively closing between enemy and mario
        is_moving_towards_mario = False

        # if enemy is to the right of mario (positive distance) and its physics vector is going left
        if horizontal_distance > 0 and enemy_direction == "left":
            is_moving_towards_mario = True

        # if enemy is to the left of mario (negative distance) and its physics vector is going right
        elif horizontal_distance < 0 and enemy_direction == "right":
            is_moving_towards_mario = True

        # calculating time to collision with time = distance / speed - note that if enemy moving away from mario, set time to infinity to avoid any errors
        if is_moving_towards_mario:
            time_to_collision_frames = abs(horizontal_distance) / enemy_speed
        else:
            time_to_collision_frames = float('inf')

        # appending spatial calculations, distance, and real time collision predictions to output dictionary
        enemy_metrics.append({
            "enemy_slot_number": enemy["slot"],
            "horizontal_distance": horizontal_distance,
            "vertical_distance": vertical_distance,
            "distance": round(distance, 2),
            "speed_per_frame": enemy_speed,
            "is_coming_towards": is_moving_towards_mario,
            "time_to_collision_frames": time_to_collision_frames
        })

    # returning calculated metrics
    return enemy_metrics

# most code taken from Joshua's existing file to run the environment/emulator
env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='rgb_array')  # For the Emulator Environment
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs, info = env.reset()

window_name = 'Super Mario Bros'  # For Display Size
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1920, 1080)

# control additions to fix jumping and running
mario_action = 0
move_left = False
move_right = False
move_down = False
running = False
jumping = False        # better adjustment for jumping height
jump_start = 0

# initializing auto tracking controller for goombas
auto_enemy_tracking = AutoEnemyTracking()

def on_press(key):
    global move_left, move_right, move_down, running, jumping, jump_start

    try:
        if key.char.lower() == 'a':
            move_left = True
        elif key.char.lower() == 'd':
            move_right = True
        elif key.char.lower() == 'z':
            running = True
        elif key.char.lower() == 's':
            move_down = True
        elif key.char.lower() == 'e':
            # activate auto tracking if e key is pressed
            auto_enemy_tracking.activate()
    except AttributeError:
        pass

    if key == keyboard.Key.space and not jumping:
        jumping = True
        jump_start = time.time()


def on_release(key):
    global move_left, move_right, move_down, running, jumping
    try:
        if key.char.lower() == 'a':
            move_left = False
        elif key.char.lower() == 'd':
            move_right = False
        elif key.char.lower() == 'z':
            running = False
        elif key.char.lower() == 's':
            move_down = False
    except AttributeError:
        pass
    if key == keyboard.Key.space:
        jumping = False


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

while True:
    start_time = time.time()

    key_cv2 = cv2.waitKey(1) & 0xFF
    if key_cv2 == ord('q'):
        break

    # if auto tracking is active,
    if auto_enemy_tracking.active:
        positions = get_game_positions(env)  # get positions of enemies
        enemy_profiles = get_distances_to_enemies(env, positions)  # get metrics of enemies
        mario_action = auto_enemy_tracking.get_action(enemy_profiles, info.get('score', 0))  # and determine which action mario should go for
    else:
        # helps mario's movement while jumping
        if move_down:
            mario_action = COMPLEX_MOVEMENT.index(['down'])
        elif move_right and move_left:
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
            elif mario_action == COMPLEX_MOVEMENT.index(['down']):
                pass
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

    # stepping the emulator forward with chosen action
    obs, reward, terminated, truncated, info = env.step(mario_action)
    done = terminated or truncated

    if auto_enemy_tracking.active:
        if auto_enemy_tracking.confirming_stopping_kill(info.get('score', 0)):
            print("Stopping auto tracking.")

    # fetching core absolute coordinates for player
    positions = get_game_positions(env)

    # calculating distance and time until collision for enemies
    enemy_profiles = get_distances_to_enemies(env, positions)

    # commented out as lagging too much
    # # outputing results if threats are detected
    # if enemy_profiles:
    #     print(f"\nMario's current absolute position -- x: {positions['mario']['x']}, y: {positions['mario']['y']}")

    #     # for each threat in the enemy profiles returned/found ...
    #     for threat in enemy_profiles:
    #         coming_status = "YES" if threat['is_coming_towards'] else "NO"

    #         print(f"Slot {threat['enemy_slot_number']}, Distance: {threat['distance']} pixels | Closing in?: {coming_status} | Time to collision: {threat['time_to_collision_frames']} frames")

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
    env.unwrapped.ram[0x07F8] = 0x09  # hundreds
    env.unwrapped.ram[0x07F9] = 0x09  # tens
    env.unwrapped.ram[0x07FA] = 0x09  # ones

env.close()
cv2.destroyAllWindows()