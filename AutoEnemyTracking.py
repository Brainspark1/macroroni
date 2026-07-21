from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import json

from SemanticMapper import SemanticMapper

class AutoEnemyTracking:
    # defining constants
    # MAX_X_DISTANCE = 15  # how far mario should be at most from a goomba before trying to jump on it
    # MAX_KILL_X = 22 # the maximum distance between mario and goomba that initiates mario's actions to jump on the goomba
    # FRAMES_DECIDE_RUN = 20 # no need for mario to run up to goomba if framesframes/time between goomba and mario less than this value
    # MARIO_APPROACH_SPEED_FALLBACK = 2.0 # in pixels per frame, used if we can't read mario's real speed yet
    # MIN_HOLD_FRAMES = 3 # clamping minimum hold jump frames to prevent mario from doing too short hop
    # MAX_HOLD_FRAMES = 14 # clamping maximum hold jump frames to prevent mario from doing too long leap

    # :param json_path - structure varies based on OS (mac, full path; windows, full path with r in front of string)
    def __init__(self, json_path="/Users/nihaalgarud/auto_track/target_tracking/macroroni/json_file.json", max_x_distance=15, max_kill_x=22, frames_decide_run=20, mario_approach_speed_fallback=2.0, min_hold_frames=3, max_hold_frames=14): 
        self.active = False
        self.target_slot = None
        self.jump_direction = None
        self.recovery_frames_left = 0    
        self.total_recovery_frames = 0
        self.jump_hold_frames = 0 
        self.starting_score = None
        self.awaiting_kill_confirmation = False
        
        self.data = self.read_json_file(json_path)
        self.character_data = self.data["characters"]
        self.enemy_data = self.data["targets"]["enemy"]
        self.item_data = self.data["items"]
        self.env_data = self.data["environment"]

        # getting name to address dictionary for all targets
        self.target_type_lookup = {
            name: int(info["address"], 16)
            for name, info in self.data["targets"].items()
            if name != "enemy"
        }

        self.semantic_mapper = SemanticMapper(json_path=json_path)
        self.target_type_address = None
        self.target_type_name = None

        self.max_x_distance = max_x_distance
        self.max_x_kill = max_kill_x
        self.frames_decide_run = frames_decide_run
        self.mario_approach_speed_fallback = mario_approach_speed_fallback
        self.min_hold_frames = min_hold_frames
        self.max_hold_frames = max_hold_frames

        self.character_absolute_page_number, self.character_vertical_screen_position, self.character_horizontal_position = self.initialize_character_variables()
        self.enemy_active_state, self.enemy_type_address, self.enemy_horizontal_velocity, self.enemy_absolute_map_num, self.enemy_horizontal_page_position, self.enemy_vertical_screen_position = self.initialize_enemy_variables()
        self.initialize_item_veriables()
        self.env_time_hundred, self.env_time_ten, self.env_time_one = self.initialize_environment_variables()

    def initialize_character_variables(self):
        character_absolute_page_number = int(self.character_data["absolute_page_number"], 16)
        character_vertical_screen_position = int(self.character_data["vertical_screen_position"], 16)
        character_horizontal_position = int(self.character_data["horizontal_position"], 16)

        return character_absolute_page_number, character_vertical_screen_position, character_horizontal_position

    def initialize_enemy_variables(self):
        enemy_active_state = int(self.enemy_data["active_state"], 16)
        enemy_type_address = int(self.enemy_data["enemy_type"], 16)
        enemy_horizontal_velocity = int(self.enemy_data["horizontal_velocity"], 16)
        enemy_absolute_map_num = int(self.enemy_data["absolute_map_num"], 16)
        enemy_horizontal_page_position = int(self.enemy_data["horizontal_page_position"], 16)
        enemy_vertical_screen_position = int(self.enemy_data["enemy_vertical_screen_position"], 16)

        return enemy_active_state, enemy_type_address, enemy_horizontal_velocity, enemy_absolute_map_num, enemy_horizontal_page_position, enemy_vertical_screen_position

    def initialize_item_veriables(self):
        pass
        
    def initialize_environment_variables(self):
        env_time_hundred = int(self.env_data["time_hundred"], 16)
        env_time_ten = int(self.env_data["time_ten"], 16)
        env_time_one = int(self.env_data["time_one"], 16)

        return env_time_hundred, env_time_ten, env_time_one

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
        self.target_type_address = None
        self.target_type_name = None

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
        self.target_type_address = None
        self.target_type_name = None

# HERE - add method to get target from sentence/return name and confidence score calling find_max_similarity() in sml file

    # needs to return name and confidence score
    def set_target_from_similarity(self, transcript_sentence, min_confidence=0.2):
        name, score = self.semantic_mapper.find_max_similarity(transcript_sentence)

        if score < min_confidence:
            self.target_type_address = None
            self.target_type_name = None
            return None, score
        
        self.target_type_address = self.target_type_lookup.get(name)
        self.target_type_name = name
        return name, score
    
    def activate_set_target(self, transcript_sentence):
        self.activate()
        return self.set_target_from_similarity(transcript_sentence)

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
        approach_speed = self.mario_approach_speed_fallback

        # if enemy is coming towards mario, add the speed of the enemy to the approach speed
        if enemy.get("is_coming_towards"):
            approach_speed = approach_speed + enemy.get("speed_per_frame", 0)

        # preventing speed from being negative or 0 as fallback in case this happens
        if approach_speed <= 0:
            approach_speed = self.mario_approach_speed_fallback

        # holds how many frames until mario and goomba are at same x position (time = distance between them / speed goomba is coming at)
        time_to_collision = abs(horizontal_distance) / approach_speed

        # hold jump button for around half the total time to get to vertex of arc
        hold_jump_frames = round(time_to_collision / 2)

        # produces hold frame value in middle to prevent overshooting or undershooting
        hold_jump_frames = max(self.min_hold_frames, min(self.max_hold_frames, hold_jump_frames))

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
        
        # if the address isnt none get candidate/enemy profile, if no enemies in range do nothing(['NOOP])
        if self.target_type_address is not None:
            candidates = [enemy for enemy in enemy_profiles if enemy.get("enemy_type") == self.target_type_address]
        else:
            candidates = enemy_profiles

        if not candidates:
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
            abs(horizontal_distance) <= self.max_x_kill
        ) or (
            # OR x distance is around two times kill distance (approximation taking into account goomba's velocity) and time to collision is within number of frames to count enemy as target
            abs(horizontal_distance) <= self.max_x_kill * 2
            and time_to_collision <= self.frames_decide_run
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
        if abs(horizontal_distance) > self.max_x_distance:
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
    def get_game_positions(self, env):
        # accessing raw ram bytes for environment (2048 bytes array for nes games)
        ram = env.unwrapped.ram

        # page on which mario currently is at (width of 256 pixels)
        mario_x_page = int(ram[self.character_absolute_page_number])

        # horizontal position of mario in current page
        mario_x_screen = int(ram[self.character_horizontal_position])

        # gets position of mario from very start of level, since each page = 256 pixels long,
        # so multiplying page number by page width then adding mario x position on current page produces distance from start of level
        mario_x_pos = (mario_x_page * 256) + mario_x_screen

        # mario y position
        mario_y_pos = int(ram[self.character_vertical_screen_position])

        # initializing empty list to hold data for any enemies currently on screen
        enemy_slots = []

        # looping through the 5 available hardware slots that the nes uses to track active enemies (from data crystal)
        for i in range(5):

            # checking to see if current enemy slot contains a live enemy (note that 00 means empty/dead)
            enemy_active = ram[self.enemy_active_state + i]  # + i allows active enemies to be checked for in all 5 possible enemy ram slots

            # if there is an active enemy
            if enemy_active:

                # tracking absolute positions of enemies in each slot
                enemy_x_page = int(ram[self.enemy_absolute_map_num + i])
                enemy_x_screen = int(ram[self.enemy_horizontal_page_position + i])

                # getting absolute position of enemy from start of level to properly match mario's coordinate scale
                enemy_x_pos = (enemy_x_page * 256) + enemy_x_screen

                # enemy y position
                enemy_y_pos = int(ram[self.enemy_vertical_screen_position + i])

                # enemy type
                enemy_type = int(ram[self.enemy_type_address + i])

                # adding to currently looped over slot the x and y positions of an enemy if one is there
                enemy_slots.append({
                    "slot": i,
                    "x": enemy_x_pos,
                    "y": enemy_y_pos,
                    "type": enemy_type
                })

        # returning dictionary containing mario's x and y positions, and the positions of enemies in each slot
        return {
            "mario": {"x": mario_x_pos, "y": mario_y_pos},
            "enemies": enemy_slots
        }

    # method to get the horizontal and vertical distances between mario and enemies
    def get_distances_to_enemies(self, env, positions):
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

            raw_speed_byte = int(ram[self.enemy_horizontal_velocity + enemy["slot"]])  # cast to a standard python int right away

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
                "enemy_type": enemy["type"],
                "horizontal_distance": horizontal_distance,
                "vertical_distance": vertical_distance,
                "distance": round(distance, 2),
                "speed_per_frame": enemy_speed,
                "is_coming_towards": is_moving_towards_mario,
                "time_to_collision_frames": time_to_collision_frames
            })

        # returning calculated metrics
        return enemy_metrics
    
    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file) 

        return data