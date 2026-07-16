from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2  
from pynput import keyboard
import time

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
        enemy_active = ram[0x000F + i] # + i allows active enemies to be checked for in all 5 possible enemy ram slots

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
        distance = (horizontal_distance**2 + vertical_distance**2) ** 0.5
        
        raw_speed_byte = int(ram[0x0058 + enemy["slot"]]) # cast to a standard python int right away
        
        # determining actual direction of enemy by checking if the memory byte is signed as negative (going left) or as positive (going right)

        # if byte raw value is above 128/represents negative values/moving left in nes ...
        if raw_speed_byte > 128:
            enemy_direction = "left"
            enemy_speed = abs(256 - raw_speed_byte) # representing negative values as difference between actual negative value and 256 (unable to represent negative values without taking up important ram space)
        
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
    except:
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
    except:
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
    
    # fetching core absolute coordinates for player
    positions = get_game_positions(env)
    
    # calculating distance and time until collision for enemies
    enemy_profiles = get_distances_to_enemies(env, positions)
    
    # outputing results if threats are detected
    if enemy_profiles:
        print(f"\nMario's current absolute position -- x: {positions['mario']['x']}, y: {positions['mario']['y']}")

        # for each threat in the enemy profiles returned/found ...
        for threat in enemy_profiles:
            coming_status = "YES" if threat['is_coming_towards'] else "NO"

            print(f"Slot {threat['enemy_slot_number']}, Distance: {threat['distance']} pixels | Closing in?: {coming_status} | Time to collision: {threat['time_to_collision_frames']} frames")
            
    cv2.imshow(window_name, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    if done:
        obs, info = env.reset()
        move_left = False
        move_right = False
        move_down = False
        running = False
        jumping = False

env.close()
cv2.destroyAllWindows()