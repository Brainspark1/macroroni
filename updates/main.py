import sys
import stable_retro as retro
import pygame

from ManualActionHandler import ManualActionHandler
from TrackingActionHandler import TrackingActionHandler
from NESVoiceController import NESVoiceController

# Initialize Pygame to handle window rendering and keyboard input
pygame.init()

# 2. Setup the Stable Retro environment
ENV_NAME = "SuperMarioBros-Nes-v0"
try:
    env = retro.make(game=ENV_NAME, state=retro.State.DEFAULT)
except Exception as e:
    print(f"Error loading environment: {e}")
    print("Ensure your ROM is imported using: python3 -m retro.import /path/to/roms")
    sys.exit(1)

manual_action_handler = ManualActionHandler()
tracking_action_handler = TrackingActionHandler()
nes_voice_controller = NESVoiceController()

obs, info = env.reset()

SCREEN_SCALE = 3
screen_width = obs.shape[1] * SCREEN_SCALE
screen_height = obs.shape[0] * SCREEN_SCALE
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Kirby's Adventure")

clock = pygame.time.Clock()
running = True

while running:
    # Handles window close or escape key
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # blank NES controller array: [B, A, MODE, START, UP, DOWN, LEFT, RIGHT]
    # All buttons default to False (0)
    # action = [0, 0, 0, 0, 0, 0, 0, 0]

    action = manual_action_handler.get_action_array()
    duration = manual_action_handler.get_duration_array()

    # keyboard states
    keys = pygame.key.get_pressed()

    # WASD to D-pad 
    if keys[pygame.K_w]: action[4] = 1  # UP
    if keys[pygame.K_s]: action[5] = 1  # DOWN
    if keys[pygame.K_a]: action[6] = 1  # LEFT
    if keys[pygame.K_d]: action[7] = 1  # RIGHT

    # E to Attack 
    if keys[pygame.K_e]: action[0] = 1  # B

    # Spacebar to Jump (Button A)
    if keys[pygame.K_Space]: action[1] = 1  # A

    # Step the environment forward with custom actions
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    for d in duration:
        if d == 0:
            d = 0
        else:
            d -= 1

    # Converts the environment's RGB frame array to a Pygame surface and display it
    # Transpose frame array from (Height, Width, Channel) to Pygame's (Width, Height, Channel)
    frame = obs.transpose(1, 0, 2)
    surf = pygame.surfarray.make_surface(frame)
    
    # Scale and draw the game frame
    scaled_surf = pygame.transform.scale(surf, (screen_width, screen_height))
    screen.blit(scaled_surf, (0, 0))
    pygame.display.flip()  

    # limits fps to 60
    clock.tick(60)

env.close()
pygame.quit()
sys.exit()
