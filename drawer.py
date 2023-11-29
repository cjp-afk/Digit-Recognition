# Digit Recogniser by Cameron Paton
# Date: 2021-08-09


import pygame
from model_cnn import predict
from image_processing import process_image
import os


# Initialize pygame
def run_app():
    pygame.init()

    # Set the dimensions of the window
    window_width = 280
    window_height = 280

    # Create the window
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Drawing Window")

    # Set the initial background color to black
    background_color = (0, 0, 0)
    window.fill(background_color)

    # Set the initial drawing color to white
    drawing_color = (255, 255, 255)

    # Variable to keep track of whether the mouse button is pressed
    is_drawing = False

    # Variable to keep track of the image variant number

    # Function to save the image
    def save_image():
        global variant_number
        image_name = "digit.png"
        pygame.image.save(window, image_name)
        window.fill(background_color)

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                is_drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                is_drawing = False
            elif event.type == pygame.MOUSEMOTION and is_drawing:
                # Get the position of the mouse
                mouse_pos = pygame.mouse.get_pos()

                # Draw a white pixel at the mouse position
                pygame.draw.circle(window, drawing_color, mouse_pos, 8)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                save_image()
                image = process_image("digit.png")
                predict(image)
                os.remove("digit.png")

        # Update the display
        pygame.display.flip()

    # Quit pygame
    pygame.quit()
