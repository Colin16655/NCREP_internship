import os
from PIL import Image
import re
import numpy as np

def create_gif_from_folder(image_folder, gif_filename='fft_animation.gif', fps=2):
    """
    Reads saved images from a folder, sorts them by filenames (like fft_<number>.png), and creates an animated GIF.
    
    Parameters:
    image_folder (str): Folder containing the images.
    gif_filename (str): Filename for the output GIF.
    fps (int): Frames per second for the GIF.
    """
    # Gather all image files (assuming PNG format) from the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # Function to extract the number from the filename (e.g., fft_<number>.png)
    def extract_number(file):
        match = re.search(r'fft_(\d+).png', file)
        return int(match.group(1)) if match else -1
    
    # Sort files by the number extracted from their filenames
    image_files_sorted = sorted(image_files, key=extract_number)

    if not image_files_sorted:
        print("No image files found in the specified folder.")
        return

    # Load images into a list
    images = [Image.open(os.path.join(image_folder, file)) for file in image_files_sorted]

    # Save as GIF
    images[0].save(
        gif_filename, save_all=True, append_images=images[1:], 
        duration=1000 // fps, loop=0
    )

create_gif_from_folder(f'PART_I/results/exp3_fft_plots', gif_filename='fft_linear_8-24_10min.gif', fps=4)
