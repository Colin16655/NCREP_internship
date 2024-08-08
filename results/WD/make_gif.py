import imageio
import os
from natsort import natsorted

# Directory containing PNG files
png_dir = 'results/WD'

def create(directory, name):
    # List of PNG files in the directory
    images = []
    for file_name in natsorted(os.listdir(png_dir)):
        if file_name.startswith(name):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))

    # Output GIF file path
    gif_path =  directory + name + '.gif'

    # Save as GIF
    imageio.mimsave(gif_path, images, duration=1)


# Create the GIF
create(png_dir, 'MAC_Matrix')
create(png_dir, 'PSD')