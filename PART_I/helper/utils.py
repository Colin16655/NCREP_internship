import os
import matplotlib.pyplot as plt

def create_output_dir(folder_name, output_dir):
    """
    Creates a directory for saving figures based on the provided folder name.

    Parameters:
    folder_name (str): Name of the folder to create.
    output_dir (str): Name of the output directory containing folder folder_name (to create if not existing).
    """
    path = os.path.join(output_dir, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_figure(fig, file_name, folder_name, output_dir, format='pdf'):
    """
    Saves the figure in PDF or PNG formats.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        file_name (str): The base name for the saved files.
        folder_name (str): The folder where the files will be saved.
    """
    output_path = create_output_dir(folder_name, output_dir)
    file_path = os.path.join(output_path, f"{file_name}.{format}")
    
    fig.savefig(file_path, format=format, bbox_inches='tight')
    plt.close(fig)