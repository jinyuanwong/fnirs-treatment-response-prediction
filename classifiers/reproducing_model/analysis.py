import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def imshow_blue_red(data, title, xlabel, ylabel):
    max_val = np.max(np.abs(data))

    # Define the colors for the custom colormap (blue, white, red)
    cmap_colors = [(0, "blue"), (0.5, "white"), (1, "red")]

    # Create the colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)

    # Apply the colormap with normalization
    plt.imshow(data, cmap=custom_cmap, vmin=-max_val, vmax=max_val)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
