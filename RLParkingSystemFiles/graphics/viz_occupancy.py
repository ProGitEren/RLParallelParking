import numpy as np
import matplotlib.pyplot as plt


def show_occupancy_grid(ogm_1d, grid_size=(20, 20), window_name="Occupancy Grid"):
    if ogm_1d is None or len(ogm_1d) == 0:
        print("No OGM data.")
        return

    ogm_2d = np.reshape(ogm_1d, grid_size)

    plt.clf()  # Clear previous figure
    plt.imshow(ogm_2d, origin='lower', cmap='gray')
    plt.title(window_name)
    plt.colorbar(label="Occupied=1 / Free=0")
    plt.pause(0.001)  # Small pause to update plot
