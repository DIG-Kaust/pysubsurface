"""
Custom Colormaps
================

This example shows additional colormaps that are provided by PySubsurface for visualization of various data sources.

Some of such colormaps are inspired by commercial softwares and their name is sometimes an
indication of that. The aim to give experienced users the feeling of being at home even outside of their
favourite interpretation software.
"""
import numpy as np
import matplotlib.pyplot as plt
from pysubsurface.visual.cmap import *

plt.close('all')


###############################################################################
# Let's create a gradient and display it with different colormaps
cmaps_names = [('Additional colormaps', list(cmaps.keys()))]
nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps_names)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.25, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=cmaps[name])
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=8)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps_names:
    plot_color_gradients(cmap_category, cmap_list, nrows)


plt.show()
