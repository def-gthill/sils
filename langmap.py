"""
Tools for plotting languages and language features on the world map
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Adapted from Kenneth Kelly's maximally contrastive colour list
colors = np.array(
    [
        [0, 0, 0],
        [255, 179, 0],
        [143, 62, 128],
        [255, 104, 0],
        [166, 189, 215],
        [193, 0, 32],
        [206, 162, 98],
        [129, 112, 102],
        [246, 118, 142],
        [0, 83, 138],
        [83, 55, 122],
        [0, 125, 52],
        [255, 122, 92],
        [179, 40, 81],
        [244, 200, 0],
        [127, 24, 13],
        [147, 170, 0],
        [255, 142, 0],
        [89, 51, 21],
        [241, 58, 19],
        [35, 44, 22],
    ]
) / 256


def plot(points, labels=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    world.plot(ax=ax, color='w', edgecolor='k', linewidth=0.5)
    if labels is None:
        points.plot(ax=ax, marker='o', color='b', markersize=35)
    else:
        labelled_points = points.copy()
        labelled_points['label'] = labels
        num_unique_labels = len(np.unique(labels))
        cmap = ListedColormap(colors[:num_unique_labels])
        labelled_points.plot(
            ax=ax, marker='o', column='label', markersize=35,
            cmap=cmap, categorical=True,
            legend=True, legend_kwds={
                'loc': 'lower left'
            }
        )
