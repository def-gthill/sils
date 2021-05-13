"""
Tools for plotting languages and language features on the world map
"""

import matplotlib.pyplot as plt
import geopandas as gpd


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


def plot(points, labels=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    world.plot(ax=ax, color='w', edgecolor='k', linewidth=0.5)
    if labels is None:
        points.plot(ax=ax, marker='o', color='b', markersize=25)
    else:
        labelled_points = points.copy()
        labelled_points['label'] = labels
        labelled_points.plot(ax=ax, marker='o', column='label', markersize=25, cmap='Set1')
