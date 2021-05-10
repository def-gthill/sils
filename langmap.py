"""
Tools for plotting languages and language features on the world map
"""

import matplotlib.pyplot as plt
import geopandas as gpd


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


def plot(points):
    fig, ax = plt.subplots(figsize=(10, 16))
    world.plot(ax=ax, color='w', edgecolor='k', linewidth=0.5)
    points.plot(ax=ax, marker='o', color='b', markersize=5)