import dynalearn
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple



locations = {
    "center center": (0.5, 0.5, "center", "center"),
    "upper right": (0.95, 0.95, "top", "right"),
    "lower right": (0.95, 0.05, "bottom", "right"),
    "upper left": (0.05, 0.95, "top", "left"),
    "lower left": (0.05, 0.05, "bottom", "left"),
}

color_dark = {
    "blue": "#1f77b4",
    "orange": "#f19143",
    "purple": "#9A80B9",
    "red": "#d73027",
    "grey": "#525252",
    "green": "#33b050",
}

color_pale = {
    "blue": "#7bafd3",
    "orange": "#f7be90",
    "purple": "#c3b4d6",
    "red": "#e78580",
    "grey": "#999999",
    "green": "#9fdaac",
}

colormap = "bone"

m_list = ["o", "s", "v", "^"]
l_list = ["solid", "dashed", "dotted", "dashdot"]
cd_list = [
    color_dark["blue"],
    color_dark["orange"],
    color_dark["purple"],
    color_dark["red"],
]
cp_list = [
    color_pale["blue"],
    color_pale["orange"],
    color_pale["purple"],
    color_pale["red"],
]

large_fontsize=18
small_fontsize=14

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({'text.latex.preamble' : [r'\usepackage{amsmath}']})

def label_plot(ax, label, loc="center center", fontsize=large_fontsize):
    if isinstance(loc, tuple):
        h, v, va, ha = loc
    elif isinstance(loc, str):
        h, v, va, ha = locations[loc]
    ax.text(h, v, label, color="k", transform=ax.transAxes, 
        verticalalignment=va, horizontalalignment=ha, fontsize=fontsize,
    )