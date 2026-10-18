import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
#fig_dir = "/home/ns8802/figures/growth_diss_paper/"
#os.makedirs(fig_dir, exist_ok=True)

print("PATH =", os.environ["PATH"])
print("latex   =", shutil.which("latex"))
print("pdflatex=", shutil.which("pdflatex"))
print("dvipng  =", shutil.which("dvipng"))

# ----------------------------------------------------
# General options
# ----------------------------------------------------
do_show = 0

cm_u = 1.0 / 2.54
wid = 12.0
hei = 8.5
pad_in = 0.1

line_width = 2.0
fontsize_leg = 14
fontsize_lab = 14
fontsize_gen = 14

# ----------------------------------------------------
# Figure style
# ----------------------------------------------------
def set_fig_params():
    plt.rcParams.update(plt.rcParamsDefault)

    plt.rcParams.update({
        # Use real LaTeX
        "text.usetex": True,

        # Use LaTeX-style serif font
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",

        # Optional LaTeX preamble
        "text.latex.preamble": r"\usepackage{amsmath}",

        # Font sizes
        "font.size": fontsize_gen,
        "axes.labelsize": fontsize_lab,
        "xtick.labelsize": fontsize_lab,
        "ytick.labelsize": fontsize_lab,
        "legend.fontsize": fontsize_leg,

        # Line widths
        "lines.linewidth": line_width,
        "axes.linewidth": 1.5,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,

        # Spines
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # Colors
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",

        # Save settings
        "savefig.bbox": "tight",
        "savefig.pad_inches": pad_in,
    })


#
# Define colormaps
#
#map_cm = 'Blues'; 
#colors_v1 = get_colormap(map_cm,ext_v=np.array([1,5]),minmax_v=np.array([0.4,1]),st=1);
#map_cm = 'Reds'; 
#colors_v2 = get_colormap(map_cm,ext_v=np.array([1,4]),minmax_v=np.array([0.4,1]),st=1);
#
#colors = np.concatenate((colors_v1, colors_v2))