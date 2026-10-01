"""Figure style for the journal: Times New Roman at 8 pt, figures drawn at their printed size.

Jurnal ELTIKOM asks for Times New Roman at 8-11 pt inside figures, so every figure is created at
the width it will have on the page (TEXT_WIDTH = 16 cm for A4 with 2.5 cm margins) and inserted
at 100% in the manuscript.
"""
import matplotlib

TEXT_WIDTH = 16 / 2.54  # inches


def use_paper_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "savefig.dpi": 600,
    })
