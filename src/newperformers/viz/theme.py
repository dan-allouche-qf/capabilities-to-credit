"""Matplotlib and plotly themes. Apply once at module load via apply()."""

from __future__ import annotations

from cycler import cycler
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from . import palette as pal

_APPLIED = False


def _serif_family() -> list[str]:
    return [
        "Source Serif Pro",
        "Source Serif 4",
        "EB Garamond",
        "Charter",
        "DejaVu Serif",
        "serif",
    ]


def _sans_family() -> list[str]:
    return ["Inter", "Source Sans Pro", "Source Sans 3", "Helvetica", "Arial", "sans-serif"]


def _register_colormaps() -> None:
    neg, mid, pos = pal.diverging_anchors()
    div = LinearSegmentedColormap.from_list("np_diverging", [neg, mid, pos], N=256)
    low, high = pal.sequential_anchors()
    seq = LinearSegmentedColormap.from_list("np_sequential", [low, high], N=256)
    for cmap in (div, seq):
        try:
            mpl.colormaps.register(cmap)
        except ValueError:
            pass  # already registered


def apply(*, for_paper: bool = True) -> None:
    """Apply the matplotlib rcParams for the rest of the process."""
    global _APPLIED
    if _APPLIED:
        return
    _register_colormaps()

    rc = {
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "savefig.facecolor": "white",

        "font.family": _serif_family() if for_paper else _sans_family(),
        "font.size": 10.5 if for_paper else 11.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": pal.hex("dark"),
        "axes.linewidth": 0.8,
        "axes.labelcolor": pal.hex("dark"),
        "axes.titlecolor": pal.hex("dark"),
        "text.color": pal.hex("dark"),
        "xtick.color": pal.hex("dark"),
        "ytick.color": pal.hex("dark"),

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": pal.hex("gridline"),
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "grid.alpha": 0.9,

        "axes.prop_cycle": cycler(color=pal.cycle()),

        "legend.frameon": False,
        "legend.borderaxespad": 0.5,
        "legend.handlelength": 1.4,

        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    mpl.rcParams.update(rc)
    _APPLIED = True


def plotly_template() -> dict:
    """Plotly template aligned with the matplotlib theme. Lazily built."""
    return {
        "layout": {
            "font": {"family": "Inter, Source Sans Pro, Helvetica, sans-serif",
                     "size": 13, "color": pal.hex("dark")},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "colorway": pal.cycle(),
            "xaxis": {
                "showgrid": False, "zeroline": False,
                "linecolor": pal.hex("dark"), "ticks": "outside",
                "tickcolor": pal.hex("dark"),
            },
            "yaxis": {
                "gridcolor": pal.hex("gridline"), "zeroline": False,
                "linecolor": pal.hex("dark"), "ticks": "outside",
                "tickcolor": pal.hex("dark"),
            },
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
            "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        }
    }
