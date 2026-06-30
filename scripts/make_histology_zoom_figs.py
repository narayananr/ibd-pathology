"""Build the zoomed + annotated histology teaching figures (deck slides 8B-ZOOM, 8C-ZOOM).

Crops a tight region out of the Wikimedia/Nephron H&E source images and draws leader-line
callouts naming the cells, matching the style of histo_normal_zoom_annotated.png. Colour code
(shared across all "How to read the histology" slides):
    green = epithelium · red = active / neutrophil · blue = lamina propria
    purple = goblet · grey = other / structural

Run from the repo root:  python scripts/make_histology_zoom_figs.py
Outputs into slides/images/. Crop boxes are hard-coded for reproducibility.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "slides", "images")

GRN = "#1e8449"  # epithelium
RED = "#c0392b"  # active / neutrophil
BLU = "#2471a3"  # lamina propria
PUR = "#7d3c98"  # goblet
GRY = "#555555"  # other / structural


def note(ax, xy, xytext, text, color):
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=15, fontweight="bold", color="white", ha="center", va="center",
        annotation_clip=False,
        bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="none"),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6, shrinkA=8, shrinkB=4),
    )


def frame(img, figsize, xlim, ylim, title):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)  # pass inverted (bottom, top) for image coords
    ax.axis("off")
    ax.set_title(title, fontsize=20, fontweight="bold", color=RED, pad=18)
    return fig, ax


def save(fig, name):
    out = os.path.join(IMG, name)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- cryptitis zoom
im = Image.open(os.path.join(IMG, "active_cryptitis_high.jpg")).convert("RGB")
cz = np.array(im.crop((968, 427, 2335, 1794)))  # 1367 x 1367
fig, ax = frame(cz, (12.5, 12.5), (-560, 1960), (1980, -520),
                "ACTIVE — cryptitis, zoomed in (read the cells)")
note(ax, (470, 150), (250, -360),
     "GOBLET CELLS\npale mucin vacuoles\n(now DEPLETED vs normal)", PUR)
note(ax, (980, 1010), (1730, 980), "epithelial NUCLEI\npale · oval", GRN)
note(ax, (150, 700), (-470, 640),
     "DENSE inflammatory infiltrate\n(carpet of small dark nuclei —\nnormal has only a few dots)", RED)
note(ax, (560, 470), (1480, 250),
     "RED BLOOD CELLS\nbright salmon · NO nucleus\n(leaked from damaged vessels)", GRY)
note(ax, (760, 880), (740, 1860),
     "small dark cells crowding\nthe epithelium = CRYPTITIS", RED)
save(fig, "histo_cryptitis_zoom_annotated.png")

# ------------------------------------------------------------- crypt abscess zoom
im = Image.open(os.path.join(IMG, "active_crypt_abscess.jpg")).convert("RGB")
az = np.array(im.crop((150, 150, 610, 560)).resize((920, 820)))  # 920 x 820
fig, ax = frame(az, (12.5, 11.4), (-380, 1320), (1240, -380),
                "ACTIVE — crypt abscess, zoomed in (read the cells)")
note(ax, (414, 451), (430, -270),
     "lumen PACKED with neutrophils\n= CRYPT ABSCESS", RED)
note(ax, (286, 360), (-330, 360),
     "NEUTROPHIL\nsmall · multi-lobed\n(segmented) nucleus", RED)
note(ax, (490, 130), (1110, 120),
     "crypt epithelium\n(gland wall rings the lumen)", GRN)
note(ax, (588, 560), (1170, 470),
     "red blood cells\nbright salmon · NO nucleus", GRY)
note(ax, (140, 715), (130, 1140),
     "lamina propria\n(inflammatory cells all around)", BLU)
save(fig, "histo_abscess_zoom_annotated.png")
