"""Schematic of this study's workflow and the proposed CNN next step (manuscript Figure 5)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import PLOTS

FIG = PLOTS / "supplementary"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(12, 6.4))
ax.set_xlim(0, 120); ax.set_ylim(0, 64); ax.axis("off")


def box(x, y, w, h, text, fc, ec="#2f5d8a", fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True,
            fontweight="bold" if bold else "normal", color="#1b2a3a")


def arrow(x1, y1, x2, y2, color="#2f5d8a", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color, lw=1.4, linestyle=ls))


blue, grey, gold, green = "#dce6f0", "#eeeeee", "#fbecc8", "#dff0e1"

# ---------------- A. This study
ax.text(1, 61, "A. This study: mRNA levels of six marker genes stand in for the IHC panel", fontsize=11, fontweight="bold", color="#1f3a5f")
box(1, 44, 19, 12, "TCGA tumor\nRNA sequencing\n(~20,500 genes,\n175 TNBC patients)", grey)
box(25, 44, 21, 12, "Keep 7 genes for the\n6 stains: AR, FOXA1,\nMKI67, KRT5 + KRT6A,\nEGFR, VIM\nlog2(RSEM + 1)", blue)
box(51, 44, 19, 12, "Random forest\n(300 trees)\n5×5 repeated\ncross-validation", blue)
box(75, 44, 20, 12, "Predicted subtype\nBL1 · BL2 · LAR · M\nmacro F1 ≈ 0.60", green, bold=True)
box(100, 44, 19, 12, "Answer key:\nLehmann TNBCtype-4\nlabels (assigned\nfrom tumor RNA)", gold)
for x1, x2 in [(20.8, 24.2), (46.8, 50.2), (70.8, 74.2)]:
    arrow(x1, 50, x2, 50)
arrow(99.2, 50, 95.8, 50, color="#b07d12", ls="--")

# ---------------- B. Next step
ax.text(1, 37, "B. Next step: a CNN reads each IHC slide and estimates its gene's expression", fontsize=11, fontweight="bold", color="#1f3a5f")
box(1, 18, 16, 14, "Routine biopsy\nblock\n6 IHC slides,\nscanned as whole-\nslide images", grey)
box(21, 18, 18, 14, "Preprocess\n• find tumor tissue\n• cut into tiles\n• normalize color\n• separate brown\n  (DAB) stain", blue, fs=8.2)
box(43, 18, 18, 14, "CNN per stain\n(pretrained encoder +\nattention pooling\nover tiles)", blue)
box(65, 18, 18, 14, "Estimated expression\nof the matching gene\n(e.g., AR slide →\nAR mRNA level)", gold)
box(87, 18, 14, 14, "Same subtype\nclassifier\n(random forest)", blue)
box(105, 18, 14, 14, "Predicted\nsubtype\nBL1 · BL2 ·\nLAR · M", green, bold=True)
for x1, x2 in [(17.8, 20.2), (39.8, 42.2), (61.8, 64.2), (83.8, 86.2), (101.8, 104.2)]:
    arrow(x1, 25, x2, 25)
box(31, 2, 64, 10, "Training cohort only: tumors with BOTH IHC slides and RNA-seq.\nRNA-seq gives each slide's target (measured gene expression)\nand the Lehmann subtype. New patients need only the slides.", "#fff7e6", ec="#b07d12", fs=8.2)
arrow(63, 12.6, 63, 17.2, color="#b07d12", ls="--")
plt.savefig(FIG / "fig_workflow_cnn.png", dpi=200, bbox_inches="tight"); plt.close()
print("wrote", FIG / "fig_workflow_cnn.png")
