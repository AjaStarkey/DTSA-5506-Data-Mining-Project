"""Manuscript figures built from saved metrics (run after supplementary.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import METRICS, PLOTS

FIG = PLOTS / "supplementary"
FIG.mkdir(parents=True, exist_ok=True)
S = METRICS / "supplementary"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Figure: macro F1 by feature set and model, with transcriptome reference
cv = pd.read_csv(METRICS / "cv_results.csv")
rna = pd.read_csv(S / "rna_reference.csv").iloc[0]
names = {"clinical": "Routine clinical\n(9 features)", "ihc_panel": "Six-stain IHC panel\n(7 mRNA stand-ins)", "clinical_ihc": "Clinical + IHC panel\n(16 features)"}
order = ["clinical", "ihc_panel", "clinical_ihc"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(order) + 1)
w = 0.38
for k, (m, col, lab) in enumerate([("logistic_regression", "#9db4cc", "Logistic regression"),
                                   ("random_forest", "#2f5d8a", "Random forest")]):
    d = cv[cv.model == m].set_index("feature_set").loc[order]
    ax.bar(x[:-1] + (k - 0.5) * w, d.f1_macro_mean, w, yerr=d.f1_macro_sd, capsize=3, color=col, label=lab)
    for xi, v in zip(x[:-1] + (k - 0.5) * w, d.f1_macro_mean):
        ax.text(xi, v + 0.09, f"{v:.2f}", ha="center", fontsize=8)
ax.bar(x[-1], rna.f1_macro_mean, w, yerr=rna.f1_macro_sd, capsize=3, color="#b9b9b9", hatch="//", label="Whole-transcriptome reference (LR)")
ax.text(x[-1], rna.f1_macro_mean + 0.08, f"{rna.f1_macro_mean:.2f}", ha="center", fontsize=8)
ax.axhline(0.25, ls="--", color="#c0392b", lw=1)
ax.text(2.55, 0.265, "chance ≈ 0.25", color="#c0392b", fontsize=8, ha="center")
ax.set_xticks(x); ax.set_xticklabels([names[o] for o in order] + ["Top 1,000 genes\n(RNA-seq reference)"], fontsize=8)
ax.set_ylabel("Macro F1 (mean ± SD, 25 CV folds)"); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig_model_comparison.png", dpi=200); plt.close()

# Figure: panel size vs. performance and cost; coarse scoring
p = pd.read_csv(S / "panels.csv")
c = pd.read_csv(S / "coarse_scoring.csv")
cost = {1: 102.06}
for n in range(2, 7):
    cost[n] = 102.06 + 87.51 * (n - 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
p = p.sort_values("f1_macro_mean")
colors = ["#9db4cc" if n == 1 else ("#2f5d8a" if "Full" in s else "#5b86b0") for n, s in zip(p.n_stains, p.panel)]
ax.barh(p.panel, p.f1_macro_mean, xerr=p.f1_macro_sd, color=colors, capsize=2)
for i, (v, sd, n) in enumerate(zip(p.f1_macro_mean, p.f1_macro_sd, p.n_stains)):
    ax.text(0.02, i, f"${cost[n]:,.0f}", va="center", color="white", fontsize=7.5, fontweight="bold")
    ax.text(v + sd + 0.015, i, f"{v:.2f}", va="center", fontsize=8)
ax.axvline(0.25, ls="--", color="#c0392b", lw=1)
ax.set_xlim(0, 0.85); ax.set_xlabel("Macro F1 (random forest, 25 CV folds)")
ax.set_title("A. Single stains and smaller panels\n(white label = approx. Medicare IHC fee)", fontsize=10, loc="left")
ax.tick_params(axis="y", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
lab = ["Continuous\n(main analysis)", "4 levels", "3 levels", "2 levels\n(pos/neg)"]
ax.bar(lab, c.f1_macro_mean, yerr=c.f1_macro_sd, color=["#2f5d8a", "#5b86b0", "#7fa3c4", "#a9c2da"], capsize=3)
for i, (v, sd) in enumerate(zip(c.f1_macro_mean, c.f1_macro_sd)):
    ax.text(i, v + sd + 0.015, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0.25, ls="--", color="#c0392b", lw=1)
ax.set_ylim(0, 0.85); ax.set_ylabel("Macro F1")
ax.set_title("B. Coarser, IHC-like scoring of the\nsix-stain panel", fontsize=10, loc="left")
ax.tick_params(axis="x", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig_panels_scoring.png", dpi=200); plt.close()

# Figure: SEER 5-year relative survival by subtype and stage
seer = pd.DataFrame({
    "HR+/HER2-": [100.0, 91.1, 38.1, 95.8], "HR+/HER2+": [99.5, 91.5, 48.7, 92.2],
    "HR-/HER2+": [97.8, 86.4, 43.1, 87.0], "Triple-negative": [92.8, 68.3, 14.9, 78.7]},
    index=["Localized", "Regional", "Distant", "All stages"])
fig, ax = plt.subplots(figsize=(8.5, 3.8))
x = np.arange(4); w = 0.2
cols = ["#cfcfcf", "#a8a8a8", "#7d7d7d", "#c0392b"]
for k, s in enumerate(seer.columns):
    ax.bar(x + (k - 1.5) * w, seer[s], w, color=cols[k], label=s)
    for xi, v in zip(x + (k - 1.5) * w, seer[s]):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, fontweight="bold" if k == 3 else "normal")
ax.set_xticks(x); ax.set_xticklabels(seer.index)
ax.set_ylabel("5-year relative survival (%)"); ax.set_ylim(0, 112)
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig_seer_survival.png", dpi=200); plt.close()
print("figures written to", FIG)

# Figure: scoring detail and measurement noise; candidate extra markers
det = pd.read_csv(S / "scoring_detail.csv")
cand = pd.read_csv(S / "candidate_markers.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
lv = det[det.test == "levels"].copy()
lab = [str(s) if s != "continuous" else "cont." for s in lv.setting]
ax.errorbar(range(len(lv)), lv.f1_macro_mean, yerr=lv.f1_macro_sd, marker="o", color="#2f5d8a", capsize=3, label="Recorded in fewer levels")
for i, v in enumerate(lv.f1_macro_mean):
    ax.text(i, v + 0.085, f"{v:.2f}", ha="center", fontsize=8)
nz = det[det.test == "noise_sd"]
txt = "Continuous values with added\nmeasurement error (log2 scale):\n" + "\n".join(f"  error SD {sd}: macro F1 {v:.2f}" for v, sd in zip(nz.f1_macro_mean, nz.setting))
ax.text(len(lv) - 1, 0.04, txt, color="#c0392b", fontsize=8, ha="right", va="bottom")
ax.axhline(0.25, ls="--", color="grey", lw=0.8)
ax.set_xticks(range(len(lv))); ax.set_xticklabels(lab)
ax.set_xlabel("Number of levels each stain is recorded in"); ax.set_ylabel("Macro F1 (mean ± SD, 25 folds)")
ax.set_ylim(0, 0.8); ax.set_title("A. How much scoring detail the model needs", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
c = cand[~cand.added.str.startswith("none")].sort_values("gain").tail(10)
cols = ["#2f5d8a" if "+" in a else ("#9db4cc" if a.startswith("none") else "#5b86b0") for a in c.added]
ax.barh(c.added, c.macro_f1, color=cols)
for i, (v, bl2, m) in enumerate(zip(c.macro_f1, c.BL2, c.M)):
    ax.text(v + 0.005, i, f"{v:.2f}  (BL2 {bl2:.2f}, M {m:.2f})", va="center", fontsize=7.5)
base = float(cand[cand.added.str.startswith("none")].macro_f1.iloc[0])
ax.axvline(base, color="grey", ls="--", lw=0.8); ax.text(base + 0.002, 9.55, f"six-stain\npanel {base:.2f}", fontsize=7.5, color="grey")
ax.set_xlim(0.5, 0.82); ax.set_xlabel("Macro F1 with the gene added to the six-stain panel")
ax.set_title("B. Candidate seventh markers (top 9 of 22) and\nthe best pair; per-subtype F1 for BL2 and M", fontsize=10, loc="left")
ax.tick_params(axis="y", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig_detail_markers.png", dpi=200); plt.close()
print("wrote fig_detail_markers.png")
