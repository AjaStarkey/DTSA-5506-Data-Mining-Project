"""Supplementary analyses for the manuscript (not part of the DVC pipeline).

Writes results to metrics/supplementary/ and figures to plots/supplementary/.
Run from the project root:  python src/supplementary.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from common import (METRICS, PLOTS, RAW, ihc_columns, load_dataset, load_params,
                    make_model, make_pipeline, preprocessor)

OUT = METRICS / "supplementary"
FIG = PLOTS / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

params = load_params()
SEED = params["cv"]["random_state"]
SUB = params["data"]["subtypes"]
data = load_dataset(params)
y = data["Subtype"].to_numpy(dtype=object)
IHC = ihc_columns(params)
STAIN = {"IHC_AR": "AR", "IHC_FOXA1": "FOXA1", "IHC_MKI67": "Ki-67", "IHC_KRT5": "CK5 (KRT5)",
         "IHC_KRT6A": "CK6 (KRT6A)", "IHC_EGFR": "EGFR", "IHC_VIM": "Vimentin"}
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
which = sys.argv[1:] or ["table1", "oof", "binary", "coarse", "single", "rna", "orig_ihc", "detail", "markers", "perm"]


def rf_pipe(cols, binner=None):
    steps = [("prep", preprocessor(cols, []))]
    if binner is not None:
        steps = [("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                                ("bin", binner)]), cols)]))]
    steps.append(("model", make_model("random_forest", params)))
    return Pipeline(steps)


def cv_f1(pipe, X, cv=rcv):
    s = cross_validate(pipe, X, y, cv=cv, scoring=["f1_macro", "accuracy"])
    return {"f1_macro_mean": round(s["test_f1_macro"].mean(), 4), "f1_macro_sd": round(s["test_f1_macro"].std(), 4),
            "accuracy_mean": round(s["test_accuracy"].mean(), 4)}


# ---------------------------------------------------------------- Table 1
if "table1" in which:
    rows = []
    def add(name, by_sub, overall, p):
        rows.append({"Characteristic": name, **{s: by_sub[s] for s in SUB}, "All": overall, "p": p})
    g = data.groupby("Subtype")
    add("Patients, n", {s: str((y == s).sum()) for s in SUB}, str(len(y)), "")
    med = lambda v: f"{np.nanmedian(v):.0f} ({np.nanpercentile(v,25):.0f}–{np.nanpercentile(v,75):.0f})"
    p_age = stats.kruskal(*[data.loc[y == s, "AGE"].dropna() for s in SUB]).pvalue
    add("Age, median (IQR)", {s: med(data.loc[y == s, "AGE"]) for s in SUB}, med(data["AGE"]), f"{p_age:.3f}")
    def stage_group(v):
        if pd.isna(v): return "Unknown"
        v = str(v).upper().replace("STAGE ", "")
        for k in ["IV", "III", "II", "I"]:
            if v.startswith(k): return f"Stage {k}"
        return "Unknown"
    st = data["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(stage_group)
    def hist(v):
        v = str(v)
        return {"8500/3": "Ductal (IDC)", "8520/3": "Lobular (ILC)", "8575/3": "Metaplastic", "8510/3": "Medullary"}.get(v, "Other/unknown")
    hi = data["ICD_O_3_HISTOLOGY"].map(hist)
    nodes = data["PATH_N_STAGE"].astype(str).str.upper().str[:2].map(lambda v: "N0" if v == "N0" else ("N1–N3" if v in ("N1", "N2", "N3") else "Unknown"))
    race = data["RACE"].fillna("Unknown").replace({"Black or African American": "Black", "White": "White", "Asian": "Asian"})
    race = race.where(race.isin(["White", "Black", "Asian"]), "Other/unknown")
    for label, ser, order in [("Stage", st, ["Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"]),
                              ("Lymph nodes", nodes, ["N0", "N1–N3", "Unknown"]),
                              ("Histology", hi, ["Ductal (IDC)", "Metaplastic", "Medullary", "Lobular (ILC)", "Other/unknown"]),
                              ("Race", race, ["White", "Black", "Asian", "Other/unknown"])]:
        ct = pd.crosstab(ser, y)
        known = ct.drop(index=[i for i in ct.index if "nknown" in i], errors="ignore")
        known = known.loc[known.sum(axis=1) > 0]
        p = stats.chi2_contingency(known)[1] if known.shape[0] > 1 else np.nan
        first = True
        for lvl in order:
            if lvl not in ct.index: continue
            cnt = ct.loc[lvl]
            add(f"{label}: {lvl}", {s: f"{cnt.get(s,0)} ({100*cnt.get(s,0)/(y==s).sum():.0f}%)" for s in SUB},
                f"{cnt.sum()} ({100*cnt.sum()/len(y):.0f}%)", f"{p:.3f}" if first else "")
            first = False
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "table1.csv", index=False)
    print(t1.to_string())

    # Marker expression by subtype (Kruskal-Wallis + median per subtype)
    mk = []
    for c in IHC:
        p = stats.kruskal(*[data.loc[y == s, c].dropna() for s in SUB]).pvalue
        mk.append({"marker": STAIN[c], **{s: round(data.loc[y == s, c].median(), 2) for s in SUB}, "kruskal_p": f"{p:.1e}"})
    mk = pd.DataFrame(mk)
    mk.to_csv(OUT / "marker_by_subtype.csv", index=False)
    print(mk.to_string())
    long = data[IHC + ["Subtype"]].melt(id_vars="Subtype", var_name="marker", value_name="expr")
    long["marker"] = long["marker"].map(STAIN)
    fig, axes = plt.subplots(1, 7, figsize=(15, 3.4), sharey=False)
    pal = {"BL1": "#c0392b", "BL2": "#e67e22", "LAR": "#2980b9", "M": "#27ae60"}
    for ax, (m, d) in zip(axes, long.groupby("marker", sort=False)):
        sns.boxplot(data=d, x="Subtype", y="expr", order=SUB, hue="Subtype", palette=pal, ax=ax, fliersize=2, legend=False)
        ax.set_title(m, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("log2(RSEM+1)" if ax is axes[0] else "")
        ax.tick_params(labelsize=8)
    plt.tight_layout(); plt.savefig(FIG / "fig_markers_by_subtype.png", dpi=200); plt.close()

# ---------------------------------------------------------------- OOF per-class with bootstrap CIs
if "oof" in which:
    res, cms = {}, {}
    rng = np.random.default_rng(SEED)
    for fs in ["clinical_original", "clinical", "ihc_panel", "clinical_ihc"]:
        pipe, cols = make_pipeline(fs, "random_forest", params)
        pred = cross_val_predict(pipe, data[cols], y, cv=cv5)
        rep = classification_report(y, pred, labels=SUB, output_dict=True, zero_division=0)
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            boots.append([f1_score(y[i], pred[i], labels=SUB, average="macro", zero_division=0)] +
                         list(f1_score(y[i], pred[i], labels=SUB, average=None, zero_division=0)))
        b = np.array(boots)
        lo, hi = np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
        r = {"macro_f1": round(rep["macro avg"]["f1-score"], 3), "macro_ci": [round(lo[0], 3), round(hi[0], 3)],
             "accuracy": round(rep["accuracy"], 3)}
        for k, s in enumerate(SUB):
            r[s] = {"precision": round(rep[s]["precision"], 3), "recall": round(rep[s]["recall"], 3),
                    "f1": round(rep[s]["f1-score"], 3), "f1_ci": [round(lo[k+1], 3), round(hi[k+1], 3)]}
        res[fs] = r
        cms[fs] = confusion_matrix(y, pred, labels=SUB)
    (OUT / "oof_per_class.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = {"clinical_original": "A. Original 12 clinical features", "clinical": "B. Routine clinical (9)",
              "ihc_panel": "C. Six-stain IHC panel"}
    for ax, fs in zip(axes, titles):
        cm = cms[fs]; norm = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{norm[i,j]:.2f}\n(n={cm[i,j]})" for j in range(4)] for i in range(4)])
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=SUB, yticklabels=SUB, ax=ax, annot_kws={"fontsize": 8})
        ax.set_title(f"{titles[fs]}\nmacro F1 = {res[fs]['macro_f1']:.2f}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True (Lehmann)")
    plt.tight_layout(); plt.savefig(FIG / "fig_confusion.png", dpi=200); plt.close()

# ---------------------------------------------------------------- binary clinical questions
if "binary" in which:
    tasks = {"LAR vs. other": ["LAR"], "Basal-like (BL1+BL2) vs. other": ["BL1", "BL2"],
             "M vs. other": ["M"], "BL1 vs. other": ["BL1"], "BL2 vs. other": ["BL2"]}
    rows = []
    for fs in ["clinical", "ihc_panel"]:
        cols = ihc_columns(params) if fs == "ihc_panel" else make_pipeline(fs, "random_forest", params)[1]
        for t, pos in tasks.items():
            yb = np.isin(y, pos).astype(int)
            pipe = make_pipeline(fs, "random_forest", params)[0]
            s = cross_validate(pipe, data[cols], yb, cv=rcv, scoring=["roc_auc", "f1", "balanced_accuracy"])
            rows.append({"feature_set": fs, "task": t, "positives": int(yb.sum()),
                         "auc_mean": round(s["test_roc_auc"].mean(), 3), "auc_sd": round(s["test_roc_auc"].std(), 3),
                         "f1_pos_mean": round(s["test_f1"].mean(), 3),
                         "bal_acc_mean": round(s["test_balanced_accuracy"].mean(), 3)})
    b = pd.DataFrame(rows); b.to_csv(OUT / "binary_tasks.csv", index=False); print(b.to_string())

# ---------------------------------------------------------------- coarse (IHC-like) scoring
if "coarse" in which:
    rows = [{"scoring": "Continuous mRNA (main analysis)", **cv_f1(rf_pipe(IHC), data[IHC])}]
    for k, name in [(4, "4 levels (like 0 / 1+ / 2+ / 3+)"), (3, "3 levels (low / medium / high)"),
                    (2, "2 levels (negative / positive)")]:
        binner = KBinsDiscretizer(n_bins=k, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf") \
            if "quantile_method" in KBinsDiscretizer().get_params() else KBinsDiscretizer(n_bins=k, encode="ordinal", strategy="quantile")
        rows.append({"scoring": name, **cv_f1(rf_pipe(IHC, binner), data[IHC])})
    c = pd.DataFrame(rows); c.to_csv(OUT / "coarse_scoring.csv", index=False); print(c.to_string())

# ---------------------------------------------------------------- single stains and small panels
if "single" in which:
    groups = {"AR": ["IHC_AR"], "FOXA1": ["IHC_FOXA1"], "Ki-67": ["IHC_MKI67"], "CK5/6": ["IHC_KRT5", "IHC_KRT6A"],
              "EGFR": ["IHC_EGFR"], "Vimentin": ["IHC_VIM"]}
    rows = [{"panel": f"{k} only", "n_stains": 1, **cv_f1(rf_pipe(v), data[v])} for k, v in groups.items()]
    for name, ks in [("AR + Vimentin", ["AR", "Vimentin"]),
                     ("AR + Vimentin + CK5/6", ["AR", "Vimentin", "CK5/6"]),
                     ("AR + Vimentin + CK5/6 + Ki-67", ["AR", "Vimentin", "CK5/6", "Ki-67"]),
                     ("AR + Vimentin + CK5/6 + EGFR", ["AR", "Vimentin", "CK5/6", "EGFR"]),
                     ("Prior-literature trio: AR + CK5/6 + EGFR", ["AR", "CK5/6", "EGFR"]),
                     ("Full six-stain panel", list(groups))]:
        cols = sum([groups[k] for k in ks], [])
        rows.append({"panel": name, "n_stains": len(ks), **cv_f1(rf_pipe(cols), data[cols])})
    s = pd.DataFrame(rows); s.to_csv(OUT / "panels.csv", index=False); print(s.to_string())

# ---------------------------------------------------------------- whole-transcriptome reference
if "rna" in which:
    expr = pd.read_csv(RAW / "tcga_brca_mrna_rsem.txt", sep="\t").dropna(subset=["Hugo_Symbol"])
    expr = expr.drop(columns="Entrez_Gene_Id").groupby("Hugo_Symbol").mean()
    expr = expr.loc[:, expr.columns.str.endswith("-01")]
    expr.columns = expr.columns.str[:12]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    X = np.log2(expr.clip(lower=0) + 1).T.reindex(data.index)
    X = X.loc[:, X.notna().all()]
    rows = []
    for k in [1000]:
        pipe = Pipeline([("var", VarianceThreshold(0.5)), ("select", SelectKBest(f_classif, k=k)),
                         ("scale", StandardScaler()),
                         ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))])
        rows.append({"features": f"Top {k} genes (selected within each fold), ~{X.shape[1]} measured",
                     **cv_f1(pipe, X.values)})
    r = pd.DataFrame(rows); r.to_csv(OUT / "rna_reference.csv", index=False); print(r.to_string())

# ---------------------------------------------------------------- original 12 clinical features + IHC panel
if "orig_ihc" in which:
    f = params["features"]
    num = f["clinical_numeric"] + f["original_extra_numeric"] + IHC
    cat = f["clinical_categorical"] + f["original_extra_categorical"]
    rows = []
    for m in ["logistic_regression", "random_forest"]:
        pipe = Pipeline([("prep", preprocessor(num, cat)), ("model", make_model(m, params))])
        rows.append({"feature_set": "clinical_original_ihc", "model": m, **cv_f1(pipe, data[num + cat])})
    o = pd.DataFrame(rows); o.to_csv(OUT / "original12_plus_ihc.csv", index=False); print(o.to_string())

# ---------------------------------------------------------------- scoring detail and measurement noise
if "detail" in which:
    rows = []
    for k in [2, 3, 4, 6, 10, 20, None]:
        qm = {"quantile_method": "averaged_inverted_cdf"} if "quantile_method" in KBinsDiscretizer().get_params() else {}
        binner = None if k is None else KBinsDiscretizer(n_bins=k, encode="ordinal", strategy="quantile", **qm)
        rows.append({"test": "levels", "setting": "continuous" if k is None else k, **cv_f1(rf_pipe(IHC, binner), data[IHC])})
    rng = np.random.default_rng(SEED)
    for sd in [0.5, 1.0, 1.5]:
        noisy = data[IHC] + rng.normal(0, sd, data[IHC].shape)   # random error added to every value (log2 scale)
        rows.append({"test": "noise_sd", "setting": sd, **cv_f1(rf_pipe(IHC), noisy)})
    r = pd.DataFrame(rows); r.to_csv(OUT / "scoring_detail.csv", index=False); print(r.to_string())

# ---------------------------------------------------------------- candidate extra markers for BL2 and M
if "markers" in which:
    # Candidate genes chosen in advance from the literature, all with established IHC antibodies.
    cands = {"CD8A": "CD8 (cytotoxic T cells; FUSCC IHC marker)", "FOXC1": "FOXC1 (FUSCC IHC marker)", "DCLK1": "DCLK1 (FUSCC IHC marker)",
             "TP63": "p63 (myoepithelial)", "MME": "CD10 (myoepithelial)", "MET": "MET (growth factor receptor)", "KIT": "c-KIT (CD117)",
             "SOX10": "SOX10 (neural crest/basal)", "NGFR": "NGFR (CD271)", "ALDH1A1": "ALDH1 (stem-like)",
             "CLDN3": "claudin-3", "CLDN4": "claudin-4", "CLDN7": "claudin-7", "EPCAM": "EpCAM", "CDH1": "E-cadherin",
             "CDH2": "N-cadherin", "ZEB1": "ZEB1 (EMT)", "SNAI2": "SLUG (EMT)", "TWIST1": "TWIST1 (EMT)", "FN1": "fibronectin",
             "PDGFRB": "PDGFR-beta", "TGFB1": "TGF-beta1"}
    expr = pd.read_csv(RAW / "tcga_brca_mrna_rsem.txt", sep="\t")
    expr = expr[expr["Hugo_Symbol"].isin(cands)].drop(columns="Entrez_Gene_Id").groupby("Hugo_Symbol").mean()
    expr = expr.loc[:, expr.columns.str.endswith("-01")]
    expr.columns = expr.columns.str[:12]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    extra = np.log2(expr.clip(lower=0) + 1).T.reindex(data.index)
    extra.columns = [f"X_{c}" for c in extra.columns]
    D = pd.concat([data, extra], axis=1)

    def per_class(cols, reps=5):
        out = []
        for r in range(reps):
            pred = cross_val_predict(rf_pipe(cols), D[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + r))
            out.append([f1_score(y, pred, average="macro")] + list(f1_score(y, pred, labels=SUB, average=None)))
        return np.mean(out, axis=0)

    base = per_class(IHC)
    rows = [{"added": "none (six-stain panel)", "protein": "", "macro_f1": base[0], **dict(zip(SUB, base[1:]))}]
    for g, prot in cands.items():
        if f"X_{g}" not in D: continue
        s_ = per_class(IHC + [f"X_{g}"])
        rows.append({"added": g, "protein": prot, "macro_f1": s_[0], **dict(zip(SUB, s_[1:]))})
    m = pd.DataFrame(rows)
    m["gain"] = m["macro_f1"] - base[0]
    best2 = m.iloc[1:].sort_values("gain", ascending=False)["added"].head(2).tolist()
    s_ = per_class(IHC + [f"X_{g}" for g in best2])
    m.loc[len(m)] = {"added": " + ".join(best2), "protein": "top two combined", "macro_f1": s_[0], **dict(zip(SUB, s_[1:])), "gain": s_[0] - base[0]}
    m = m.round(3).sort_values("gain", ascending=False)
    m.to_csv(OUT / "candidate_markers.csv", index=False); print(m.to_string())
    med = D.groupby("Subtype")[[f"X_{g}" for g in best2]].median().round(2)
    med.to_csv(OUT / "candidate_markers_medians.csv"); print(med)

# ---------------------------------------------------------------- permutation tests
if "perm" in which:
    rows = []
    for fs in ["clinical", "clinical_original", "ihc_panel"]:
        pipe, cols = make_pipeline(fs, "random_forest", params)
        score, perm, p = permutation_test_score(pipe, data[cols], y, cv=cv5, scoring="f1_macro",
                                                n_permutations=100, random_state=SEED)
        rows.append({"feature_set": fs, "f1_macro": round(score, 3), "null_mean": round(perm.mean(), 3),
                     "null_95th": round(np.percentile(perm, 95), 3), "p_value": round(p, 4)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "permutation_tests.csv", index=False)
