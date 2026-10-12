"""Stage: prepare
Raw files -> one modeling table (data/processed/dataset.parquet):
Lehmann subtype labels + 9 routine clinical features + 7 IHC stand-in genes.
"""
import json

import numpy as np
import pandas as pd

from common import PROCESSED, RAW, ihc_columns, load_params


def load_labels(subtypes):
    s1 = pd.read_excel(RAW / "lehmann_2016_S1.xlsx").iloc[1:]  # first row is header text
    labels = pd.DataFrame({
        "TCGA_ID": s1.iloc[:, 0].astype(str).str[:12],          # barcode -> patient ID
        "Subtype": s1.iloc[:, 5].astype(str).str.strip(),       # "Refined TNBCtype"
    })
    labels = labels[labels["TCGA_ID"].str.startswith("TCGA") & labels["Subtype"].isin(subtypes)]
    return labels.drop_duplicates("TCGA_ID").set_index("TCGA_ID")


def load_clinical(params):
    num = params["features"]["clinical_numeric"]
    cat = params["features"]["clinical_categorical"]
    clin = pd.read_csv(RAW / "tcga_brca_clinical_patient.txt", sep="\t", comment="#").set_index("PATIENT_ID")
    clin = clin[num + cat].copy()
    clin[num] = clin[num].apply(pd.to_numeric, errors="coerce")
    clin[cat] = clin[cat].astype("string")
    return clin


def load_ihc(genes):
    expr = pd.read_csv(RAW / "tcga_brca_mrna_rsem.txt", sep="\t")
    expr = expr[expr["Hugo_Symbol"].isin(genes)].drop(columns="Entrez_Gene_Id")
    expr = expr.groupby("Hugo_Symbol").mean()
    expr = expr.loc[:, expr.columns.str.endswith("-01")]       # primary tumors
    expr.columns = expr.columns.str[:12]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    ihc = np.log2(expr.clip(lower=0) + 1).T                    # log2(RSEM + 1)
    ihc.columns = [f"IHC_{g}" for g in ihc.columns]
    return ihc


def main():
    params = load_params()
    labels = load_labels(params["data"]["subtypes"])
    clinical = load_clinical(params)
    ihc = load_ihc(list(params["features"]["ihc_genes"]))

    data = labels.join(clinical, how="inner").join(ihc, how="inner")
    data = data[[c for c in clinical.columns] + ihc_columns(params) + ["Subtype"]]
    data.index.name = "TCGA_ID"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    data.to_parquet(PROCESSED / "dataset.parquet")

    summary = {
        "n_patients": int(len(data)),
        "n_labeled": int(len(labels)),
        "subtype_counts": data["Subtype"].value_counts().to_dict(),
        "age_median": float(data["AGE"].median()),
    }
    (PROCESSED / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
