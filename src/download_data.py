"""One-time download of the raw data into data/raw/.

After running this, version the raw data with:  dvc add data/raw
"""
import urllib.request

import requests

from common import RAW

DATAHUB = ("https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
           "public/brca_tcga_pan_can_atlas_2018/")
FILES = {
    "tcga_brca_clinical_patient.txt": DATAHUB + "data_clinical_patient.txt",
    "tcga_brca_mrna_rsem.txt": DATAHUB + "data_mrna_seq_v2_rsem.txt",  # ~150 MB
    "tcga_brca_hypoxia.txt": DATAHUB + "data_clinical_supp_hypoxia.txt",  # Buffa score (original version only)
}
S1_URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
          "&id=10.1371/journal.pone.0157368.s006")
S1_FILE = "lehmann_2016_S1.xlsx"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists():
            print(f"✓ {name} already present")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)

    dest = RAW / S1_FILE
    if dest.exists():
        print(f"✓ {S1_FILE} already present")
    else:
        try:
            r = requests.get(S1_URL, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"Downloaded {S1_FILE}")
        except Exception as e:  # PLOS sometimes blocks scripted downloads
            print(f"Could not download the Lehmann S1 table ({e}).\n"
                  "Download it manually from https://doi.org/10.1371/journal.pone.0157368 "
                  f"(Supporting Information, S1 Table) and save it as data/raw/{S1_FILE}")
    print("Raw data ready in data/raw/")


if __name__ == "__main__":
    main()
