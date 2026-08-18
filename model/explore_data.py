from pathlib import Path

import kagglehub
import pandas as pd

dataset_path = kagglehub.dataset_download("chaitanyakck/medical-text")
print(f"Dataset baixado em: {dataset_path}\n")

files = list(Path(dataset_path).rglob("*"))
print("Arquivos encontrados:")
for f in files:
    if f.is_file():
        print(f" - {f.name} ({f.stat().st_size / 1024:.1f}) KB")

print("\n --- Inspecionando arquivos \n")
for f in files:
    if f.suffix.lower() in (".csv", ".dat", ".tsv", ".txt"):
        print (f"## {f.name} ##")
        try:

            try:
                df = pd.read_csv(f, nrows=5)
            except Exception:
                df = pd.read_csv(f, sep="\t", header=None, nrows=5)
            print("colunas:", list(df.columns))
            print(df.head(5))
        except Exception as e:
            print(f"Não foi possível ler: {e}")
        print()
