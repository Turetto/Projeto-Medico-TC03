import json
from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

LABEL_MAP = {
    1: "neoplasms",
    2: "digestive_system_diseases",
    3: "nervous_system_diseases",
    4: "cardiovascular_diseases",
    5: "general_pathological_conditions",
}

ARTIFACTS_DIR = Path("model/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_train_data() -> pd.DataFrame:
    """
    Baixar o dataset e ler o train.dat
    """
    dataset_path = Path(kagglehub.dataset_download("chaitanyakck/medical-text"))
    train_file = dataset_path / "train.dat"

    rows = []
    with open(train_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            label_str, text = line.split("\t", 1)
            try:
                label = int(label_str)
            except ValueError:
                continue
            rows.append({"condition_label": label, "medical_abstract": text})

    df = pd.DataFrame(rows)
    df["condition_name"] = df["condition_label"].map(LABEL_MAP)
    return df


def main():
    print("carregando dados...")
    df = load_train_data()
    print(f"total de registros: {len(df)}")
    print("distribuição das clases:")
    print(df["condition_name"].value_counts())

    x = df["medical_abstract"]
    y = df["condition_label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1312, stratify=y)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            ("oversample", RandomOverSampler(random_state=42)),
            (
                "clf",
                LinearSVC(
                    random_state=42,
                    max_iter=5000,
                ),
            ),
        ]
    )

    print("\n Treinando modelo...")
    pipeline.fit(x_train, y_train)

    print("\n Avaliando no conjunto de teste (holdout)...")
    y_pred = pipeline.predict(x_test)
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    joblib.dump(pipeline, model_path)

    with open(ARTIFACTS_DIR / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"\n Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
