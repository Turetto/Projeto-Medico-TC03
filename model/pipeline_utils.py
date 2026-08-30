import json
from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
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
PRODUCTION_MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"
CANDIDATE_MODEL_PATH = ARTIFACTS_DIR / "candidate_model.joblib"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"

MIN_MACRO_F1 = 0.45


def download_dataset() -> str:
    """
    Baixa (ou usa cache) o dataset do Kaggle e retorna o caminho local
    """
    return kagglehub.dataset_download("chaitanyakck/medical-text")


def load_train_data(dataset_path: str) -> pd.DataFrame:
    train_file = Path(dataset_path) / "train.dat"
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
    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    return Pipeline(
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
            ("clf", LinearSVC(random_state=42, max_iter=5000)),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    x, y = df["medical_abstract"], df["condition_label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test,
        y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        output_dict=True,
    )
    return pipeline, {"macro_f1": macro_f1, "report": report}


def save_candidate(pipeline: Pipeline) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, CANDIDATE_MODEL_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(LABEL_MAP, f, indent=2)


def promote_candidate_to_production() -> None:

    CANDIDATE_MODEL_PATH.replace(PRODUCTION_MODEL_PATH)
