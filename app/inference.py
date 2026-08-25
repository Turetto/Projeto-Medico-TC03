# app/inference.py
import json
from functools import lru_cache
from pathlib import Path

import joblib

ARTIFACTS_DIR = Path("model/artifacts")
MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"


@lru_cache(maxsize=1)
def get_model():
    """
    Carrega o modelo uma única vez e mantém em cache
    """
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_label_map() -> dict[int, str]:
    with open(LABEL_MAP_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def predict(texto: str) -> tuple[str, int]:
    """
    Retorna (nome_condicao, label_id) para um texto de entrada
    """
    model = get_model()
    label_map = get_label_map()

    label_id = int(model.predict([texto])[0])
    condicao = label_map[label_id]
    return condicao, label_id
