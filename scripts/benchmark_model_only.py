import json
import statistics
import time
from pathlib import Path

import joblib

MODEL_PATH = Path("model/artifacts/baseline_model.joblib")
BASELINE_OUTPUT = Path("model/artifacts/latency_baseline.json")

SAMPLE_TEXTS = [
    "Patient presented with chest pain and elevated troponin levels consistent with myocardial infarction",
    "Biopsy revealed abnormal cell proliferation suggestive of malignant neoplasm in the colon",
    "MRI showed lesions in the white matter consistent with demyelinating nervous system disease",
    "Patient reports chronic abdominal pain with symptoms of inflammatory bowel disease",
    "General weakness and fatigue with nonspecific laboratory findings",
]

N_RUNS = 500


def main():
    print(f"Carregando modelo de {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    model_size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Tamanho do artefato: {model_size_kb:.1f} KB")

    print("Aquecendo (1a chamada aloca estruturas internas do sklearn)...")
    model.predict([SAMPLE_TEXTS[0]])

    print(f"Rodando {N_RUNS} inferências in-process...")
    latencies_ms = []
    for i in range(N_RUNS):
        texto = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        start = time.perf_counter()
        model.predict([texto])
        latencies_ms.append((time.perf_counter() - start) * 1000)

    q = statistics.quantiles(latencies_ms, n=100)
    resultado = {
        "model_size_kb": round(model_size_kb, 1),
        "n_runs": N_RUNS,
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 4),
            "p50": round(q[49], 4),
            "p95": round(q[94], 4),
            "p99": round(q[98], 4),
            "min": round(min(latencies_ms), 4),
            "max": round(max(latencies_ms), 4),
        },
    }

    print("\n--- Baseline (pré-otimização) ---")
    print(json.dumps(resultado, indent=2))

    with open(BASELINE_OUTPUT, "w") as f:
        json.dump(resultado, f, indent=2)
    print(f"\nSalvo em {BASELINE_OUTPUT} (referência para comparar na Subetapa 4.2).")


if __name__ == "__main__":
    main()
