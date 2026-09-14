import json
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

ARTIFACTS_DIR = Path("model/artifacts")
BASELINE_JSON = ARTIFACTS_DIR / "latency_baseline.json"

SAMPLE_TEXTS = [
    "Patient presented with chest pain and elevated troponin levels consistent with myocardial infarction",
    "Biopsy revealed abnormal cell proliferation suggestive of malignant neoplasm in the colon",
    "MRI showed lesions in the white matter consistent with demyelinating nervous system disease",
    "Patient reports chronic abdominal pain with symptoms of inflammatory bowel disease",
    "General weakness and fatigue with nonspecific laboratory findings",
]
N_RUNS = 500


def benchmark_sklearn():
    model = joblib.load(ARTIFACTS_DIR / "baseline_model.joblib")
    model.predict([SAMPLE_TEXTS[0]])  # aquecimento
    latencies = []
    for i in range(N_RUNS):
        start = time.perf_counter()
        model.predict([SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]])
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_onnx(onnx_path: Path):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def run(texto):
        input_array = np.array([[texto]], dtype=object)
        return session.run(None, {input_name: input_array})

    run(SAMPLE_TEXTS[0])  # aquecimento
    latencies = []
    for i in range(N_RUNS):
        start = time.perf_counter()
        run(SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)])
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def summarize(name: str, latencies_ms: list[float], size_kb: float) -> dict:
    q = statistics.quantiles(latencies_ms, n=100)
    resumo = {
        "nome": name,
        "tamanho_kb": round(size_kb, 1),
        "p50_ms": round(q[49], 4),
        "p95_ms": round(q[94], 4),
        "p99_ms": round(q[98], 4),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return resumo


def main():
    resultados = []

    print("### sklearn (baseline original) ###")
    sklearn_size = (ARTIFACTS_DIR / "baseline_model.joblib").stat().st_size / 1024
    resultados.append(summarize("sklearn_fp32", benchmark_sklearn(), sklearn_size))

    print("\n### ONNX FP32 ###")
    onnx_fp32_size = (ARTIFACTS_DIR / "model_fp32.onnx").stat().st_size / 1024
    resultados.append(summarize("onnx_fp32", benchmark_onnx(ARTIFACTS_DIR / "model_fp32.onnx"), onnx_fp32_size))

    # Nota: quantize_dynamic (onnxruntime) só quantiza pesos guardados como
    # initializer em nós MatMul/Gemm/Conv. O skl2onnx converte LogisticRegression
    # no operador especializado ai.onnx.ml.LinearClassifier, que guarda os
    # coeficientes como atributo do próprio nó (não como initializer) - por isso
    # o INT8 abaixo sai praticamente do mesmo tamanho/latência que o FP32 aqui.
    # A comparação que importa nesse pipeline é sklearn (in-process) vs ONNX FP32.
    print("\n### ONNX INT8 (quantizado) ###")
    onnx_int8_size = (ARTIFACTS_DIR / "model_int8.onnx").stat().st_size / 1024
    resultados.append(summarize("onnx_int8", benchmark_onnx(ARTIFACTS_DIR / "model_int8.onnx"), onnx_int8_size))

    print("\n--- Comparativo final ---")
    baseline_p50 = resultados[0]["p50_ms"]
    for r in resultados:
        variacao = ((r["p50_ms"] - baseline_p50) / baseline_p50) * 100
        print(f"{r['nome']}: p50={r['p50_ms']}ms ({variacao:+.1f}% vs sklearn), tamanho={r['tamanho_kb']}KB")


if __name__ == "__main__":
    main()
