import statistics
import time

import requests

API_URL = "http://127.0.0.1:8000/classificar"
N_REQUESTS = 100

SAMPLE_TEXTS = [
    "Patient presented with chest pain and elevated troponin levels consistent with myocardial infarction",
    "Biopsy revealed abnormal cell proliferation suggestive of malignant neoplasm in the colon",
    "MRI showed lesions in the white matter consistent with demyelinating nervous system disease",
    "Patient reports chronic abdominal pain with symptoms of inflammatory bowel disease",
    "General weakness and fatigue with nonspecific laboratory findings",
]


def warm_up():
    """
    Primeira chamada carrega o modelo em memória (lru_cache)
    """
    requests.post(API_URL, json={"texto": SAMPLE_TEXTS[0]}, timeout=10)


def run_benchmark():
    print("Aquecendo (primeira chamada carrega o modelo em memória)...")
    warm_up()

    latencies = []
    print(f"Enviando {N_REQUESTS} requisições para {API_URL}...")

    for i in range(N_REQUESTS):
        texto = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        start = time.perf_counter()
        response = requests.post(API_URL, json={"texto": texto}, timeout=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            print(f"  Requisição {i} falhou: {response.status_code}")
            continue
        latencies.append(elapsed_ms)

    if not latencies:
        print("Nenhuma requisição bem-sucedida.")
        return

    q = statistics.quantiles(latencies, n=N_REQUESTS)
    print("\n--- Resultado: baseline pré-otimização ---")
    print(f"Requisições bem-sucedidas: {len(latencies)}/{N_REQUESTS}")
    print(f"Latência média: {statistics.mean(latencies):.2f} ms")
    print(f"p50 (mediana):  {q[49]:.2f} ms")
    print(f"p95:            {q[94]:.2f} ms")
    print(f"p99:            {q[98]:.2f} ms")
    print(f"Mínimo:         {min(latencies):.2f} ms")
    print(f"Máximo:         {max(latencies):.2f} ms")


if __name__ == "__main__":
    run_benchmark()
