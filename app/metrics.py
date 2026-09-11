from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total de requisições recebidas pelo endpoint de classificação",
    labelnames=["status"],
)

PREDICTIONS_BY_CLASS = Counter(
    "api_predictions_by_class",
    "Total de projeções por condição médica",
    labelnames=["condicao"],
)

INFERENCE_LATENCY = Histogram(
    "api_inference_latency_seconds",
    "Tempo de inferencia do modelo",
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
