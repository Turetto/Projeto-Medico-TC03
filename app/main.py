import os
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.inference import predict, reload_model
from app.metrics import INFERENCE_LATENCY, PREDICTIONS_BY_CLASS, REQUEST_COUNT
from app.schemas import ClassificacaoResponse, LaudoRequest

app = FastAPI(
    title="API de Triagem de Laudos Médicos",
    description=(
        "Classifica laudos médicos em 5 categorias de condição. "
        "Limitação conhecida: o modelo foi treinado exclusivamente com textos em "
        "inglês (Medical Abstracts TC Corpus); textos em outros idiomas não são "
        "reconhecidos pelo vocabulário do TF-IDF e produzem classificações não confiáveis."
    ),
    version="0.1.0",
)

ADMIN_RELOAD_TOKEN = os.environ.get("ADMIN_RELOAD_TOKEN", "")


@app.get("/health")
def health():
    """
    Endpoint de verificação de saúde do serviço
    """
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """
    Endpoint para metricas de avaliação do prometheus
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/classificar", response_model=ClassificacaoResponse)
def classificar(request: LaudoRequest):
    start = time.perf_counter()
    try:
        condicao, label_id = predict(request.texto)
    except Exception as e:
        REQUEST_COUNT.labels(status="erro").inc()
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {e}") from e

    elapsed = time.perf_counter() - start
    INFERENCE_LATENCY.observe(elapsed)
    REQUEST_COUNT.labels(status="sucesso").inc()
    PREDICTIONS_BY_CLASS.labels(condicao=condicao).inc()

    return ClassificacaoResponse(condicao=condicao, label_id=label_id)


@app.post("/admin/reload-model")
def reload_model_endpoint(x_admin_token: str = Header(default="")):
    """
    Reiniciar o modelo do disco sem reiniciar o container.
    chamar manualmente depois de promovido no airflow
    """
    if not ADMIN_RELOAD_TOKEN or x_admin_token != ADMIN_RELOAD_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido.")
    reload_model()
    return {"status": "modelo recarregado"}
