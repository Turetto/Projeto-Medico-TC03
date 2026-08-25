from fastapi import FastAPI, HTTPException

from app.inference import predict
from app.schemas import ClassificacaoResponse, LaudoRequest

app = FastAPI(
    title="API de Triagem de Laudos Médicos",
    description="Classifica laudos médicos em 5 categorias de condição",
    version="0.1.0",
)


@app.get("/health")
def health():
    """
    Endpoint de verificação de saúde do serviço
    """
    return {"status": "ok"}


@app.post("/classificar", response_model=ClassificacaoResponse)
def classificar(request: LaudoRequest):
    try:
        condicao, label_id = predict(request.texto)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {e}") from e

    return ClassificacaoResponse(condicao=condicao, label_id=label_id)
