from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """O endpoint de health deve responder 200 e status ok - usado por
    orquestradores/monitoramento para saber se o serviço está de pé."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classificar_retorna_schema_esperado():
    """Testa a forma da resposta, não um valor específico de classe -
    garante que o contrato de saída da API não quebrou."""
    response = client.post(
        "/classificar",
        json={"texto": "Patient presented with severe abdominal pain and diarrhea"},
    )
    assert response.status_code == 200

    body = response.json()
    assert "condicao" in body
    assert "label_id" in body
    assert isinstance(body["condicao"], str)
    assert body["label_id"] in range(1, 6)


def test_classificar_caso_conhecido_cardiovascular():
    """Teste de regressão: um texto claramente cardiovascular deve continuar
    sendo classificado como tal. Se esse teste falhar após uma mudança no
    modelo, é sinal de que o retreino degradou o comportamento esperado."""
    response = client.post(
        "/classificar",
        json={
            "texto": (
                "Patient presented with chest pain and elevated troponin "
                "levels consistent with myocardial infarction"
            )
        },
    )
    assert response.status_code == 200
    assert response.json()["label_id"] == 4
