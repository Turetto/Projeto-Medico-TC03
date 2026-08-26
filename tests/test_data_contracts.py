from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_rejeita_texto_muito_curto():
    """Contrato: min_length=10 no schema deve barrar textos curtos antes de
    chegar no modelo."""
    response = client.post("/classificar", json={"texto": "abc"})
    assert response.status_code == 422


def test_rejeita_campo_texto_ausente():
    """Contrato: 'texto' é obrigatório no corpo da requisição."""
    response = client.post("/classificar", json={})
    assert response.status_code == 422


def test_rejeita_tipo_incorreto():
    """Contrato: 'texto' deve ser string, não número/lista/etc."""
    response = client.post("/classificar", json={"texto": 12345})
    assert response.status_code == 422
