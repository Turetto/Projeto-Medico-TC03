from pydantic import BaseModel, Field


class LaudoRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=10,
        description="Texto do laudo/resumo médico a ser classificado",
    )


class ClassificacaoResponse(BaseModel):
    condicao: str = Field(..., description="Condição médica prevista")
    label_id: int = Field(..., description="ID numérico da classe (1-5)")
