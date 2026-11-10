"""API FastAPI de moderação e guardrails em tempo real.

Executar: uv run uvicorn project_nov26_p5_guardrails.api:app
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import joblib
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from project_nov26_p5_guardrails.cascata import moderar
from project_nov26_p5_guardrails.dominio import (
    LIMIAR_CONFIANCA_PADRAO,
    RAIZ_PROJETO,
    Acao,
    Categoria,
    TipoPII,
)

app = FastAPI(
    title="Guardrails & Moderação em Tempo Real",
    description=(
        "Pipeline em cascata (Regras + Classificador CPU + LLM fallback) para "
        "proteção de assistentes, redação de PII e moderação de chat/avaliações."
    ),
    version="1.0.0",
)


class RequisicaoModeracao(BaseModel):
    texto: str = Field(
        min_length=1,
        max_length=4000,
        examples=["Meu pedido atrasou, me estorna no CPF 123.456.789-09 por favor."],
    )
    canal: Literal["chat_entrega", "avaliacao_restaurante", "assistente_ia"] = "chat_entrega"
    usar_llm_fallback: bool = False
    limiar_confianca: float = Field(default=LIMIAR_CONFIANCA_PADRAO, ge=0.0, le=1.0)


class RespostaModeracao(BaseModel):
    categoria: Categoria
    acao: Acao
    camada: str
    confianca: float
    texto_redigido: str
    pii_detectado: list[TipoPII]
    latencia_ms: float
    motivo: str


@lru_cache
def get_model():
    caminho = Path(os.environ.get("MODEL_PATH", str(RAIZ_PROJETO / "models/guardrail_ml.joblib")))
    if caminho.exists():
        return joblib.load(caminho)
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/moderate", response_model=RespostaModeracao)
def moderate(
    req: RequisicaoModeracao,
    model: Annotated[object, Depends(get_model)],
) -> RespostaModeracao:
    decisao = moderar(
        texto=req.texto,
        modelo_ml=model,
        limiar_confianca=req.limiar_confianca,
        usar_llm=req.usar_llm_fallback,
    )
    return RespostaModeracao(
        categoria=decisao.categoria,
        acao=decisao.acao,
        camada=decisao.camada,
        confianca=decisao.confianca,
        texto_redigido=decisao.texto_redigido,
        pii_detectado=decisao.pii_detectado,
        latencia_ms=decisao.latencia_ms,
        motivo=decisao.motivo,
    )
