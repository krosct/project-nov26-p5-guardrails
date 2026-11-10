"""Constantes, taxonomia e tipos centrais do pipeline de Guardrails (fonte única)."""

import json
from pathlib import Path
from typing import Literal, get_args

# Ordem de severidade usada nas métricas, matriz de confusão e desempate
Categoria = Literal[
    "seguro",
    "pii_exposto",
    "prompt_injection",
    "fraude_golpe",
    "ofensa_ameaca",
]
CLASSES: list[str] = list(get_args(Categoria))

Acao = Literal["permitir", "mascarar", "bloquear"]

# Mapeamento determinístico: cada categoria leva a uma ação de produto clara
ACAO_POR_CLASSE: dict[Categoria, Acao] = {
    "seguro": "permitir",
    "pii_exposto": "mascarar",
    "prompt_injection": "bloquear",
    "fraude_golpe": "bloquear",
    "ofensa_ameaca": "bloquear",
}

TipoPII = Literal["cpf", "cartao", "telefone", "email", "chave_pix"]

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

# Limiar de confiança da cascata: ESCOLHIDO por scripts/treinar.py, só na validação, pelo critério
# fixo abaixo, e gravado em models/limiar.json (fonte única, lida aqui, na API e na avaliação).
ORCAMENTO_ESCALONAMENTO = 0.15  # fração máxima de mensagens que podem ir ao LLM (custo/latência)
LIMIARES_CANDIDATOS = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0,30 a 0,95
CAMINHO_LIMIAR = RAIZ_PROJETO / "models/limiar.json"
LIMIAR_SEM_ARQUIVO = 0.50  # só vale antes do primeiro treino


def ler_limiar() -> float:
    if CAMINHO_LIMIAR.exists():
        return float(json.loads(CAMINHO_LIMIAR.read_text(encoding="utf-8"))["limiar"])
    return LIMIAR_SEM_ARQUIVO


LIMIAR_CONFIANCA_PADRAO = ler_limiar()

CAMINHO_TAXONOMIA = RAIZ_PROJETO / "data/taxonomia.md"
TAXONOMIA_TEXTO = (
    CAMINHO_TAXONOMIA.read_text(encoding="utf-8") if CAMINHO_TAXONOMIA.exists() else ""
)
