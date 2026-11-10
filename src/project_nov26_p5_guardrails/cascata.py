"""Orquestrador em cascata (L1 Regras -> L2 Classificador CPU -> L3 LLM Juiz).

Garante baixa latência (< 1 ms na maioria das mensagens) e custo mínimo de API,
escalando para o LLM apenas os casos em que o classificador L2 tem confiança abaixo do limiar.
"""

import time
from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from project_nov26_p5_guardrails.classificador import prever_com_confianca
from project_nov26_p5_guardrails.dominio import (
    ACAO_POR_CLASSE,
    LIMIAR_CONFIANCA_PADRAO,
    Acao,
    Categoria,
    TipoPII,
)
from project_nov26_p5_guardrails.llm import MODELO_PADRAO, julgar_com_llm
from project_nov26_p5_guardrails.regras import inspecionar_regras


@dataclass(frozen=True)
class DecisaoGuardrail:
    categoria: Categoria
    acao: Acao
    camada: str  # "L1_regras" | "L2_ml" | "L3_llm"
    confianca: float
    texto_redigido: str
    pii_detectado: list[TipoPII]
    latencia_ms: float
    motivo: str


def moderar(
    texto: str,
    modelo_ml: Pipeline | None = None,
    limiar_confianca: float = LIMIAR_CONFIANCA_PADRAO,
    usar_llm: bool = True,
    modelo_llm: str = MODELO_PADRAO,
) -> DecisaoGuardrail:
    """Executa a cascata completa de proteção e moderação sobre o texto."""
    inicio = time.perf_counter()

    # Camada L1: Regras determinísticas e redação de PII em código
    r_l1 = inspecionar_regras(texto)
    if r_l1.categoria_bloqueio is not None:
        cat = r_l1.categoria_bloqueio
        return DecisaoGuardrail(
            categoria=cat,
            acao=ACAO_POR_CLASSE[cat],
            camada="L1_regras",
            confianca=1.0,
            texto_redigido=r_l1.texto_redigido,
            pii_detectado=r_l1.tipos_pii,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 3),
            motivo=r_l1.regra_acionada or "regra_deterministica",
        )

    # Se não houver modelo L2 carregado, decide apenas com L1 (modo somente regras)
    if modelo_ml is None:
        cat_l1: Categoria = "pii_exposto" if r_l1.tipos_pii else "seguro"
        return DecisaoGuardrail(
            categoria=cat_l1,
            acao=ACAO_POR_CLASSE[cat_l1],
            camada="L1_regras",
            confianca=1.0 if r_l1.tipos_pii else 0.5,
            texto_redigido=r_l1.texto_redigido,
            pii_detectado=r_l1.tipos_pii,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 3),
            motivo="pii_redigido_em_codigo" if r_l1.tipos_pii else "sem_bloqueio_l1",
        )

    # Camada L2: Classificador estatístico rápido na CPU
    cat_l2, conf_l2 = prever_com_confianca(modelo_ml, texto)

    # 1. Se L1 detectou PII matemático válido e L2 acha que é "seguro" ou "pii_exposto"
    # (não é ataque de prompt_injection, fraude_golpe ou ofensa_ameaca), decide "pii_exposto" em L1
    if r_l1.tipos_pii and cat_l2 in ("seguro", "pii_exposto"):
        return DecisaoGuardrail(
            categoria="pii_exposto",
            acao="mascarar",
            camada="L1_regras",
            confianca=1.0,
            texto_redigido=r_l1.texto_redigido,
            pii_detectado=r_l1.tipos_pii,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 3),
            motivo="pii_detectado_por_checksum_ou_formato",
        )

    # 2. Se L2 previu "pii_exposto", mas o verificador L1 comprovou que não há PII,
    # reavalia entre as demais classes (evita falsos positivos de números de pedido)
    if cat_l2 == "pii_exposto" and not r_l1.tipos_pii:
        probs = modelo_ml.predict_proba([texto])[0]
        opcoes = [
            (c, float(p))
            for c, p in zip(modelo_ml.classes_, probs, strict=True)
            if c != "pii_exposto"
        ]
        cat_l2, conf_l2 = max(opcoes, key=lambda x: x[1])
        conf_l2 = round(conf_l2, 4)

    # Se a confiança de L2 atinge o limiar (ou se L3 estiver desativado), encerra aqui
    if conf_l2 >= limiar_confianca or not usar_llm:
        return DecisaoGuardrail(
            categoria=cat_l2,
            acao=ACAO_POR_CLASSE[cat_l2],
            camada="L2_ml",
            confianca=conf_l2,
            texto_redigido=r_l1.texto_redigido,
            pii_detectado=r_l1.tipos_pii,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 3),
            motivo=f"classificador_cpu (conf={conf_l2:.2f})",
        )

    # Camada L3: LLM Juiz para exemplos ambíguos / fronteira semântica
    # Envia o texto já redigido por L1 para não vazar PII de clientes para provedores externos!
    tempo_cpu_ms = (time.perf_counter() - inicio) * 1000
    cat_l3, motivo_l3, meta_l3 = julgar_com_llm(r_l1.texto_redigido, modelo=modelo_llm)
    if r_l1.tipos_pii and cat_l3 == "seguro":
        cat_l3 = "pii_exposto"
    # Soma a latência real da chamada de rede do LLM (mesmo quando lida do cache em disco)
    lat_llm_ms = float(meta_l3.get("latencia_s", 0.0) or 0.0) * 1000
    lat_total_ms = (
        round(tempo_cpu_ms + lat_llm_ms, 3)
        if lat_llm_ms > 0
        else round((time.perf_counter() - inicio) * 1000, 3)
    )
    return DecisaoGuardrail(
        categoria=cat_l3,
        acao=ACAO_POR_CLASSE[cat_l3],
        camada="L3_llm",
        confianca=0.95,
        texto_redigido=r_l1.texto_redigido,
        pii_detectado=r_l1.tipos_pii,
        latencia_ms=lat_total_ms,
        motivo=motivo_l3,
    )
