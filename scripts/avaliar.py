"""Avaliação final na referência: cascata completa, LLM em 100% do tráfego, curva de limiar e custo.

Uso: uv run --group train python scripts/avaliar.py   (usa a chave do OpenRouter, modelo :free)
- O limiar vem de models/limiar.json (escolhido só na validação por scripts/treinar.py).
- O juiz LLM roda nas 250 mensagens da referência (com cache em disco): isso permite MEDIR a
  linha "LLM em 100% do tráfego" e traçar a curva de limiar sem chamadas extras.
- Tokens, latência e custo do juiz saem das chamadas reais (nada é suposto, exceto os preços).
Grava reports/cascata_completa.json, reports/llm_100pct.json e reports/latencia_custo.json.
"""

import json
import os
import platform
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from project_nov26_p5_guardrails.avaliar import relatorio
from project_nov26_p5_guardrails.cascata import moderar
from project_nov26_p5_guardrails.dominio import (
    LIMIARES_CANDIDATOS,
    ler_limiar,
)
from project_nov26_p5_guardrails.llm import julgar_com_llm
from project_nov26_p5_guardrails.regras import inspecionar_regras

RAIZ = Path(__file__).resolve().parents[1]

PRECOS = {
    "data_consulta": "2026-11-10",
    "llm_entrada_usd_por_1m_tokens": 0.60,  # versão paga do mesmo modelo
    "llm_saida_usd_por_1m_tokens": 2.40,
    "cpu_usd_por_hora": 0.10,
}


def custo_llm_por_mensagem(tokens_entrada: float, tokens_saida: float) -> float:
    return (
        tokens_entrada * PRECOS["llm_entrada_usd_por_1m_tokens"]
        + tokens_saida * PRECOS["llm_saida_usd_por_1m_tokens"]
    ) / 1e6


def main() -> None:
    ref = pd.read_csv(RAIZ / "data/referencia.csv")
    modelo = joblib.load(RAIZ / "models/guardrail_ml.joblib")
    limiar = ler_limiar()
    verdade, textos, fontes = ref["rotulo"].tolist(), ref["texto"].tolist(), ref["fonte"].tolist()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY não definida (crie o .env.shared, ver README).")

    # 1. Juiz LLM em TODAS as mensagens (texto já redigido pela L1, como na cascata)
    redigidos = [inspecionar_regras(t).texto_redigido for t in textos]
    print(f"Juiz LLM em {len(redigidos)} mensagens (cache em disco quando já feito)...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        juizes = list(ex.map(julgar_com_llm, redigidos))
    cat_llm = [j[0] for j in juizes]
    tokens_in = float(np.mean([j[2]["tokens_entrada"] for j in juizes]))
    tokens_out = float(np.mean([j[2]["tokens_saida"] for j in juizes]))
    lat_llm_s = [j[2]["latencia_s"] for j in juizes]
    falhas_parse = sum(j[1] == "fallback_erro_parse_llm" for j in juizes)

    llm_puro = {
        **relatorio(verdade, cat_llm, fontes),
        "medido_em": f"{len(textos)} mensagens da referência, texto redigido pela L1",
        "falhas_de_parse_do_juiz": falhas_parse,
        "latencia_ms_p50": round(float(np.percentile(lat_llm_s, 50)) * 1000),
        "latencia_ms_p95": round(float(np.percentile(lat_llm_s, 95)) * 1000),
    }
    (RAIZ / "reports/llm_100pct.json").write_text(
        json.dumps(llm_puro, indent=2, ensure_ascii=False)
    )

    # 2. Cascata completa no limiar escolhido na validação
    decisoes = [
        moderar(t, modelo_ml=modelo, limiar_confianca=limiar, usar_llm=True) for t in textos
    ]
    camadas = Counter(d.camada for d in decisoes)
    fracao_l3 = camadas.get("L3_llm", 0) / len(textos)
    completa = {
        "limiar_validacao": limiar,
        **relatorio(verdade, [d.categoria for d in decisoes], fontes, [d.camada for d in decisoes]),
        "distribuicao_camadas": dict(camadas),
    }
    (RAIZ / "reports/cascata_completa.json").write_text(
        json.dumps(completa, indent=2, ensure_ascii=False)
    )

    # 3. Curva de limiar (ANÁLISE na referência; o limiar não é escolhido por ela)
    curva = []
    for lim in LIMIARES_CANDIDATOS:
        decs = [moderar(t, modelo_ml=modelo, limiar_confianca=lim, usar_llm=True) for t in textos]
        c = Counter(d.camada for d in decs)
        pct_llm = c.get("L3_llm", 0) / len(textos)
        m = relatorio(verdade, [d.categoria for d in decs], fontes)
        curva.append(
            {
                "limiar": lim,
                "f1_macro": m["f1_macro"],
                "f1_macro_ic95": m["ic95"]["f1_macro"],
                "taxa_falso_bloqueio": m["taxa_falso_bloqueio"],
                "pct_escalado_llm": round(pct_llm * 100, 1),
                "custo_usd_1m": round(
                    (0.05 + custo_llm_por_mensagem(tokens_in, tokens_out) * 1e6 * pct_llm), 2
                ),
            }
        )

    # 4. Vazão e latência na CPU
    inicio = time.perf_counter()
    d_cpu = [moderar(t, modelo_ml=modelo, limiar_confianca=limiar, usar_llm=False) for t in textos]
    msg_por_s = round(len(textos) / max(time.perf_counter() - inicio, 1e-6))
    d_l1 = [moderar(t, modelo_ml=None, usar_llm=False) for t in textos]
    cpu_nome = next(
        (
            linha.split(":", 1)[1].strip()
            for linha in Path("/proc/cpuinfo").read_text().splitlines()
            if linha.startswith("model name")
        ),
        platform.machine(),
    )
    p = lambda ds, q: round(float(np.percentile([d.latencia_ms for d in ds], q)), 3)  # noqa: E731
    por_msg = custo_llm_por_mensagem(tokens_in, tokens_out)
    saida = {
        "hardware": cpu_nome,
        "premissas": PRECOS,
        "tokens_medios_juiz_medidos": {
            "entrada": round(tokens_in, 1),
            "saida": round(tokens_out, 1),
        },
        "mensagens_por_segundo_cpu": msg_por_s,
        "limiar_validacao": limiar,
        "latencia_ms": {
            "l1_regras_p50": p(d_l1, 50),
            "l1_regras_p95": p(d_l1, 95),
            "cascata_cpu_p50": p(d_cpu, 50),
            "cascata_cpu_p95": p(d_cpu, 95),
            "cascata_completa_p50": p(decisoes, 50),
            "cascata_completa_p95": p(decisoes, 95),
            "llm_100pct_p50": llm_puro["latencia_ms_p50"],
            "llm_100pct_p95": llm_puro["latencia_ms_p95"],
        },
        "custo_usd_por_1m_mensagens": {
            "l1_regras": 0.01,
            "cascata_cpu": 0.05,
            "cascata_completa": round(0.05 + por_msg * 1e6 * fracao_l3, 2),
            "llm_puro_100pct": round(por_msg * 1e6, 2),
        },
        "curva_limiar": curva,
    }
    (RAIZ / "reports/latencia_custo.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False)
    )
    print(
        f"cascata completa: F1 macro {completa['f1_macro']} IC95 {completa['ic95']['f1_macro']} "
        f"| LLM 100%: {llm_puro['f1_macro']} IC95 {llm_puro['ic95']['f1_macro']} "
        f"| escalado ao LLM: {fracao_l3:.1%}"
    )


if __name__ == "__main__":
    main()
