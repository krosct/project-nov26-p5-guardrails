"""Treina o classificador L2, escolhe o limiar só na validação e mede L1/L2/L1+L2 na referência.

Uso: uv run --group train python scripts/treinar.py
- Treina L2 em data/treino.csv.
- Limiar: o MAIOR candidato cuja fração escalada ao LLM na validação cabe no orçamento
  (`ORCAMENTO_ESCALONAMENTO`). Critério e candidatos são fixos em dominio.py, definidos antes de
  olhar a referência; o resultado vai para models/limiar.json, lido pela cascata e pela avaliação.
- A referência (data/referencia.csv) só é usada depois, para medir. Sem chamar o LLM.
"""

import json
from pathlib import Path

import pandas as pd

from project_nov26_p5_guardrails.avaliar import relatorio
from project_nov26_p5_guardrails.cascata import moderar
from project_nov26_p5_guardrails.classificador import treinar_classificador
from project_nov26_p5_guardrails.dominio import (
    CAMINHO_LIMIAR,
    LIMIARES_CANDIDATOS,
    ORCAMENTO_ESCALONAMENTO,
)

RAIZ = Path(__file__).resolve().parents[1]


def escolher_limiar(modelo, textos_validacao: list[str]) -> dict:
    """Maior limiar cujo escalonamento na validação ≤ orçamento. Só usa a validação."""
    decisoes = [moderar(t, modelo_ml=modelo, usar_llm=False) for t in textos_validacao]
    escalonamento = {
        lim: sum(d.camada == "L2_ml" and d.confianca < lim for d in decisoes) / len(decisoes)
        for lim in LIMIARES_CANDIDATOS
    }
    cabem = [lim for lim, frac in escalonamento.items() if frac <= ORCAMENTO_ESCALONAMENTO]
    limiar = max(cabem) if cabem else min(LIMIARES_CANDIDATOS)
    return {
        "limiar": limiar,
        "criterio": f"maior limiar com escalonamento na validação <= {ORCAMENTO_ESCALONAMENTO}",
        "escalonamento_validacao": round(escalonamento[limiar], 3),
        "escalonamento_por_candidato": {str(k): round(v, 3) for k, v in escalonamento.items()},
    }


def main() -> None:
    treino = pd.read_csv(RAIZ / "data/treino.csv")
    val = pd.read_csv(RAIZ / "data/validacao.csv")
    ref = pd.read_csv(RAIZ / "data/referencia.csv")

    modelo = treinar_classificador(
        treino["texto"].tolist(),
        treino["rotulo"].tolist(),
        destino=RAIZ / "models/guardrail_ml.joblib",
    )
    escolha = escolher_limiar(modelo, val["texto"].tolist())
    CAMINHO_LIMIAR.write_text(json.dumps(escolha, indent=2, ensure_ascii=False))

    verdade, textos, fontes = ref["rotulo"].tolist(), ref["texto"].tolist(), ref["fonte"].tolist()
    limiar = escolha["limiar"]
    d_l1 = [moderar(t, modelo_ml=None, usar_llm=False) for t in textos]
    d_cpu = [moderar(t, modelo_ml=modelo, limiar_confianca=limiar, usar_llm=False) for t in textos]
    prev_l2 = modelo.predict(textos).tolist()

    relatorios = {
        "regras_l1": relatorio(verdade, [d.categoria for d in d_l1], fontes),
        "classificador_l2": relatorio(verdade, prev_l2, fontes),
        "cascata_cpu": {
            "limiar_validacao": limiar,
            **relatorio(verdade, [d.categoria for d in d_cpu], fontes, [d.camada for d in d_cpu]),
        },
    }
    for nome, r in relatorios.items():
        (RAIZ / f"reports/{nome}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))

    esc = escolha["escalonamento_validacao"]
    print(f"Limiar escolhido na validação: {limiar} (escalonamento {esc})")
    for nome, r in relatorios.items():
        print(f"{nome:18} F1 macro {r['f1_macro']}  IC95 {r['ic95']['f1_macro']}")


if __name__ == "__main__":
    main()
