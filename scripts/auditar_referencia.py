"""Audita os rótulos da referência com o LLM professor (independente do classificador L2).

Uso: uv run python scripts/auditar_referencia.py   (usa a chave do OpenRouter, modelo :free)
Os rótulos da referência vêm da construção dos dados (fonte + regra), não de um anotador humano.
Esta auditoria mede a concordância de um juiz independente e lista os DESACORDOS em
data/referencia_revisar.csv para revisão humana. Grava reports/auditoria_referencia.json.
O texto vai ao LLM SEM redigir: o PII da referência é sintético (nenhum dado real) e a
auditoria precisa ver o dado para julgar o rótulo `pii_exposto`.
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from project_nov26_p5_guardrails.dominio import CLASSES
from project_nov26_p5_guardrails.llm import rotular_em_lotes

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY não definida (crie o .env.shared, ver README).")
    ref = pd.read_csv(RAIZ / "data/referencia.csv")
    ref["rotulo_llm"] = rotular_em_lotes(ref["texto"].tolist(), tamanho_lote=25, paralelo=3)
    validos = ref.dropna(subset=["rotulo_llm"])
    concorda = validos["rotulo"] == validos["rotulo_llm"]

    resumo = {
        "n": len(ref),
        "sem_rotulo_do_llm": int(ref["rotulo_llm"].isna().sum()),
        "concordancia": round(float(concorda.mean()), 3),
        "kappa": round(
            float(cohen_kappa_score(validos["rotulo"], validos["rotulo_llm"], labels=CLASSES)), 3
        ),
        "concordancia_por_classe": {
            c: round(float(concorda[validos["rotulo"] == c].mean()), 3) for c in CLASSES
        },
        "concordancia_por_fonte": {
            f: round(float(g.mean()), 3) for f, g in concorda.groupby(validos["fonte"])
        },
        "desacordos_para_revisao_humana": int((~concorda).sum()),
    }
    (RAIZ / "reports/auditoria_referencia.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False)
    )
    validos[~concorda][["id", "fonte", "texto", "rotulo", "rotulo_llm"]].to_csv(
        RAIZ / "data/referencia_revisar.csv", index=False
    )
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
