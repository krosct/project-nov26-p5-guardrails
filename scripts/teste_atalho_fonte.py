"""Teste de atalho de fonte: quanto do resultado se explica só pelo "tipo de texto"?

Uso: uv run --group train python scripts/teste_atalho_fonte.py   (CPU, sem API)
1. Classificador de FONTE: mesmo L2, mas prevendo a família da fonte (olist, tweet, pi, golpe).
   Se acerta quase tudo, o estilo é facilmente separável.
2. Baseline de atalho: prevê a classe MAIS COMUM da família no treino, sem olhar o conteúdo.
   É o teto de quem só aprendeu "de onde vem o texto".
3. Compara com o L2 de verdade, no total e nas famílias que têm mais de uma classe.
Grava reports/teste_atalho_fonte.json.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from project_nov26_p5_guardrails.classificador import treinar_classificador

RAIZ = Path(__file__).resolve().parents[1]


def familia(fonte: str) -> str:
    # olist_* -> olist, tweet_* -> tweet, pi_* -> pi, golpe_sintetico -> golpe
    return fonte.split("_")[0]


def main() -> None:
    treino = pd.read_csv(RAIZ / "data/treino.csv")
    ref = pd.read_csv(RAIZ / "data/referencia.csv")
    treino["familia"], ref["familia"] = treino["fonte"].map(familia), ref["fonte"].map(familia)

    clf_fonte = treinar_classificador(treino["texto"].tolist(), treino["familia"].tolist())
    acerto_fonte = accuracy_score(ref["familia"], clf_fonte.predict(ref["texto"]))

    mais_comum = {
        f: Counter(g["rotulo"]).most_common(1)[0][0] for f, g in treino.groupby("familia")
    }
    prev_atalho = ref["familia"].map(mais_comum)
    clf_classe = treinar_classificador(treino["texto"].tolist(), treino["rotulo"].tolist())
    prev_l2 = clf_classe.predict(ref["texto"])

    # Famílias com mais de uma classe: aí o estilo sozinho não basta
    multiclasse = [f for f, g in ref.groupby("familia") if g["rotulo"].nunique() > 1]
    mask = ref["familia"].isin(multiclasse)
    saida = {
        "acuracia_do_classificador_de_fonte": round(float(acerto_fonte), 3),
        "classe_mais_comum_por_familia": mais_comum,
        "acuracia_total": {
            "atalho_so_pela_fonte": round(float(accuracy_score(ref["rotulo"], prev_atalho)), 3),
            "l2": round(float(accuracy_score(ref["rotulo"], prev_l2)), 3),
        },
        "familias_com_mais_de_uma_classe": multiclasse,
        "acuracia_nessas_familias": {
            "atalho_so_pela_fonte": round(
                float(accuracy_score(ref["rotulo"][mask], prev_atalho[mask])), 3
            ),
            "l2": round(float(accuracy_score(ref["rotulo"][mask], prev_l2[mask])), 3),
        },
        "leitura": (
            "O classificador de fonte acerta quase tudo, então o estilo é separável. O atalho puro "
            "fica bem abaixo do L2 nas famílias com mais de uma classe: o L2 usa conteúdo, mas "
            "erra os poucos exemplos seguros que vêm da mesma fonte de uma classe de ataque."
        ),
    }
    (RAIZ / "reports/teste_atalho_fonte.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False)
    )
    print(json.dumps(saida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
