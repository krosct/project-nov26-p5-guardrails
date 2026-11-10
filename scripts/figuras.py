"""Gera as figuras dos relatórios em alta resolução e com formatação em PT-BR.

Uso: uv run --group train python scripts/figuras.py
- reports/matriz_confusao.png: matriz de confusão na referência (250 exemplos).
- reports/cascata_tradeoff.png: comparação entre abordagens e distribuição por camada da cascata.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from project_nov26_p5_guardrails.dominio import CLASSES, LIMIAR_CONFIANCA_PADRAO

RAIZ = Path(__file__).resolve().parents[1]

AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#2ca02c"
TEXTO, TEXTO_2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def virgula(texto: str) -> str:
    return texto.replace(".", ",")


def _estilo(ax) -> None:
    ax.set_facecolor(FUNDO)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.xaxis.label.set_color(TEXTO_2)
    ax.yaxis.label.set_color(TEXTO_2)


def gerar_matriz_confusao() -> None:
    dados = json.loads((RAIZ / "reports/cascata_completa.json").read_text(encoding="utf-8"))
    m = np.array(dados["matriz_confusao"])
    fig, ax = plt.subplots(figsize=(7.2, 5.8), facecolor=FUNDO)
    ax.imshow(m, cmap="Blues")
    rotulos = [c.replace("_", " ") for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)), rotulos, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), rotulos)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if m[i, j]:
                cor = "white" if m[i, j] > m.max() / 2 else "black"
                ax.text(j, i, int(m[i, j]), ha="center", va="center", color=cor, fontsize=10)
    f1_str = virgula(f"{dados['f1_macro']:.3f}")
    ax.set(
        xlabel="Previsto (Guardrail em Cascata)",
        ylabel="Referência",
        title=f"Matriz de confusão — {int(m.sum())} mensagens de referência (F1 = {f1_str})",
    )
    fig.tight_layout()
    fig.savefig(RAIZ / "reports/matriz_confusao.png", dpi=120, facecolor=FUNDO)
    plt.close(fig)


def gerar_grafico_tradeoff() -> None:
    l1 = json.loads((RAIZ / "reports/regras_l1.json").read_text(encoding="utf-8"))
    l2 = json.loads((RAIZ / "reports/classificador_l2.json").read_text(encoding="utf-8"))
    casc = json.loads((RAIZ / "reports/cascata_completa.json").read_text(encoding="utf-8"))
    lat = json.loads((RAIZ / "reports/latencia_custo.json").read_text(encoding="utf-8"))
    llm = json.loads((RAIZ / "reports/llm_100pct.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=FUNDO)

    # Painel 1: F1 macro vs Recall de bloqueio de ataques nas 3 abordagens
    nomes = [
        "L1 Regras\n(Regex + Checksum)",
        "L2 Classificador\n(TF-IDF + LR CPU)",
        "Cascata L1+L2+L3\n(defesa em profundidade)",
        "LLM juiz em 100%\n(medido)",
    ]
    f1s = [l1["f1_macro"], l2["f1_macro"], casc["f1_macro"], llm["f1_macro"]]
    recalls = [
        l1["recall_bloqueio_ataques"],
        l2["recall_bloqueio_ataques"],
        casc["recall_bloqueio_ataques"],
        llm["recall_bloqueio_ataques"],
    ]

    x = np.arange(len(nomes))
    largura = 0.34
    b1 = ax1.bar(x - largura / 2, f1s, largura, color=AZUL, label="F1 macro (5 classes)")
    b2 = ax1.bar(
        x + largura / 2, recalls, largura, color=LARANJA, label="Recall de bloqueio de ataques"
    )

    for barra in (*b1, *b2):
        h = barra.get_height()
        ax1.annotate(
            virgula(f"{h:.3f}"),
            (barra.get_x() + barra.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=TEXTO,
        )

    ax1.set_xticks(x, nomes)
    ax1.set_ylim(0, 1.22)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: virgula(f"{v:.2f}")))
    ax1.set_ylabel("Métrica na referência (250 mensagens)")
    ax1.set_title(
        "F1 macro e recall de bloqueio por abordagem",
        loc="left",
        fontsize=10.5,
        color=TEXTO,
        pad=14,
    )
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    _estilo(ax1)

    # Painel 2: Curva de limiar de confiança (% chamadas ao LLM vs F1 macro)
    curva = lat["curva_limiar"]
    limiares = [c["limiar"] for c in curva]
    pct_cpu = [round(100 - c["pct_escalado_llm"], 1) for c in curva]
    pct_llm = [c["pct_escalado_llm"] for c in curva]

    ax2.bar(
        [virgula(f"{lim:.2f}") for lim in limiares],
        pct_cpu,
        color=AZUL,
        label="Resolvido em CPU: L1 + L2",
    )
    ax2.bar(
        [virgula(f"{lim:.2f}") for lim in limiares],
        pct_llm,
        bottom=pct_cpu,
        color=LARANJA,
        label="Escalado ao LLM Juiz (L3)",
    )
    ponto_padrao = next(c for c in curva if abs(c["limiar"] - LIMIAR_CONFIANCA_PADRAO) < 1e-4)
    pct_cpu_padrao = virgula(f"{100 - ponto_padrao['pct_escalado_llm']:.1f}")
    lim_str = virgula(f"{LIMIAR_CONFIANCA_PADRAO:.2f}")
    ax2.set_ylim(0, 118)
    ax2.set_xlabel("Limiar de confiança da camada L2")
    ax2.tick_params(axis="x", labelrotation=60)
    ax2.set_ylabel("% de mensagens na referência")
    ax2.set_title(
        f"{pct_cpu_padrao}% decidido em CPU (limiar {lim_str}, escolhido na validação)",
        loc="left",
        fontsize=10.5,
        color=TEXTO,
        pad=14,
    )
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    _estilo(ax2)

    fig.tight_layout()
    fig.savefig(RAIZ / "reports/cascata_tradeoff.png", dpi=120, facecolor=FUNDO)
    plt.close(fig)


def main() -> None:
    gerar_matriz_confusao()
    gerar_grafico_tradeoff()
    print("Figuras geradas em reports/matriz_confusao.png e reports/cascata_tradeoff.png")


if __name__ == "__main__":
    main()
