"""Métricas de avaliação multi-classe e de segurança operacional contra a referência."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from project_nov26_p5_guardrails.dominio import ACAO_POR_CLASSE, CLASSES


def metricas(verdade: list[str], previsto: list[str]) -> dict:
    """Calcula F1 macro, acurácia, Kappa de Cohen, F1 por classe e falsos bloqueios."""
    p, r, f1, _ = precision_recall_fscore_support(
        verdade, previsto, labels=CLASSES, zero_division=0
    )
    por_classe = {
        classe: {
            "precisao": round(float(p[i]), 3),
            "recall": round(float(r[i]), 3),
            "f1": round(float(f1[i]), 3),
        }
        for i, classe in enumerate(CLASSES)
    }

    # Taxa de bloqueio indevido: mensagens 'seguro' ou 'pii_exposto' que foram bloqueadas
    nao_bloqueaveis = [i for i, v in enumerate(verdade) if ACAO_POR_CLASSE[v] != "bloquear"]
    falsos_bloqueios = sum(1 for i in nao_bloqueaveis if ACAO_POR_CLASSE[previsto[i]] == "bloquear")
    # Recall de ataques: fração de (prompt_injection, fraude_golpe, ofensa_ameaca) bloqueada
    ataques = [i for i, v in enumerate(verdade) if ACAO_POR_CLASSE[v] == "bloquear"]
    ataques_bloqueados = sum(1 for i in ataques if ACAO_POR_CLASSE[previsto[i]] == "bloquear")

    return {
        "f1_macro": round(
            float(f1_score(verdade, previsto, labels=CLASSES, average="macro", zero_division=0)), 3
        ),
        "acuracia": round(float(accuracy_score(verdade, previsto)), 3),
        "kappa": round(float(cohen_kappa_score(verdade, previsto, labels=CLASSES)), 3),
        "recall_bloqueio_ataques": round(ataques_bloqueados / max(len(ataques), 1), 3),
        "taxa_falso_bloqueio": round(falsos_bloqueios / max(len(nao_bloqueaveis), 1), 3),
        "por_classe": por_classe,
    }


def matriz(verdade: list[str], previsto: list[str]) -> list[list[int]]:
    """Matriz de confusão (linhas = referência, colunas = previsto) na ordem de CLASSES."""
    return confusion_matrix(verdade, previsto, labels=CLASSES).tolist()


REAMOSTRAGENS = 2000
SEED = 42
# Cenário ILUSTRATIVO de tráfego (suposição, não medida): quase tudo é seguro. A referência é
# balanceada (20% por classe), então os números "por tráfego" mudam bastante com a prevalência.
PREVALENCIA_REALISTA: dict[str, float] = {
    "seguro": 0.97,
    "pii_exposto": 0.01,
    "prompt_injection": 0.005,
    "fraude_golpe": 0.005,
    "ofensa_ameaca": 0.01,
}


def ic95_bootstrap(verdade: list[str], previsto: list[str]) -> dict:
    """IC95 por bootstrap dos pares (verdade, previsto): F1 macro, falso bloqueio e recall."""
    rng = np.random.default_rng(SEED)
    v, p = np.asarray(verdade), np.asarray(previsto)
    chaves = ("f1_macro", "taxa_falso_bloqueio", "recall_bloqueio_ataques")
    valores: dict[str, list[float]] = {k: [] for k in chaves}
    for _ in range(REAMOSTRAGENS):
        i = rng.integers(0, len(v), len(v))
        m = metricas(v[i].tolist(), p[i].tolist())
        for k in chaves:
            valores[k].append(m[k])
    return {k: np.percentile(x, [2.5, 97.5]).round(3).tolist() for k, x in valores.items()}


def por_fonte(verdade: list[str], previsto: list[str], fontes: list[str]) -> dict:
    """Acurácia por fonte de dados. Fontes que aparecem em duas classes (tweets, dataset de
    injeção) mostram se o modelo separa o CONTEÚDO e não só o estilo da fonte."""
    acertos: dict[str, list[bool]] = defaultdict(list)
    for v, p, f in zip(verdade, previsto, fontes, strict=True):
        acertos[f].append(v == p)
    return {
        f: {"n": len(a), "acuracia": round(sum(a) / len(a), 3)} for f, a in sorted(acertos.items())
    }


def com_prevalencia(
    verdade: list[str],
    previsto: list[str],
    camadas: list[str] | None = None,
    prevalencia: dict[str, float] = PREVALENCIA_REALISTA,
) -> dict:
    """Reponderam a referência (balanceada) para uma prevalência de tráfego dada.

    Cada exemplo pesa prevalência[classe] / n_classe. Devolve, para a ação `bloquear`, a
    precisão esperada (fração dos bloqueios que eram ataques) e, se `camadas` for dado, a fração
    do tráfego resolvida em cada camada.
    """
    n_por_classe: dict[str, int] = defaultdict(int)
    for v in verdade:
        n_por_classe[v] += 1
    pesos = np.array([prevalencia[v] / n_por_classe[v] for v in verdade])
    bloqueou = np.array([ACAO_POR_CLASSE[p] == "bloquear" for p in previsto])
    era_ataque = np.array([ACAO_POR_CLASSE[v] == "bloquear" for v in verdade])
    saida = {
        "prevalencia_suposta": prevalencia,
        "precisao_do_bloqueio": round(
            float(pesos[bloqueou & era_ataque].sum() / max(pesos[bloqueou].sum(), 1e-12)), 3
        ),
        "taxa_falso_bloqueio": round(
            float(pesos[bloqueou & ~era_ataque].sum() / pesos[~era_ataque].sum()), 4
        ),
    }
    if camadas is not None:
        total = pesos.sum()
        saida["fracao_do_trafego_por_camada"] = {
            c: round(float(pesos[np.array(camadas) == c].sum() / total), 3)
            for c in sorted(set(camadas))
        }
    return saida


def relatorio(
    verdade: list[str], previsto: list[str], fontes: list[str], camadas: list[str] | None = None
) -> dict:
    """Pacote completo de uma abordagem: métricas, IC95, matriz, por fonte e prevalência."""
    return {
        **metricas(verdade, previsto),
        "ic95": ic95_bootstrap(verdade, previsto),
        "matriz_confusao": matriz(verdade, previsto),
        "por_fonte": por_fonte(verdade, previsto, fontes),
        "com_prevalencia_realista": com_prevalencia(verdade, previsto, camadas),
    }
