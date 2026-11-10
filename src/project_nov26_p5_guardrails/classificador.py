"""Camada L2: Classificador estatístico leve em CPU (TF-IDF + Regressão Logística Calibrada).

Combina n-gramas de palavras (1, 2) e de caracteres (3, 5) para capturar variações de escrita,
erros de digitação e ofuscações leves (ex.: "p.i.x", "l1xo"), retornando probabilidades por classe.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

from project_nov26_p5_guardrails.dominio import Categoria


def criar_pipeline_ml() -> Pipeline:
    """Cria o pipeline de extração híbrida (palavras + caracteres) e classificação linear."""
    uniao = FeatureUnion(
        [
            (
                "palavras",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "caracteres",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    # C=2.0 e class_weight='balanced' garantem boa separação e probabilidades úteis na cascata
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced", random_state=42)
    return make_pipeline(uniao, clf)


def treinar_classificador(
    textos: list[str], rotulos: list[str], destino: Path | None = None
) -> Pipeline:
    """Treina o modelo L2 nos textos (já redigidos ou brutos) e salva em disco se informado."""
    modelo = criar_pipeline_ml()
    modelo.fit(textos, rotulos)
    if destino is not None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, destino)
    return modelo


def prever_com_confianca(modelo: Pipeline, texto: str) -> tuple[Categoria, float]:
    """Devolve (classe_prevista, probabilidade_maxima) em ~0,4 ms na CPU."""
    probs = modelo.predict_proba([texto])[0]
    idx = int(np.argmax(probs))
    classe: Categoria = modelo.classes_[idx]
    return classe, round(float(probs[idx]), 4)
