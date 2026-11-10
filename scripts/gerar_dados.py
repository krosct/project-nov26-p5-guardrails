"""Monta treino, validação e referência a partir de bases públicas (Kaggle/HF) + síntese declarada.

Uso: uv run python scripts/gerar_dados.py
Fontes reais em data/raw/ (ver data/README.md): Olist (avaliações), Brazilian Portuguese Hate
Speech (tweets) e prompt-injection-br-synthetic (dataset sintético da comunidade).

Regras que evitam rótulo circular e atalho de fonte:
- `fraude_golpe` = TENTATIVA de golpe escrita na mensagem (gerada em `sintese.py`). O RELATO de um
  cliente que sofreu fraude é `seguro`: as avaliações reais da Olist que citam golpe/falsificado
  entram como `seguro`, de modo que as palavras-chave não separam as classes.
- `pii_exposto` = avaliação real + um dado pessoal em posição e formato variados, sem filtrar pelo
  que a regra L1 detecta (parte é escrita de um jeito que só o L2/L3 pega). Números inocentes
  (pedido, protocolo, CEP) entram como `seguro`.
- Cada fonte aparece em mais de uma classe: tweets não ofensivos e exemplos "não injeção" do
  dataset de injeção entram como `seguro`, então o estilo da fonte não revela a classe.
- Textos duplicados (após normalizar caixa e pontuação) são removidos antes de dividir, e a
  divisão treino/validação/referência é estratificada por fonte (200/30/50 de 280 por classe).
"""

import csv
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from project_nov26_p5_guardrails.dominio import CLASSES
from project_nov26_p5_guardrails.regras import redigir_pii
from project_nov26_p5_guardrails.sintese import frase_pii, golpe, inserir_frase, numero_inocente

RAIZ = Path(__file__).resolve().parents[1]
SEED = 42
POR_CLASSE = 280
N_TREINO, N_VALIDACAO, N_REFERENCIA = 200, 30, 50  # soma = POR_CLASSE

# Composição da classe `seguro` (soma = POR_CLASSE): fonte -> quantidade
COMPOSICAO_SEGURO = {
    "olist_elogio": 60,
    "olist_reclamacao": 50,
    "olist_relato_fraude": 50,
    "olist_pix_link_codigo": 30,
    "olist_numero_inocente": 40,
    "tweet_nao_ofensivo": 25,
    "pi_nao_injecao": 25,
}
# Palavras de fraude usadas só para ACHAR relatos reais (que viram `seguro`) e para excluí-los
# das demais fontes; nenhuma classe é rotulada por essa regex.
RE_RELATO_FRAUDE = (
    r"\b(?:golpe|fraude|clonad\w*|pirata\w*|falsificad\w*|estelionat\w*|enganad\w*|enganar|"
    r"roubad\w*|roubo|roubaram|falso|falsa|cobran[cç]a indevida|propaganda enganosa|"
    r"m[aá][\s-]?f[eé]|desonest\w*|picareta\w*|calote\w*|duplicidade|cobrou duas vezes|"
    r"cobrando duas vezes|cobraram duas vezes|n[aã]o comprei|envelope vazio|caixa vazia|"
    r"pedra na caixa|r[eé]plica|contraband\w*|adulterad\w*)\b"
)
RE_OFENSA_LEVE = r"\b(?:lixo|vagabund\w*|idiota|imbecil|burr\w*|palha[cç]ada|incompetente)\b"
RE_CONTATO = (
    r"\b(?:reembolso|estorno|devolver meu dinheiro|devolu[cç][aã]o|entrar em contato|"
    r"entrem em contato|me liguem|ligar|telefone|email|e-mail|nota fiscal|comprovante|fatura)\b"
)
RE_PIX_LINK = r"\b(?:pix|boleto|link|c[oó]digo|senha)\b"
CANAL = {"olist": "avaliacao_restaurante", "pii": "chat_entrega", "tweet": "chat_entrega"}


def _limpar(s: str) -> str:
    s = re.sub(r"http\S+|@\w+|RT\s+", "", str(s))
    return re.sub(r"\s+", " ", s).strip()


def normalizar(s: str) -> str:
    """Chave para achar duplicatas que só diferem em caixa, pontuação ou espaços."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-zà-ú0-9 ]", "", s.lower())).strip()


def _amostrar(df: pd.DataFrame, n: int, usados: set[str]) -> list[str]:
    """Sorteia n textos ainda não usados (por chave normalizada) e os marca como usados."""
    chave = df["texto"].map(normalizar)
    # Sem chave já usada e sem repetição DENTRO do próprio conjunto sorteado
    livres = df[~chave.isin(usados) & ~chave.duplicated()]
    textos = livres.sample(n=n, random_state=SEED)["texto"].tolist()
    usados.update(normalizar(t) for t in textos)
    return textos


def carregar_olist() -> pd.DataFrame:
    olist = pd.read_csv(RAIZ / "data/raw/olist_order_reviews_dataset.csv")
    olist = olist.dropna(subset=["review_comment_message"]).copy()
    olist["texto"] = olist["review_comment_message"].map(_limpar)
    olist = olist[olist["texto"].str.len().between(30, 340)].drop_duplicates(subset=["texto"])
    # Sem PII acidental: as regras de L1 só entram aqui para tirar textos que já tenham dado pessoal
    return olist[olist["texto"].map(lambda x: not redigir_pii(x)[1])]


def montar_candidatos(rng: random.Random) -> list[dict]:
    olist = carregar_olist()
    usados: set[str] = set()
    itens: list[dict] = []

    def add(textos: list[str], rotulo: str, fonte: str, canal: str, detalhes=None) -> None:
        for i, t in enumerate(textos):
            itens.append(
                {
                    "texto": t,
                    "rotulo": rotulo,
                    "fonte": fonte,
                    "canal": canal,
                    "detalhe": detalhes[i] if detalhes else "",
                }
            )

    fraude = olist["texto"].str.contains(RE_RELATO_FRAUDE, case=False, na=False)
    ofensa = olist["texto"].str.contains(RE_OFENSA_LEVE, case=False, na=False)
    limpo = olist[~fraude & ~ofensa]

    # 1. pii_exposto: reclamação real de contato/estorno + dado pessoal em posição variada
    contato = limpo[limpo["texto"].str.contains(RE_CONTATO, case=False, na=False)]
    base_pii = _amostrar(contato, POR_CLASSE, usados)
    textos_pii, tipos = [], []
    for t in base_pii:
        frase, tipo = frase_pii(rng)
        textos_pii.append(inserir_frase(t, frase, rng))
        tipos.append(tipo)
    add(textos_pii, "pii_exposto", "olist_com_pii_sintetico", CANAL["pii"], tipos)

    # 2. seguro (várias fontes, para que nenhuma fonte ou marcador revele a classe)
    n = COMPOSICAO_SEGURO
    add(
        _amostrar(limpo[limpo["review_score"].isin([4, 5])], n["olist_elogio"], usados),
        "seguro",
        "olist_elogio",
        CANAL["olist"],
    )
    add(
        _amostrar(limpo[limpo["review_score"].isin([1, 2])], n["olist_reclamacao"], usados),
        "seguro",
        "olist_reclamacao",
        CANAL["olist"],
    )
    add(  # relatos reais de golpe/falsificação: legítimos, vão para atendimento, não são ataque
        _amostrar(olist[fraude & ~ofensa], n["olist_relato_fraude"], usados),
        "seguro",
        "olist_relato_fraude",
        CANAL["olist"],
    )
    pix_link = limpo[limpo["texto"].str.contains(RE_PIX_LINK, case=False, na=False)]
    add(
        _amostrar(pix_link, n["olist_pix_link_codigo"], usados),
        "seguro",
        "olist_pix_link_codigo",
        CANAL["olist"],
    )
    base_num = _amostrar(limpo, n["olist_numero_inocente"], usados)
    add(
        [inserir_frase(t, numero_inocente(rng), rng) for t in base_num],
        "seguro",
        "olist_numero_inocente",
        CANAL["pii"],
    )

    hs = pd.read_csv(RAIZ / "data/raw/2019-05-28_portuguese_hate_speech_binary_classification.csv")
    # Só tweets em que os 3 anotadores concordam (a maioria simples tem ~40% de ruído: a auditoria
    # com um juiz LLM mostrou que muitos "ódio" por maioria não ofendem ninguém)
    soma = hs[["hatespeech_G1", "hatespeech_G2", "hatespeech_G3"]].sum(axis=1)
    hs = hs[soma.isin([0, 3])].copy()
    hs["hatespeech_comb"] = (soma[hs.index] == 3).astype(int)
    hs["texto"] = hs["text"].map(_limpar)
    hs = hs[
        hs["texto"].str.len().between(30, 340) & hs["texto"].map(lambda x: not redigir_pii(x)[1])
    ]
    hs = hs.drop_duplicates(subset=["texto"])
    add(
        _amostrar(hs[hs["hatespeech_comb"] == 0], n["tweet_nao_ofensivo"], usados),
        "seguro",
        "tweet_nao_ofensivo",
        CANAL["tweet"],
    )

    pi = pd.read_csv(RAIZ / "data/raw/prompt_injection_br.csv")
    pi["texto"] = pi["content"].map(_limpar)
    pi = pi[
        pi["texto"].str.len().between(35, 360) & pi["texto"].map(lambda x: not redigir_pii(x)[1])
    ]
    pi = pi.drop_duplicates(subset=["texto"])
    add(
        _amostrar(pi[pi["label"] == 0], n["pi_nao_injecao"], usados),
        "seguro",
        "pi_nao_injecao",
        "assistente_ia",
    )

    # 3. prompt_injection: dataset sintético da comunidade (declarado como sintético)
    add(
        _amostrar(pi[pi["label"] == 1], POR_CLASSE, usados),
        "prompt_injection",
        "pi_injecao_sintetica",
        "assistente_ia",
    )

    # 4. ofensa_ameaca: tweets reais anotados como discurso de ódio
    add(
        _amostrar(hs[hs["hatespeech_comb"] == 1], POR_CLASSE, usados),
        "ofensa_ameaca",
        "tweet_odio",
        CANAL["tweet"],
    )

    # 5. fraude_golpe: tentativas de golpe sintéticas, sem repetir (após normalizar)
    golpes: list[str] = []
    while len(golpes) < POR_CLASSE:
        g = golpe(rng)
        if normalizar(g) not in usados:
            usados.add(normalizar(g))
            golpes.append(g)
    add(golpes, "fraude_golpe", "golpe_sintetico", CANAL["pii"])
    return itens


def dividir(itens: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Por classe: 200 treino / 30 validação / 50 referência, estratificado por fonte."""
    treino, validacao, referencia = [], [], []
    for classe in CLASSES:
        grupo = [x for x in itens if x["rotulo"] == classe]
        assert len(grupo) == POR_CLASSE, f"{classe}: {len(grupo)} itens"
        fontes = [x["fonte"] for x in grupo]
        resto, ref = train_test_split(
            grupo, test_size=N_REFERENCIA, stratify=fontes, random_state=SEED
        )
        tr, val = train_test_split(
            resto,
            test_size=N_VALIDACAO,
            stratify=[x["fonte"] for x in resto],
            random_state=SEED,
        )
        treino += tr
        validacao += val
        referencia += ref
    return treino, validacao, referencia


def salvar_csv(caminho: Path, linhas: list[dict], prefixo_id: str) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "canal", "texto", "rotulo", "fonte", "detalhe"])
        w.writeheader()
        for idx, item in enumerate(linhas, start=1):
            w.writerow(
                {
                    "id": f"{prefixo_id}-{idx:04d}",
                    **{k: item[k] for k in ("canal", "texto", "rotulo", "fonte", "detalhe")},
                }
            )


def main() -> None:
    rng = random.Random(SEED)
    itens = montar_candidatos(rng)
    treino, validacao, referencia = dividir(itens)
    for lista in (treino, validacao, referencia):
        rng.shuffle(lista)

    chaves = [{normalizar(x["texto"]) for x in lista} for lista in (treino, validacao, referencia)]
    assert not (chaves[0] & chaves[1]), "Vazamento entre treino e validação!"
    assert not (chaves[0] & chaves[2]), "Vazamento entre treino e referência!"
    assert not (chaves[1] & chaves[2]), "Vazamento entre validação e referência!"

    salvar_csv(RAIZ / "data/treino.csv", treino, "trn")
    salvar_csv(RAIZ / "data/validacao.csv", validacao, "val")
    salvar_csv(RAIZ / "data/referencia.csv", referencia, "ref")
    print(
        f"treino={len(treino)} validacao={len(validacao)} referencia={len(referencia)} "
        "(sem sobreposição, nem de texto normalizado)"
    )
    por_fonte = pd.DataFrame(referencia).groupby(["rotulo", "fonte"]).size()
    print(por_fonte.to_string())


if __name__ == "__main__":
    main()
