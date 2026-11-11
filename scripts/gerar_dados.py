"""Monta treino, validação e referência a partir de bases públicas (Kaggle/HF) + síntese declarada.

Uso: uv run python scripts/gerar_dados.py
Fontes reais em data/raw/ (ver data/README.md): Olist (avaliações), Brazilian Portuguese Hate
Speech (tweets) e prompt-injection-br-synthetic (dataset sintético da comunidade).

Tamanho: cada classe tem N exemplos, onde N é o tamanho da MENOR fonte real (os tweets de ódio
com consenso dos 3 anotadores). Divisão por classe: 60% treino, 20% validação, 20% referência.

Estende sem reamostrar: se data/*.csv já existem, os itens existentes ficam como estão (mesmos ids
e mesmos conjuntos, então o cache do juiz LLM continua valendo) e só o que ULTRAPASSA o que já há
é sorteado. Sem os CSVs, gera tudo do zero com a mesma lógica.

Regras que evitam rótulo circular e atalho de fonte:
- `fraude_golpe` = TENTATIVA de golpe escrita na mensagem (gerada em `sintese.py`). O RELATO de um
  cliente que sofreu fraude é `seguro`: as avaliações reais da Olist que citam golpe/falsificado
  entram como `seguro`, de modo que as palavras-chave não separam as classes.
- `pii_exposto` = avaliação real + um dado pessoal em posição e formato variados, sem filtrar pelo
  que a regra L1 detecta. Números inocentes (pedido, protocolo, CEP) entram como `seguro`.
- Cada fonte aparece em mais de uma classe: tweets não ofensivos e exemplos "não injeção" entram
  como `seguro`, então o estilo da fonte não revela a classe.
- Os rótulos de ódio são os da base (3 anotadores concordando); nenhum rótulo é reescrito aqui.
- Textos duplicados (após normalizar caixa e pontuação) nunca se repetem entre os conjuntos.
"""

import csv
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from project_nov26_p5_guardrails.dominio import CLASSES
from project_nov26_p5_guardrails.regras import redigir_pii
from project_nov26_p5_guardrails.sintese import frase_pii, golpe, inserir_frase, numero_inocente

RAIZ = Path(__file__).resolve().parents[1]
SEED = 42
COLUNAS = ["id", "canal", "texto", "rotulo", "fonte", "detalhe"]
PREFIXOS = {"treino": "trn", "validacao": "val", "referencia": "ref"}
FRACAO_VALIDACAO, FRACAO_REFERENCIA = 0.20, 0.20  # o restante (60%) é treino

# Peso de cada fonte dentro de `seguro` (proporção; vira contagem conforme o tamanho da classe)
PESOS_SEGURO = {
    "olist_elogio": 60,
    "olist_reclamacao": 50,
    "olist_relato_fraude": 50,
    "olist_pix_link_codigo": 30,
    "olist_numero_inocente": 40,
    "tweet_nao_ofensivo": 25,
    "pi_nao_injecao": 25,
}
# Fonte única de cada uma das outras classes
FONTE_UNICA = {
    "pii_exposto": "olist_com_pii_sintetico",
    "prompt_injection": "pi_injecao_sintetica",
    "fraude_golpe": "golpe_sintetico",
    "ofensa_ameaca": "tweet_odio",
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


def ngramas(texto: str, k: int = 6) -> set[str]:
    """Trechos de k palavras seguidas (ou o texto todo, se for mais curto)."""
    palavras = normalizar(texto).split()
    if len(palavras) <= k:
        return {" ".join(palavras)}
    return {" ".join(palavras[i : i + k]) for i in range(len(palavras) - k + 1)}


def sem_bases_usadas(df: pd.DataFrame, usados: set[str]) -> pd.DataFrame:
    """Tira avaliações da Olist cujo texto já serviu de base a um item existente.

    Itens `pii_exposto` e `olist_numero_inocente` são uma avaliação com uma frase inserida; o
    texto original não aparece inteiro, mas compartilha trechos longos com o item existente.
    """
    return df[df["texto"].map(lambda x: not (ngramas(x) & usados))]


def distribuir(total: int, pesos: dict[str, int]) -> dict[str, int]:
    """Reparte `total` entre as chaves na proporção dos pesos (maior resto), somando `total`."""
    soma = sum(pesos.values())
    base = {k: total * v // soma for k, v in pesos.items()}
    sobra = total - sum(base.values())
    restos = sorted(pesos, key=lambda k: (total * pesos[k]) % soma, reverse=True)
    for k in restos[:sobra]:
        base[k] += 1
    return base


def tamanhos_por_classe(n: int) -> dict[str, int]:
    ref, val = round(n * FRACAO_REFERENCIA), round(n * FRACAO_VALIDACAO)
    return {"treino": n - ref - val, "validacao": val, "referencia": ref}


class Pools:
    """Candidatos de cada fonte, já filtrados (sem PII acidental, tamanho e duplicata)."""

    def __init__(self) -> None:
        olist = pd.read_csv(RAIZ / "data/raw/olist_order_reviews_dataset.csv")
        olist = olist.dropna(subset=["review_comment_message"]).copy()
        olist["texto"] = olist["review_comment_message"].map(_limpar)
        olist = olist[olist["texto"].str.len().between(30, 340)].drop_duplicates(subset=["texto"])
        olist = olist[olist["texto"].map(lambda x: not redigir_pii(x)[1])]
        fraude = olist["texto"].str.contains(RE_RELATO_FRAUDE, case=False, na=False)
        ofensa = olist["texto"].str.contains(RE_OFENSA_LEVE, case=False, na=False)
        limpo = olist[~fraude & ~ofensa]
        self.olist = {
            "olist_elogio": limpo[limpo["review_score"].isin([4, 5])],
            "olist_reclamacao": limpo[limpo["review_score"].isin([1, 2])],
            "olist_relato_fraude": olist[fraude & ~ofensa],
            "olist_pix_link_codigo": limpo[
                limpo["texto"].str.contains(RE_PIX_LINK, case=False, na=False)
            ],
            "olist_numero_inocente": limpo,
            "olist_com_pii_sintetico": limpo[
                limpo["texto"].str.contains(RE_CONTATO, case=False, na=False)
            ],
        }

        hs = pd.read_csv(
            RAIZ / "data/raw/2019-05-28_portuguese_hate_speech_binary_classification.csv"
        )
        # Só tweets em que os 3 anotadores concordam (a maioria simples tem muito ruído)
        soma = hs[["hatespeech_G1", "hatespeech_G2", "hatespeech_G3"]].sum(axis=1)
        hs = hs[soma.isin([0, 3])].copy()
        hs["odio"] = (soma[hs.index] == 3).astype(int)
        hs["texto"] = hs["text"].map(_limpar)
        hs = hs[hs["texto"].str.len().between(30, 340)]
        hs = hs[hs["texto"].map(lambda x: not redigir_pii(x)[1])]
        hs = hs.drop_duplicates(subset=["texto"])
        hs = hs[~hs["texto"].map(normalizar).duplicated()]
        self.hs = {"tweet_odio": hs[hs["odio"] == 1], "tweet_nao_ofensivo": hs[hs["odio"] == 0]}

        pi = pd.read_csv(RAIZ / "data/raw/prompt_injection_br.csv")
        pi["texto"] = pi["content"].map(_limpar)
        pi = pi[pi["texto"].str.len().between(35, 360)]
        pi = pi[pi["texto"].map(lambda x: not redigir_pii(x)[1])].drop_duplicates(subset=["texto"])
        self.pi = {
            "pi_injecao_sintetica": pi[pi["label"] == 1],
            "pi_nao_injecao": pi[pi["label"] == 0],
        }

    @property
    def tamanho_menor_fonte_real(self) -> int:
        """N por classe: o maior valor que a menor classe real comporta (tweets de ódio)."""
        return len(self.hs["tweet_odio"])


def _amostrar(df: pd.DataFrame, n: int, usados: set[str]) -> list[str]:
    """Sorteia n textos ainda não usados (por chave normalizada) e os marca como usados."""
    if n <= 0:
        return []
    chave = df["texto"].map(normalizar)
    livres = df[~chave.isin(usados) & ~chave.duplicated()]
    if len(livres) < n:
        raise SystemExit(f"fonte sem textos suficientes: faltam {n - len(livres)}")
    textos = livres.sample(n=n, random_state=SEED)["texto"].tolist()
    usados.update(normalizar(t) for t in textos)
    return textos


def novos_da_fonte(
    fonte: str, n: int, pools: Pools, usados: set[str], rng: random.Random
) -> list[dict]:
    """n itens NOVOS da fonte (nenhum igual, após normalizar, aos já usados)."""
    if fonte == "golpe_sintetico":
        textos: list[str] = []
        while len(textos) < n:
            g = golpe(rng)
            if normalizar(g) not in usados:
                usados.add(normalizar(g))
                textos.append(g)
        rotulo, canal, detalhes = "fraude_golpe", CANAL["pii"], [""] * n
    elif fonte in pools.hs:
        textos = _amostrar(pools.hs[fonte], n, usados)
        rotulo = "ofensa_ameaca" if fonte == "tweet_odio" else "seguro"
        canal, detalhes = CANAL["tweet"], [""] * n
    elif fonte in pools.pi:
        textos = _amostrar(pools.pi[fonte], n, usados)
        rotulo = "prompt_injection" if fonte == "pi_injecao_sintetica" else "seguro"
        canal, detalhes = "assistente_ia", [""] * n
    elif fonte == "olist_com_pii_sintetico":
        base = _amostrar(pools.olist[fonte], n, usados)
        pares = [frase_pii(rng) for _ in base]
        textos = [inserir_frase(t, f, rng) for t, (f, _) in zip(base, pares, strict=True)]
        rotulo, canal, detalhes = "pii_exposto", CANAL["pii"], [tipo for _, tipo in pares]
    elif fonte == "olist_numero_inocente":
        base = _amostrar(pools.olist[fonte], n, usados)
        textos = [inserir_frase(t, numero_inocente(rng), rng) for t in base]
        rotulo, canal, detalhes = "seguro", CANAL["pii"], [""] * n
    else:  # demais fontes da Olist, todas `seguro`
        textos = _amostrar(pools.olist[fonte], n, usados)
        rotulo, canal, detalhes = "seguro", CANAL["olist"], [""] * n
    return [
        {"texto": t, "rotulo": rotulo, "fonte": fonte, "canal": canal, "detalhe": d, "id": ""}
        for t, d in zip(textos, detalhes, strict=True)
    ]


def alocar_conjuntos(novos: list[dict], faltam: dict[str, int], rng: random.Random) -> dict:
    """Reparte os itens novos de uma classe entre os conjuntos, proporcional por fonte.

    Os itens são ordenados por fonte e recebem os rótulos de conjunto intercalados na proporção do
    que falta em cada um; cada fonte fica com uma fatia parecida em todos os conjuntos.
    """
    assert sum(faltam.values()) == len(novos), (faltam, len(novos))
    por_fonte = sorted(novos, key=lambda x: x["fonte"])
    restantes, total = dict(faltam), len(novos)
    sequencia = []
    for i in range(total):
        # escolhe o conjunto mais atrasado em relação à sua cota até este ponto
        recebidos = {c: faltam[c] - restantes[c] for c in faltam}
        alvo = max(
            (c for c in faltam if restantes[c] > 0),
            key=lambda c: faltam[c] * (i + 1) / total - recebidos[c],
        )
        sequencia.append(alvo)
        restantes[alvo] -= 1
    saida: dict[str, list[dict]] = {c: [] for c in faltam}
    for item, conj in zip(por_fonte, sequencia, strict=True):
        saida[conj].append(item)
    for lista in saida.values():
        rng.shuffle(lista)
    return saida


def ler_existente() -> dict[str, list[dict]]:
    existente = {}
    for conjunto in PREFIXOS:
        caminho = RAIZ / f"data/{conjunto}.csv"
        if caminho.exists():
            with open(caminho, newline="", encoding="utf-8") as f:
                existente[conjunto] = list(csv.DictReader(f))
        else:
            existente[conjunto] = []
    return existente


def salvar_csv(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        for linha in linhas:
            w.writerow({k: linha[k] for k in COLUNAS})


def main() -> None:
    rng = random.Random(SEED)
    pools = Pools()
    n = pools.tamanho_menor_fonte_real
    alvo_conj = tamanhos_por_classe(n)
    existente = ler_existente()
    usados = {normalizar(r["texto"]) for rs in existente.values() for r in rs}
    ja_tem = Counter((r["rotulo"], r["fonte"]) for rs in existente.values() for r in rs)
    # Avaliações da Olist que já serviram de base a um item (inclusive com frase inserida) ficam fora
    bases = set()
    for rs in existente.values():
        for r in rs:
            if r["fonte"].startswith("olist"):
                bases |= ngramas(r["texto"])
    pools.olist = {f: sem_bases_usadas(df, bases) for f, df in pools.olist.items()}
    print(f"N por classe = {n} (menor fonte real); alvo por conjunto: {alvo_conj}")

    final = {c: list(rs) for c, rs in existente.items()}
    for classe in CLASSES:
        fontes = distribuir(n, PESOS_SEGURO) if classe == "seguro" else {FONTE_UNICA[classe]: n}
        novos: list[dict] = []
        for fonte, alvo in fontes.items():
            novos += novos_da_fonte(
                fonte, max(0, alvo - ja_tem[(classe, fonte)]), pools, usados, rng
            )
        tem = {c: sum(r["rotulo"] == classe for r in existente[c]) for c in PREFIXOS}
        faltam = {c: alvo_conj[c] - tem[c] for c in PREFIXOS}
        assert all(v >= 0 for v in faltam.values()), f"{classe}: já passa do alvo {faltam}"
        for conjunto, itens in alocar_conjuntos(novos, faltam, rng).items():
            final[conjunto] += itens
        print(f"{classe:17} novos: {len(novos):4d}  -> faltavam {faltam}")

    chaves = {}
    for conjunto, linhas in final.items():
        for i, linha in enumerate(linhas, start=1):
            linha["id"] = linha["id"] or f"{PREFIXOS[conjunto]}-{i:04d}"
        chaves[conjunto] = {normalizar(r["texto"]) for r in linhas}
        assert len(chaves[conjunto]) == len(linhas), f"duplicata dentro de {conjunto}"
        assert Counter(r["rotulo"] for r in linhas) == {c: alvo_conj[conjunto] for c in CLASSES}
        salvar_csv(RAIZ / f"data/{conjunto}.csv", linhas)
    assert not (chaves["treino"] & chaves["validacao"]), "Vazamento treino × validação!"
    assert not (chaves["treino"] & chaves["referencia"]), "Vazamento treino × referência!"
    assert not (chaves["validacao"] & chaves["referencia"]), "Vazamento validação × referência!"
    print({c: len(v) for c, v in final.items()}, "(sem sobreposição, nem de texto normalizado)")
    por_fonte = pd.DataFrame(final["referencia"]).groupby(["rotulo", "fonte"]).size()
    print(por_fonte.to_string())


if __name__ == "__main__":
    main()
