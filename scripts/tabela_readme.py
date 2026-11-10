"""Gera as tabelas de resultados do README a partir de reports/*.json (README = JSON, sempre).

Uso: uv run python scripts/tabela_readme.py
Substitui o trecho entre <!-- TABELA:nome --> e <!-- /TABELA:nome --> no README.md.
"""
# ruff: noqa: E501

import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def ler(nome: str) -> dict:
    return json.loads((RAIZ / f"reports/{nome}.json").read_text())


def num(x: float, casas: int = 3) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def ic(par: list[float]) -> str:
    return f"{num(par[0], 2)}–{num(par[1], 2)}"


def tabela_resultados() -> str:
    lat, custo = ler("latencia_custo"), ler("latencia_custo")["custo_usd_por_1m_mensagens"]
    ms = lat["latencia_ms"]
    linhas = [
        ("L1: regras (regex + checksum)", "regras_l1", ms["l1_regras_p50"], ms["l1_regras_p95"], custo["l1_regras"]),
        ("L2: TF-IDF + regressão logística (CPU)", "classificador_l2", None, None, None),
        ("Cascata só CPU (L1 + L2)", "cascata_cpu", ms["cascata_cpu_p50"], ms["cascata_cpu_p95"], custo["cascata_cpu"]),
        ("**Cascata completa (L1 + L2 + L3)**", "cascata_completa", ms["cascata_completa_p50"], ms["cascata_completa_p95"], custo["cascata_completa"]),
        ("LLM juiz em 100% do tráfego (medido)", "llm_100pct", ms["llm_100pct_p50"], ms["llm_100pct_p95"], custo["llm_puro_100pct"]),
    ]  # fmt: skip
    saida = [
        "| Abordagem | F1 macro (IC95) | Acurácia | Kappa | Recall de bloqueio (IC95) | Falso bloqueio (IC95) | Latência p50 / p95 | Custo / 1M msgs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nome, arq, p50, p95, c in linhas:
        r = ler(arq)
        latencia = f"{num(p50, 2)} ms / {num(p95, 2)} ms" if p50 is not None else "—"
        saida.append(
            f"| {nome} | {num(r['f1_macro'])} ({ic(r['ic95']['f1_macro'])}) | {num(r['acuracia'])} "
            f"| {num(r['kappa'])} | {pct(r['recall_bloqueio_ataques'])} ({ic(r['ic95']['recall_bloqueio_ataques'])}) "
            f"| {pct(r['taxa_falso_bloqueio'])} ({ic(r['ic95']['taxa_falso_bloqueio'])}) "
            f"| {latencia} | {'US$ ' + num(c, 2) if c is not None else '—'} |"
        )
    return "\n".join(saida)


def tabela_prevalencia() -> str:
    saida = [
        "| Abordagem | Falso bloqueio (sobre `seguro`) | Precisão do bloqueio (97% de tráfego seguro) |",
        "|---|---|---|",
    ]
    for nome, arq in [
        ("Cascata só CPU", "cascata_cpu"),
        ("Cascata completa", "cascata_completa"),
        ("LLM em 100%", "llm_100pct"),
    ]:
        p = ler(arq)["com_prevalencia_realista"]
        saida.append(f"| {nome} | {pct(p['taxa_falso_bloqueio'])} | {pct(p['precisao_do_bloqueio'])} |")  # fmt: skip
    return "\n".join(saida)


def tabela_fontes() -> str:
    fontes = ler("cascata_completa")["por_fonte"]
    l2, cpu = ler("classificador_l2")["por_fonte"], ler("cascata_cpu")["por_fonte"]
    saida = ["| Fonte | n | L2 | Cascata CPU | Cascata completa |", "|---|---|---|---|---|"]
    for f, v in fontes.items():
        saida.append(
            f"| `{f}` | {v['n']} | {pct(l2[f]['acuracia'])} | {pct(cpu[f]['acuracia'])} | {pct(v['acuracia'])} |"
        )
    return "\n".join(saida)


BLOCOS = {
    "resultados": tabela_resultados,
    "prevalencia": tabela_prevalencia,
    "fontes": tabela_fontes,
}


def main() -> None:
    caminho = RAIZ / "README.md"
    texto = caminho.read_text()
    for nome, gerar in BLOCOS.items():
        padrao = re.compile(rf"(<!-- TABELA:{nome} -->\n).*?(\n<!-- /TABELA:{nome} -->)", re.DOTALL)
        if not padrao.search(texto):
            raise SystemExit(f"marcador TABELA:{nome} não encontrado no README.md")
        texto = padrao.sub(lambda m, g=gerar: m.group(1) + g() + m.group(2), texto)
    caminho.write_text(texto)
    print("README.md atualizado a partir de reports/*.json")


if __name__ == "__main__":
    main()
