"""Roda o pipeline do P5: dados → treino → avaliação → auditoria → figuras → tabelas → testes.

Uso: uv run --group train python scripts/pipeline.py [opções]
  --de gerar_dados   começa nesta etapa
  --ate testes       para depois desta etapa
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

RAIZ = Path(__file__).resolve().parents[1]

ETAPAS = {
    "gerar_dados": ["scripts/gerar_dados.py"],
    "treinar": ["scripts/treinar.py"],
    "avaliar": ["scripts/avaliar.py"],
    "auditar": ["scripts/auditar_referencia.py"],
    "figuras": ["scripts/figuras.py"],
    "tabelas": ["scripts/tabela_readme.py"],
    "testes": ["-m", "pytest", "-q"],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--de", choices=list(ETAPAS), default="gerar_dados")
    p.add_argument("--ate", choices=list(ETAPAS), default="testes")
    args = p.parse_args()

    load_dotenv(find_dotenv(".env.shared", usecwd=True))
    nomes = list(ETAPAS)
    escolhidas = nomes[nomes.index(args.de) : nomes.index(args.ate) + 1]

    inicio_total = time.perf_counter()
    for etapa in escolhidas:
        print(f"\n=== {etapa} ===", flush=True)
        inicio = time.perf_counter()
        res = subprocess.run([sys.executable, *ETAPAS[etapa]], cwd=RAIZ)
        if res.returncode:
            sys.exit(f"Etapa '{etapa}' falhou. Corrija e continue com: --de {etapa}")
        print(f"--- {etapa}: {time.perf_counter() - inicio:.1f} s", flush=True)
    print(f"\nPipeline concluído em {time.perf_counter() - inicio_total:.1f} s.")


if __name__ == "__main__":
    main()
