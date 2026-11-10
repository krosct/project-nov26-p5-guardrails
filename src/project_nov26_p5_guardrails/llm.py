"""Cliente de LLM e juiz de moderação (camada L3 da cascata, para casos incertos).

Reaproveita o padrão dos projetos anteriores: API compatível com OpenAI (OpenRouter),
cache determinístico em disco (hash SHA-256) e validação estrita de schema com Pydantic.
"""

import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, TypeAdapter, ValidationError

from project_nov26_p5_guardrails.dominio import TAXONOMIA_TEXTO, Categoria

# Carrega chaves do .env.shared ou .env na raiz ou na pasta do projeto
load_dotenv(find_dotenv(".env.shared", usecwd=True), override=True)
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODELO_PADRAO = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
CACHE = Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm"))

INSTRUCOES_JUIZ = f"""Você é o juiz de segurança e moderação de uma plataforma de delivery.
Classifique a mensagem recebida em EXATAMENTE UMA categoria da taxonomia abaixo.

{TAXONOMIA_TEXTO}

Responda SOMENTE com um objeto JSON no formato:
{{"categoria": "<uma_das_5_classes>", "motivo": "<frase curta e direta>"}}"""


class DecisaoJuiz(BaseModel):
    categoria: Categoria
    motivo: str


class ItemLote(BaseModel):
    n: int
    categoria: Categoria


LISTA_LOTE = TypeAdapter(list[ItemLote])

INSTRUCOES_LOTE = f"""Você é o juiz de segurança e moderação de uma plataforma de delivery.
Classifique cada mensagem numerada em EXATAMENTE UMA categoria da taxonomia abaixo.

{TAXONOMIA_TEXTO}

Responda SOMENTE com uma lista JSON no formato:
[{{"n": 1, "categoria": "seguro"}}, {{"n": 2, "categoria": "fraude_golpe"}}, ...]
usando exatamente os números recebidos."""


def _client() -> OpenAI:
    return OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
        timeout=180.0,
    )


def chamar_llm(
    mensagens: list[ChatCompletionMessageParam],
    modelo: str = MODELO_PADRAO,
    max_tentativas: int = 4,
    **kwargs,
) -> dict:
    """Chama o LLM com cache em disco e retry com backoff exponencial."""
    chave = hashlib.sha256(json.dumps([mensagens, modelo, kwargs], sort_keys=True).encode())
    arquivo = CACHE / f"{chave.hexdigest()}.json"
    if arquivo.exists():
        return {**json.loads(arquivo.read_text(encoding="utf-8")), "cache": True}

    for tentativa in range(max_tentativas):
        try:
            inicio = time.perf_counter()
            resp = _client().chat.completions.create(
                model=modelo, messages=mensagens, temperature=0, **kwargs
            )
            texto = resp.choices[0].message.content
            if not texto:
                raise ValueError("Resposta vazia do LLM")
            resultado = {
                "texto": texto,
                "latencia_s": round(time.perf_counter() - inicio, 3),
                "tokens_entrada": resp.usage.prompt_tokens if resp.usage else 0,
                "tokens_saida": resp.usage.completion_tokens if resp.usage else 0,
            }
            break
        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError, ValueError):
            if tentativa == max_tentativas - 1:
                raise
            time.sleep(min(2 ** (tentativa + 1), 16))
    else:
        raise RuntimeError("Nenhuma tentativa executada")

    CACHE.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(resultado, ensure_ascii=False), encoding="utf-8")
    return {**resultado, "cache": False}


def _extrair_json(texto: str) -> str:
    m = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    return m.group(0) if m else texto


def julgar_com_llm(
    texto: str,
    modelo: str = MODELO_PADRAO,
    max_tentativas: int = 2,
) -> tuple[Categoria, str, dict]:
    """Avalia uma mensagem ambígua no LLM e valida o retorno via Pydantic."""
    mensagens: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": INSTRUCOES_JUIZ},
        {"role": "user", "content": texto},
    ]
    meta = {"latencia_s": 0.0, "tokens_entrada": 0, "tokens_saida": 0, "requisicoes": 0}
    for _ in range(max_tentativas):
        r = chamar_llm(mensagens, modelo=modelo)
        meta["latencia_s"] = round(meta["latencia_s"] + r["latencia_s"], 3)
        meta["tokens_entrada"] += r.get("tokens_entrada") or 0
        meta["tokens_saida"] += r.get("tokens_saida") or 0
        meta["requisicoes"] += 1
        try:
            decisao = DecisaoJuiz.model_validate_json(_extrair_json(r["texto"]))
            return decisao.categoria, decisao.motivo, meta
        except (ValidationError, json.JSONDecodeError) as erro:
            mensagens = [
                *mensagens,
                {"role": "assistant", "content": r["texto"]},
                {"role": "user", "content": f"JSON inválido ({erro}). Devolva só o objeto JSON."},
            ]
    # Fallback conservador se o LLM falhar após todas as tentativas
    return "seguro", "fallback_erro_parse_llm", meta


def _extrair_lista_json(texto: str) -> str:
    m = re.search(r"\[.*\]", texto, flags=re.DOTALL)
    return m.group(0) if m else texto


def _classificar_lote(
    textos: list[str], modelo: str = MODELO_PADRAO, max_tentativas: int = 3
) -> Sequence[Categoria | None]:
    numerados = "\n".join(f"{i}. {t}" for i, t in enumerate(textos, start=1))
    mensagens: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": INSTRUCOES_LOTE},
        {"role": "user", "content": numerados},
    ]
    esperado = set(range(1, len(textos) + 1))
    for _ in range(max_tentativas):
        r = chamar_llm(mensagens, modelo=modelo)
        try:
            itens = LISTA_LOTE.validate_json(_extrair_lista_json(r["texto"]))
            recebidos = [i.n for i in itens]
            if set(recebidos) != esperado or len(recebidos) != len(esperado):
                raise ValueError(f"Esperado {sorted(esperado)}, recebido {recebidos}")
            por_n = {i.n: i.categoria for i in itens}
            return [por_n[i] for i in sorted(esperado)]
        except (ValidationError, ValueError, json.JSONDecodeError) as erro:
            mensagens = [
                *mensagens,
                {"role": "assistant", "content": r["texto"]},
                {
                    "role": "user",
                    "content": f"Lista JSON inválida ({erro}). Corrija e devolva só [..].",
                },
            ]
    return [julgar_com_llm(t, modelo=modelo)[0] for t in textos]


def rotular_em_lotes(
    textos: list[str],
    modelo: str = MODELO_PADRAO,
    tamanho_lote: int = 20,
    paralelo: int = 2,
) -> list[Categoria | None]:
    """Rotula uma lista de textos em lotes paralelos com o LLM professor."""
    lotes = [textos[i : i + tamanho_lote] for i in range(0, len(textos), tamanho_lote)]
    saida: list[Categoria | None] = []
    with ThreadPoolExecutor(max_workers=paralelo) as ex:
        for idx, lote_res in enumerate(
            ex.map(lambda lote: _classificar_lote(lote, modelo=modelo), lotes), start=1
        ):
            saida.extend(lote_res)
            print(
                f"  Lote {idx}/{len(lotes)} concluído ({len(saida)}/{len(textos)} textos)",
                flush=True,
            )
    return saida
