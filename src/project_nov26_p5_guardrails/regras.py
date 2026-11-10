"""Detecção e mascaramento determinístico de PII e bloqueio rápido de ataques em código.

Roda em < 1 ms na CPU, antes de qualquer modelo de ML ou LLM:
1. Valida e mascara CPF (dígitos verificadores), cartão (Luhn), telefone BR, e-mail e chave Pix UUID.
2. Detecta assinaturas inequívocas de prompt injection, TENTATIVAS de golpe (pagamento por fora,
   link encurtado, pedido de código) e ameaças explícitas. Relato de golpe sofrido não bloqueia.
"""

import re
import unicodedata
from dataclasses import dataclass

from project_nov26_p5_guardrails.dominio import Categoria, TipoPII

# Padrões compilados uma única vez para manter latência sub-milissegundo
RE_EMAIL = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")
RE_PIX_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# Cartão: 13 a 16 dígitos separados ou não por espaço/hífen
RE_CARTAO = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
# CPF formatado (000.000.000-00) ou 11 dígitos precedidos de contexto (cpf, doc, chave, pix)
RE_CPF_FORMATADO = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
RE_CPF_CONTEXTO = re.compile(r"(?i)(?:cpf|chave|pix|doc|documento)\s*[:=-]?\s*(\d{11})\b")
# Telefone brasileiro: (DD) 9XXXX-XXXX ou DD 9XXXXXXXX
RE_TELEFONE = re.compile(
    r"(?:(?:\+?55\s?)?(?:\(?[1-9][0-9]\)?\s?)(?:9\d{4}[-\s]?\d{4}|[2-5]\d{3}[-\s]?\d{4}))\b"
)

# Assinaturas determinísticas de alta precisão (zero falso positivo em reclamações normais)
PADROES_INJECTION = [
    re.compile(
        r"\b(?:ignore|desconsidere|esqueca)\s+(?:todas\s+as\s+|as\s+)?(?:instrucoes|regras|ordens|comandos)",
    ),
    re.compile(r"\b(?:system\s+prompt|prompt\s+de\s+sistema|instrucoes\s+internas|modo\s+dan)\b"),
    re.compile(
        r"\b(?:voce\s+agora\s+e|aja\s+como|finja\s+ser)\s+(?:um\s+)?(?:sistema|admin|gerente|robo\s+sem\s+regras)",
    ),
    re.compile(
        r"\b(?:aprove|emita|libere|gere)\s+(?:automaticamente|imediatamente|agora)\s+(?:um\s+)?(?:reembolso|estorno|cupom|voucher)\s+(?:total|integral|de\s+r\$)",
    ),
    re.compile(r"(?:\[system\]|\[inst\]|<\|im_start\|>|###\s*system\s*:)", re.IGNORECASE),
]

PADROES_GOLPE = [
    re.compile(
        r"\b(?:taxa\s+extra|taxa\s+de\s+liberacao|frete\s+complementar|diferenca\s+da\s+maquininha).*(?:pix|link|transferencia|por\s+fora)",
    ),
    re.compile(
        r"\b(?:pague|transfira|faca\s+o\s+pix)\s+(?:por\s+fora|direto\s+na\s+chave|no\s+link|para\s+liberar)",
    ),
    re.compile(
        r"\b(?:bit\.ly|tinyurl\.com|cutt\.ly|pagar-taxa|ifood-cobranca|central-estorno|liberar-pedido)\b",
    ),
    re.compile(
        r"\b(?:me\s+passa|informe|mande|envie)\s+(?:o\s+codigo|a\s+senha)\s+(?:de\s+confirmacao|de\s+entrega|antes\s+de\s+eu\s+sair|para\s+eu\s+liberar)",
    ),
]

PADROES_OFENSA = [
    re.compile(
        r"\b(?:vou\s+te\s+matar|vou\s+quebrar\s+sua\s+cara|vou\s+te\s+pegar\s+na\s+saida|queimar\s+sua\s+moto|dar\s+um\s+tiro)\b"
    ),
    re.compile(
        r"\b(?:seu\s+lixo|sua\s+vagabunda|seu\s+arrombado|macaco\s+imundo|favelado\s+lixo|bandido\s+desgracado|verme\s+inutil|tem\s+que\s+morrer|merece\s+apanhar)\b"
    ),
]


@dataclass(frozen=True)
class ResultadoRegras:
    texto_redigido: str
    tipos_pii: list[TipoPII]
    categoria_bloqueio: Categoria | None
    regra_acionada: str | None


def normalizar(texto: str) -> str:
    """Remove acentos e converte para minúsculas para casamento robusto de regras."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return sem_acento.lower()


def validar_cpf(digitos: str) -> bool:
    """Valida os 2 dígitos verificadores oficiais do CPF brasileiro."""
    nums = [int(c) for c in digitos if c.isdigit()]
    if len(nums) != 11 or len(set(nums)) == 1:
        return False
    for i in (9, 10):
        soma = sum(nums[j] * ((i + 1) - j) for j in range(i))
        digito = (soma * 10 % 11) % 10
        if nums[i] != digito:
            return False
    return True


def validar_luhn(digitos: str) -> bool:
    """Valida número de cartão de crédito pelo algoritmo de Luhn."""
    nums = [int(c) for c in digitos if c.isdigit()]
    if not (13 <= len(nums) <= 16) or len(set(nums)) == 1:
        return False
    soma = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        soma += d
    return soma % 10 == 0


def redigir_pii(texto: str) -> tuple[str, list[TipoPII]]:
    """Mascara dados pessoais sensíveis e devolve (texto_limpo, tipos_encontrados)."""
    achados: list[TipoPII] = []

    def registrar(tipo: TipoPII, mascara: str) -> str:
        if tipo not in achados:
            achados.append(tipo)
        return mascara

    # 1. Chave Pix aleatória (UUID)
    texto = RE_PIX_UUID.sub(lambda _: registrar("chave_pix", "[CHAVE_PIX]"), texto)

    # 2. E-mail
    texto = RE_EMAIL.sub(lambda _: registrar("email", "[EMAIL]"), texto)

    # 3. Cartão de crédito (checa Luhn para não mascarar protocolos longos de pedido)
    def _sub_cartao(m: re.Match) -> str:
        bruto = m.group(0)
        return registrar("cartao", "[CARTAO]") if validar_luhn(bruto) else bruto

    texto = RE_CARTAO.sub(_sub_cartao, texto)

    # 4. CPF formatado ou com palavra-chave (checa dígitos verificadores)
    def _sub_cpf(m: re.Match) -> str:
        bruto = m.group(0)
        apenas_digitos = "".join(c for c in bruto if c.isdigit())
        if validar_cpf(apenas_digitos):
            registrar("cpf", "[CPF]")
            return m.group(0).replace(m.group(1), "[CPF]") if m.lastindex else "[CPF]"
        return bruto

    texto = RE_CPF_FORMATADO.sub(_sub_cpf, texto)
    texto = RE_CPF_CONTEXTO.sub(_sub_cpf, texto)

    # 5. Telefone brasileiro (DDD + 8 ou 9 dígitos)
    texto = RE_TELEFONE.sub(lambda _: registrar("telefone", "[TELEFONE]"), texto)

    return texto, achados


def inspecionar_regras(texto: str) -> ResultadoRegras:
    """Executa a camada L1 (determinística): redige PII e checa regras críticas."""
    norm = normalizar(texto)
    texto_redigido, tipos_pii = redigir_pii(texto)

    # Ordem de severidade da taxonomia: prompt_injection -> fraude_golpe -> ofensa_ameaca
    for padrao in PADROES_INJECTION:
        if padrao.search(norm):
            return ResultadoRegras(
                texto_redigido=texto_redigido,
                tipos_pii=tipos_pii,
                categoria_bloqueio="prompt_injection",
                regra_acionada="assinatura_prompt_injection",
            )

    for padrao in PADROES_GOLPE:
        if padrao.search(norm):
            return ResultadoRegras(
                texto_redigido=texto_redigido,
                tipos_pii=tipos_pii,
                categoria_bloqueio="fraude_golpe",
                regra_acionada="assinatura_fraude_golpe",
            )

    for padrao in PADROES_OFENSA:
        if padrao.search(norm):
            return ResultadoRegras(
                texto_redigido=texto_redigido,
                tipos_pii=tipos_pii,
                categoria_bloqueio="ofensa_ameaca",
                regra_acionada="assinatura_ofensa_ameaca",
            )

    return ResultadoRegras(
        texto_redigido=texto_redigido,
        tipos_pii=tipos_pii,
        categoria_bloqueio=None,
        regra_acionada=None,
    )
