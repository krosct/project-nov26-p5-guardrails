"""Geradores sintéticos usados em `scripts/gerar_dados.py`.

Duas classes não têm base pública em português: `fraude_golpe` (tentativas de golpe escritas
dentro do chat) e `pii_exposto` (dados pessoais reais não existem em bases anonimizadas). Aqui
elas são geradas com muita variação de forma, e os negativos difíceis (números inocentes,
relatos de golpe) vêm de textos reais, para que nenhum marcador fixo separe as classes.
Tudo sintético é declarado como tal em `data/README.md`.
"""

# ruff: noqa: E501  (frases de golpe são textos longos)
import random

# --- PII ---------------------------------------------------------------------------------------


def cpf_valido(rng: random.Random) -> str:
    """CPF com os dois dígitos verificadores oficiais, só dígitos."""
    nums = [rng.randint(0, 9) for _ in range(9)]
    if len(set(nums)) == 1:
        nums[0] = (nums[0] + 1) % 10
    for i in (9, 10):
        soma = sum(nums[j] * ((i + 1) - j) for j in range(i))
        nums.append((soma * 10 % 11) % 10)
    return "".join(map(str, nums))


def cartao_luhn(rng: random.Random) -> str:
    """Cartão de 16 dígitos válido pelo algoritmo de Luhn, só dígitos."""
    parcial = [rng.choice([4, 5]), *(rng.randint(0, 9) for _ in range(14))]
    soma = 0
    for i, d in enumerate(reversed(parcial)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        soma += d
    return "".join(map(str, [*parcial, (10 - soma % 10) % 10]))


def _telefone(rng: random.Random) -> list[str]:
    ddd = rng.choice([11, 21, 31, 41, 51, 61, 71, 81, 85])
    p1, p2 = rng.randint(91000, 99999), rng.randint(1000, 9999)
    return [f"({ddd}) {p1}-{p2}", f"{ddd} {p1}{p2}", f"+55 {ddd} {p1}-{p2}", f"{ddd}{p1}{p2}"]


def _email(rng: random.Random) -> str:
    nomes = ["cliente", "contato", "silva", "souza", "oliveira", "costa", "santos", "pereira", "lu"]
    dominios = ["gmail.com", "hotmail.com", "outlook.com", "uol.com.br", "yahoo.com.br"]
    sep = rng.choice([".", "_", ""])
    return f"{rng.choice(nomes)}{sep}{rng.randint(10, 9999)}@{rng.choice(dominios)}"


def _uuid(rng: random.Random) -> str:
    blocos = [8, 4, 4, 4, 12]
    return "-".join("".join(rng.choice("0123456789abcdef") for _ in range(b)) for b in blocos)


def _agrupar(digitos: str, tamanhos: list[int], sep: str) -> str:
    partes, i = [], 0
    for t in tamanhos:
        partes.append(digitos[i : i + t])
        i += t
    return sep.join(partes)


def frase_pii(rng: random.Random) -> tuple[str, str]:
    """Devolve (frase com um dado pessoal, tipo). Sem rótulo fixo nem posição fixa."""
    tipo = rng.choice(["cpf", "cartao", "telefone", "email", "chave_pix", "ofuscado"])
    if tipo == "cpf":
        d = cpf_valido(rng)
        if rng.random() < 0.5:  # formato 000.000.000-00
            v = f"{d[:3]}.{d[3:6]}.{d[6:9]}-{d[9:]}"
            modelos = ["meu cpf é {v}", "cpf {v}", "{v} é o meu cpf", "use o documento {v}"]
        else:  # só dígitos, sempre com a palavra de contexto
            v = d
            modelos = ["meu cpf é {v}", "cpf {v}", "documento: {v}", "chave pix cpf {v}"]
    elif tipo == "cartao":
        d = cartao_luhn(rng)
        v = rng.choice([_agrupar(d, [4] * 4, " "), _agrupar(d, [4] * 4, "-"), d])
        modelos = [
            "paguei com o cartão {v}",
            "cartão {v}",
            "o número do cartão é {v}",
            "{v} (cartão)",
        ]
    elif tipo == "telefone":
        v = rng.choice(_telefone(rng))
        modelos = ["me liga no {v}", "meu zap {v}", "contato: {v}", "{v} é meu telefone", "tel {v}"]
    elif tipo == "email":
        v = _email(rng)
        modelos = ["meu email é {v}", "{v}", "responde em {v}", "contato por e-mail: {v}"]
    elif tipo == "chave_pix":
        v = _uuid(rng)
        modelos = ["minha chave pix é {v}", "chave aleatória {v}", "devolve no pix {v}", "pix: {v}"]
    else:  # escrito de um jeito que a regra por formato não pega (o classificador precisa pegar)
        d = cpf_valido(rng)
        v = rng.choice(
            [
                f"meu cpf é {_agrupar(d, [3, 3, 3, 2], ' ')}",
                f"meu e-mail é {rng.choice(['lu', 'joao', 'maria'])}{rng.randint(10, 99)} arroba "
                f"gmail ponto com",
                f"meu whats é {rng.choice([11, 21, 31])} 9{rng.randint(1000, 9999)} "
                f"{rng.randint(1000, 9999)}",
            ]
        )
        return v, tipo
    return rng.choice(modelos).format(v=v), tipo


def inserir_frase(texto: str, frase: str, rng: random.Random) -> str:
    """Põe a frase no começo, no meio (entre palavras) ou no fim do texto."""
    posicao = rng.choice(["inicio", "meio", "fim"])
    frase = rng.choice(["", "", "- ", "(", "obs: "]) + frase
    if frase.startswith("("):
        frase += ")"
    if posicao == "inicio":
        return f"{frase}. {texto}"
    if posicao == "fim":
        return f"{texto} {frase}"
    palavras = texto.split()
    corte = rng.randint(max(1, len(palavras) // 3), max(1, 2 * len(palavras) // 3))
    return " ".join([*palavras[:corte], frase + ",", *palavras[corte:]])


def numero_inocente(rng: random.Random) -> str:
    """Números que parecem dados pessoais mas não são: pedido, protocolo, CEP, rastreio, hora."""
    return rng.choice(
        [
            f"pedido nº {rng.randint(100000, 999999999)}",
            f"pedido #{rng.randint(1000, 999999)}",
            f"protocolo {rng.randint(10**10, 10**11 - 1)}",  # 11 dígitos, mas é um protocolo
            f"código de rastreio BR{rng.randint(10**8, 10**9 - 1)}BR",
            f"CEP {rng.randint(10000, 99999)}-{rng.randint(100, 999)}",
            f"às {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
            f"nota fiscal {rng.randint(10**5, 10**9)}",
            f"foram R$ {rng.randint(10, 999)},{rng.randint(0, 99):02d}",
            f"atendimento número {rng.randint(10**13, 10**14 - 1)}",  # 14 dígitos, não passa Luhn
        ]
    )


# --- Tentativas de golpe ------------------------------------------------------------------------

_PERSONAS = [
    "Olá, aqui é do suporte do aplicativo.",
    "Oi, sou o entregador do seu pedido.",
    "Central de segurança informa:",
    "Atendimento oficial de reembolsos:",
    "Bom dia! Falo em nome do restaurante.",
    "ATENÇÃO cliente,",
    "Prezado cliente, identificamos um problema.",
    "Oi tudo bem? Aqui é da equipe de entregas.",
    "Setor financeiro da plataforma:",
    "Sou o gerente da loja e preciso da sua ajuda.",
    "",
    "",
]
_DOMINIOS = ["bit.ly", "tinyurl.com", "cutt.ly", "t.co", "is.gd", "rebrand.ly"]
_SLUGS = ["taxa", "estorno", "libera", "pedido", "cupom", "seguro", "reembolso", "app-entrega"]
_TATICAS = [  # cada tática tem várias formas de escrever; {link} e {valor} são preenchidos
    [
        "Seu pedido está retido por falta de pagamento de uma taxa de R$ {valor}. Faça o Pix para liberar.",
        "Para liberar a entrega é preciso pagar R$ {valor} de taxa de liberação via Pix agora.",
        "Cobramos uma taxa extra de R$ {valor} por fora do app, manda no Pix que eu entrego em 5 minutos.",
        "Falta pagar o frete complementar de R$ {valor}. Pague direto na chave que eu te passo.",
    ],
    [
        "Clique no link para confirmar seu estorno: {link}",
        "Seu reembolso de R$ {valor} está pronto, é só acessar {link} e informar seus dados.",
        "Regularize seu cadastro em {link} ou sua conta será bloqueada hoje.",
        "Ganhe um cupom de R$ {valor} acessando {link} (válido só hoje).",
    ],
    [
        "Me passa o código de confirmação que chegou por SMS para eu liberar seu reembolso.",
        "Preciso do código de entrega antes de eu sair da loja, pode me enviar agora?",
        "Informe a senha do app para validar sua conta, é rápido.",
        "Mande o código que você recebeu por mensagem, é para cancelar a cobrança duplicada.",
    ],
    [
        "Houve um erro na maquininha, faça o Pix da diferença de R$ {valor} para a minha chave pessoal.",
        "Pode pagar direto comigo, sem passar pelo app, que sai mais barato: Pix de R$ {valor}.",
        "A forma de pagamento falhou. Transfira R$ {valor} por fora para não perder o pedido.",
        "Estamos com problema no sistema, pague adiantado pelo Pix e te devolvemos depois.",
    ],
    [
        "Envie uma foto do seu cartão (frente e verso) para validarmos o estorno.",
        "Para receber o reembolso, informe o número completo do cartão e o código de segurança.",
        "Confirme seus dados bancários respondendo esta mensagem para liberar o valor de R$ {valor}.",
        "Atualize os dados do cartão com a gente por aqui mesmo, senão o pedido será cancelado.",
    ],
]
_URGENCIAS = [
    "",
    "",
    " Faça isso em até 10 minutos.",
    " Última chance!",
    " Responda urgente.",
    " Caso contrário o pedido será cancelado.",
    " 🚨",
]


def golpe(rng: random.Random) -> str:
    """Tentativa de golpe escrita pelo golpista: persona + tática + urgência, com ruído leve."""
    tatica = rng.choice(_TATICAS)
    corpo = rng.choice(tatica).format(
        link=f"{rng.choice(_DOMINIOS)}/{rng.choice(_SLUGS)}{rng.randint(1, 999)}",
        valor=f"{rng.randint(5, 250)},{rng.choice(['00', '90', '50'])}",
    )
    texto = f"{rng.choice(_PERSONAS)} {corpo}{rng.choice(_URGENCIAS)}".strip()
    if rng.random() < 0.2:
        texto = texto.lower()
    if rng.random() < 0.15:  # erro de digitação simples
        texto = texto.replace("ç", "c", 1).replace("ã", "a", 1)
    return texto
