# Dados — Guardrails & Moderação em Tempo Real

## 📌 Fontes e o que é real ou sintético

| Fonte | Dataset | Natureza | Uso |
|---|---|---|---|
| Olist | [Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (`olist_order_reviews_dataset.csv`) | **Textos reais** de clientes | `seguro` (elogios, reclamações duras, **relatos reais de golpe/falsificação**, menções a Pix/link/código, texto com número inocente) e base do `pii_exposto` |
| Hate Speech | [Brazilian Portuguese Hate Speech](https://www.kaggle.com/datasets/hrmello/brazilian-portuguese-hatespeech-dataset) (Fortuna et al.) | **Tweets reais** anotados; só os em que os **3 anotadores concordam** (a maioria simples tinha ~40% de ruído) | `ofensa_ameaca` (discurso de ódio) e `seguro` (tweets **não** ofensivos) |
| HF | [ai-safety-brazil/prompt-injection-br-synthetic](https://huggingface.co/datasets/ai-safety-brazil/prompt-injection-br-synthetic) | **Sintético** (gerado por LLM, `evidence_tier: silver_synthetic`) | `prompt_injection` (label 1) e `seguro` (label 0: citações, mudanças autorizadas, conteúdo comum) |
| `sintese.py` | Geradores deste projeto | **Sintético** | `fraude_golpe` (tentativas de golpe) e o dado pessoal inserido nas avaliações de `pii_exposto` |

Não há dados pessoais reais em nenhum arquivo: bases anonimizadas não os têm, então CPF (com dígitos verificadores válidos), cartão (Luhn), telefone, e-mail e chave Pix são **gerados** e inseridos no texto real.

## 📁 Arquivos e separação sem vazamento

`scripts/gerar_dados.py` monta **451 exemplos por classe** (`seed=42`): o tamanho da **menor fonte real**, os tweets de ódio com consenso dos 3 anotadores (451 depois dos filtros de tamanho, PII e duplicata). Cada classe se divide em **60% treino (271), 20% validação (90) e 20% referência (90)**. O script **estende sem reamostrar**: os itens que já existiam (e já tinham sido julgados pelo LLM, então estão no cache) ficam nos mesmos conjuntos e com os mesmos ids; só o que ultrapassa foi sorteado. Duplicatas são removidas **após normalizar** caixa, pontuação e espaços, e um `assert` confere que nenhuma chave normalizada se repete entre os três arquivos.

| Arquivo | Linhas | Papel |
|---|---|---|
| `taxonomia.md` | — | 5 classes, ação associada, exemplos e regras de desempate (também é o prompt do juiz LLM) |
| `treino.csv` | 1.355 (271/classe) | Treino do classificador L2 |
| `validacao.csv` | 450 (90/classe) | **Única** base para escolher o limiar da cascata (critério fixo em `dominio.py`) |
| `referencia.csv` | 450 (90/classe) | Medição final; não participa de treino nem de nenhuma escolha |
| `revisao_anterior_discordancias.csv` | 15 | Observação: discordâncias de um revisor com a base de origem na referência antiga (13 tweets de opinião política, entre outros). **Não alteram nenhum rótulo**; ficam como registro |

Colunas: `id`, `canal`, `texto`, `rotulo`, `fonte`, `detalhe` (tipo de PII, quando aplicável).

## 🧱 Como evitar rótulo circular e atalho de fonte

- **`fraude_golpe` é a TENTATIVA de golpe**, não o relato. O cliente que conta ter recebido produto falsificado é `seguro`; as avaliações reais da Olist com palavras como "golpe", "falsificado" e "duplicidade" entram como `seguro` (fonte `olist_relato_fraude`). Nenhuma classe é rotulada por palavra-chave.
- **`pii_exposto` sem marcador fixo:** o dado entra no começo, no meio ou no fim, com frases e formatos variados, e parte é escrita de um jeito que a regra por formato não pega (ex.: "meu e-mail é lu12 arroba gmail ponto com"). O filtro **não** exige que a L1 detecte o dado. Números inocentes (pedido, protocolo de 11 dígitos, CEP, rastreio, horário) entram como `seguro`.
- **Cada tipo de fonte aparece em mais de uma classe** (tweets em `ofensa_ameaca`/`seguro`; dataset de injeção em `prompt_injection`/`seguro`; Olist em `seguro`/`pii_exposto`), então o estilo do texto não revela a classe sozinho. A avaliação reporta a acurácia por fonte (`por_fonte` nos relatórios).

## ⚠️ Limites conhecidos

- **Os rótulos de ódio são os da base**, nos três conjuntos (treino, validação e referência), para o modelo ser avaliado com o mesmo critério com que foi treinado. Um revisor discordou da base em 13 de 50 tweets da referência antiga (opinião política tratada como ódio); isso fica em `revisao_anterior_discordancias.csv` como observação, sem mudar rótulos. As demais classes têm rótulo por construção (fonte + regra). Um juiz LLM independente audita a referência (`scripts/auditar_referencia.py`).
- `fraude_golpe` e `prompt_injection` são **sintéticos**: medem o quanto o sistema detecta *esses* padrões, não golpes reais.
- Tweets "não ofensivos" são `seguro` pela anotação da base de origem; alguns podem ser ríspidos.
- A referência é balanceada (20% por classe); frações por camada e precisão do bloqueio em tráfego real exigem reponderar (ver README).
