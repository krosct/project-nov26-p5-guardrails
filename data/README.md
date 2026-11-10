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

`scripts/gerar_dados.py` monta 280 exemplos por classe (`seed=42`) e divide em treino/validação/referência **estratificado por fonte**, depois de remover duplicatas **após normalizar** caixa, pontuação e espaços (não só texto idêntico). Um `assert` confere que nenhuma chave normalizada se repete entre os três arquivos.

| Arquivo | Linhas | Papel |
|---|---|---|
| `taxonomia.md` | — | 5 classes, ação associada, exemplos e regras de desempate (também é o prompt do juiz LLM) |
| `treino.csv` | 1.000 (200/classe) | Treino do classificador L2 |
| `validacao.csv` | 150 (30/classe) | **Única** base para escolher o limiar da cascata (critério fixo em `dominio.py`) |
| `referencia.csv` | 250 (50/classe antes da revisão; 64/51/50/48/37 depois) | Medição final; não participa de treino nem de nenhuma escolha |
| `referencia_decisoes.csv` | 15 | Rótulos da referência alterados na revisão, com rótulo antes/depois e motivo |
| `referencia_revisar.csv` | 12 | Desacordos que restam entre a referência revisada e o juiz LLM independente |

Colunas: `id`, `canal`, `texto`, `rotulo`, `fonte`, `detalhe` (tipo de PII, quando aplicável).

## 🧱 Como evitar rótulo circular e atalho de fonte

- **`fraude_golpe` é a TENTATIVA de golpe**, não o relato. O cliente que conta ter recebido produto falsificado é `seguro`; as avaliações reais da Olist com palavras como "golpe", "falsificado" e "duplicidade" entram como `seguro` (fonte `olist_relato_fraude`). Nenhuma classe é rotulada por palavra-chave.
- **`pii_exposto` sem marcador fixo:** o dado entra no começo, no meio ou no fim, com frases e formatos variados, e parte é escrita de um jeito que a regra por formato não pega (ex.: "meu e-mail é lu12 arroba gmail ponto com"). O filtro **não** exige que a L1 detecte o dado. Números inocentes (pedido, protocolo de 11 dígitos, CEP, rastreio, horário) entram como `seguro`.
- **Cada tipo de fonte aparece em mais de uma classe** (tweets em `ofensa_ameaca`/`seguro`; dataset de injeção em `prompt_injection`/`seguro`; Olist em `seguro`/`pii_exposto`), então o estilo do texto não revela a classe sozinho. A avaliação reporta a acurácia por fonte (`por_fonte` nos relatórios).

## ⚠️ Limites conhecidos

- Treino e validação têm rótulo **por construção** (fonte + regra). A **referência foi revisada item a item** (250); as 15 mudanças estão em `referencia_decisoes.csv`. Treino e validação não foram revisados, então o critério de ódio da referência é mais estrito que o do treino. Um juiz LLM independente audita a referência (`scripts/auditar_referencia.py`): os desacordos restantes ficam em `referencia_revisar.csv`.
- `fraude_golpe` e `prompt_injection` são **sintéticos**: medem o quanto o sistema detecta *esses* padrões, não golpes reais.
- Tweets "não ofensivos" são `seguro` pela anotação da base de origem; alguns podem ser ríspidos.
- A referência é balanceada (20% por classe); frações por camada e precisão do bloqueio em tráfego real exigem reponderar (ver README).
