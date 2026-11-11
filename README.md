# Guardrails & Moderação em Tempo Real para Delivery

Pipeline de moderação em cascata (**regras determinísticas + classificador estatístico em CPU + juiz LLM**) para chats de entrega, avaliações de lojas e assistentes de IA: detecta *prompt injection*, tentativas de golpe, ofensas/ameaças e dados pessoais (PII), e **mascara PII antes de qualquer chamada a um LLM externo**. Resultado principal: a cascata completa chega a **F1 macro 0,969** (IC95 0,952–0,984) numa referência de 450 mensagens, mais que o LLM sozinho (0,867), com cerca de 82% das mensagens decididas na CPU em ~1,5 ms. As ressalvas pesam: parte dos dados é sintética, o falso bloqueio é de **2,8%** (IC95 0,6–5,4%) e a precisão do bloqueio em tráfego realista é baixa (~26% na cascata completa).

[![CI](https://github.com/krosct/project-nov26-p5-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/krosct/project-nov26-p5-guardrails/actions/workflows/ci.yml)

## O problema

Em uma plataforma com mais de 100 milhões de pedidos por mês, chats, avaliações e assistentes baseados em LLM sofrem três riscos: (1) *prompt injection*, (2) golpes e ofensas, (3) vazamento de dados pessoais (LGPD). Chamar um LLM grande para toda mensagem custa caro e leva segundos; só regras deixam passar quase tudo que não usa palavra-chave óbvia. **Como combinar latência de milissegundos, proteção determinística de PII e compreensão semântica?**

## Dados (leia antes dos números)

Detalhes e justificativas em [`data/README.md`](data/README.md).

- **Reais:** avaliações da Olist (`seguro` e base do `pii_exposto`) e tweets anotados (`ofensa_ameaca` e `seguro`; só os tweets em que os 3 anotadores da base concordam).
- **Sintéticos:** `prompt_injection` (dataset da comunidade gerado por LLM, sem validação humana), `fraude_golpe` (tentativas de golpe geradas por template neste projeto) e o dado pessoal inserido em `pii_exposto` (não há PII real em bases públicas). Os números dessas classes medem a detecção de *esses* padrões, não de golpes reais.
- **`fraude_golpe` é a tentativa de golpe, não o relato.** Quem conta ter recebido produto falsificado faz uma reclamação legítima (`seguro`); as avaliações reais da Olist que citam "golpe" ou "falsificado" entram como `seguro`. Assim nenhuma palavra-chave separa as classes.
- **Sem atalho de fonte:** tweets, dataset de injeção e Olist aparecem em mais de uma classe, e o dado pessoal entra em posição e formato variados.
- **Rótulos:** os de ódio são os da base (3 anotadores concordando), nos três conjuntos, para o modelo ser avaliado com o mesmo critério com que foi treinado. As demais classes têm rótulo por construção (fonte + regra). Uma revisão anterior da referência discordou da base em 13 de 50 tweets (opinião política tratada como ódio); fica como observação em [`data/revisao_anterior_discordancias.csv`](data/revisao_anterior_discordancias.csv), sem alterar nenhum rótulo.
- **Tamanho:** 451 exemplos por classe (o tamanho da menor fonte real: tweets de ódio com consenso), divididos em 60% treino (271), 20% validação (90) e 20% referência (90), sem duplicatas mesmo após normalizar caixa e pontuação.

## Resultados

Avaliação em **450 mensagens de referência** (90 por classe), que não participam de treino nem de nenhuma escolha. O limiar da cascata (**0,65**) foi escolhido **só na validação**, por um critério fixado antes (maior limiar com até 15% de escalonamento ao LLM; critério e candidatos em [`dominio.py`](src/project_nov26_p5_guardrails/dominio.py), resultado em [`models/limiar.json`](models/limiar.json)). Na validação o escalonamento foi de 13,1%; na referência, 17,6%. Tabelas geradas por `scripts/tabela_readme.py` a partir de [`reports/`](reports/); intervalos de confiança por bootstrap (2.000 reamostragens, seed 42).

<!-- TABELA:resultados -->
| Abordagem | F1 macro (IC95) | Acurácia | Kappa | Recall de bloqueio (IC95) | Falso bloqueio (IC95) | Latência p50 / p95 | Custo / 1M msgs |
|---|---|---|---|---|---|---|---|
| L1: regras (regex + checksum) | 0,395 (0,36–0,43) | 0,453 | 0,317 | 13,3% (0,09–0,17) | 0,0% (0,00–0,00) | 0,05 ms / 0,11 ms | US$ 0,01 |
| L2: TF-IDF + regressão logística (CPU) | 0,945 (0,92–0,96) | 0,947 | 0,933 | 100,0% (1,00–1,00) | 8,3% (0,05–0,13) | — | — |
| Cascata só CPU (L1 + L2) | 0,937 (0,91–0,96) | 0,938 | 0,922 | 100,0% (1,00–1,00) | 10,0% (0,06–0,14) | 1,32 ms / 1,70 ms | US$ 0,05 |
| **Cascata completa (L1 + L2 + L3)** | 0,969 (0,95–0,98) | 0,969 | 0,961 | 97,4% (0,95–0,99) | 2,8% (0,01–0,05) | 1,54 ms / 6234,31 ms | US$ 188,25 |
| LLM juiz em 100% do tráfego (medido) | 0,867 (0,84–0,90) | 0,869 | 0,836 | 95,9% (0,93–0,98) | 0,6% (0,00–0,02) | 5095,00 ms / 11370,00 ms | US$ 1072,00 |
<!-- /TABELA:resultados -->

Latência medida em um Intel Core 5 120U; custo do LLM pelo preço da versão paga do mesmo modelo (US$ 0,60 / US$ 2,40 por milhão de tokens), com os tokens médios **medidos** nas chamadas do juiz ([`latencia_custo.json`](reports/latencia_custo.json)). A linha do LLM em 100% foi medida nas mesmas 450 mensagens (texto já redigido pela L1).

**O que os números mostram**

- **A cascata supera o LLM sozinho** (0,969 contra 0,867), sobretudo por causa de PII: o LLM em 100% recebe o texto já mascarado e acerta só 50% dos exemplos de `pii_exposto` (não vê o dado), enquanto a L1 resolve 88% deles por checksum e formato. Isso é consequência do desenho (redigir antes de enviar), não defeito do modelo.
- **Falso bloqueio:** 2,8% na cascata completa (IC95 0,6–5,4%), 0,6% no LLM sozinho, 8–10% no L2 e na cascata só CPU. O LLM sozinho bloqueia menos, mas custa ~5,7× mais e perde o PII.
- **L1 sozinha** quase não bloqueia ataques (recall de 13%), mas tem **0% de falso bloqueio** e resolve 88% dos exemplos de PII (os demais estão escritos por extenso, fora do formato).
- **L2** bloqueia todos os ataques, mas erra os exemplos seguros que vêm da mesma fonte de uma classe de ataque (tabela por fonte abaixo): é aí que o juiz LLM ajuda.
- **Intervalos:** com 90 exemplos por classe, cada erro entre os 180 exemplos que não deveriam ser bloqueados (`seguro` e `pii_exposto`) vale ~0,6 ponto percentual.

### Atenção: o tráfego real não é balanceado

A referência tem 20% de cada classe. Em produção quase tudo é seguro, e então a **precisão do bloqueio despenca**: com a prevalência ilustrativa abaixo (97% seguro; é uma suposição, não uma medida), o mesmo falso bloqueio vira muitos bloqueios indevidos para cada ataque real.

<!-- TABELA:prevalencia -->
| Abordagem | Falso bloqueio (sobre `seguro`) | Precisão do bloqueio (97% de tráfego seguro) |
|---|---|---|
| Cascata só CPU | 16,5% | 11,0% |
| Cascata completa | 5,5% | 26,3% |
| LLM em 100% | 1,1% | 63,8% |
<!-- /TABELA:prevalencia -->

Ou seja: com 97% de tráfego seguro, a precisão do bloqueio fica em ~11% (só CPU), ~26% (cascata completa) e ~64% (LLM em 100%), e na cascata completa cerca de 3 de cada 4 bloqueios seriam indevidos. O LLM em 100% é o melhor nesse eixo, ao custo de ~5,7× mais caro, ~5 s de latência e de perder PII mascarado. A cascata entrega um meio-termo de custo e latência; para depender dela em tráfego real é preciso baixar o falso bloqueio do L2 (mais exemplos seguros difíceis, calibração) ou manter revisão humana nos bloqueios.

### Acurácia por fonte (conteúdo × estilo)

Se o modelo aprendesse só o estilo da fonte, erraria as fontes que aparecem nas duas classes. `tweet_nao_ofensivo` e `pi_nao_injecao` são as fontes "do lado seguro" de tweets e do dataset de injeção; têm poucos exemplos na referência (8 de cada), então leia como indício, não como medida.

<!-- TABELA:fontes -->
| Fonte | n | L2 | Cascata CPU | Cascata completa |
|---|---|---|---|---|
| `golpe_sintetico` | 90 | 100,0% | 100,0% | 100,0% |
| `olist_com_pii_sintetico` | 90 | 97,8% | 87,8% | 100,0% |
| `olist_elogio` | 20 | 100,0% | 100,0% | 100,0% |
| `olist_numero_inocente` | 12 | 91,7% | 91,7% | 91,7% |
| `olist_pix_link_codigo` | 9 | 88,9% | 100,0% | 100,0% |
| `olist_reclamacao` | 17 | 76,5% | 100,0% | 100,0% |
| `olist_relato_fraude` | 16 | 100,0% | 100,0% | 100,0% |
| `pi_injecao_sintetica` | 90 | 98,9% | 98,9% | 97,8% |
| `pi_nao_injecao` | 8 | 0,0% | 0,0% | 75,0% |
| `tweet_nao_ofensivo` | 8 | 12,5% | 12,5% | 62,5% |
| `tweet_odio` | 90 | 100,0% | 100,0% | 93,3% |
<!-- /TABELA:fontes -->

O L2 erra quase todos os exemplos seguros que vêm da mesma fonte de uma classe de ataque (usa o estilo como atalho em parte), e a cascata completa recupera boa parte deles via juiz LLM.

**Teste de atalho de fonte** ([`reports/teste_atalho_fonte.json`](reports/teste_atalho_fonte.json), `scripts/teste_atalho_fonte.py`): um classificador que só prevê a fonte do texto acerta **99,8%**, então o estilo é facilmente separável. Um baseline que ignora o conteúdo e prevê a classe mais comum de cada fonte chega a **80,0%** de acurácia na referência, contra **94,7%** do L2 (75,0% × 93,3% nas fontes que têm mais de uma classe). O L2 usa conteúdo, mas boa parte do resultado é explicável só pela fonte; por isso o número que importa para a qualidade real é o ganho sobre esse piso (~15 pontos), não a acurácia absoluta.

### Auditoria dos rótulos da referência

Um juiz LLM independente (rótulo sem ver a classe) concorda com **96,4%** da referência (kappa **0,956**; [`reports/auditoria_referencia.json`](reports/auditoria_referencia.json)). Sobram **16 desacordos**, em [`data/referencia_revisar.csv`](data/referencia_revisar.csv): 8 tweets que a base marca como ódio e o juiz vê como `seguro`, 3 pedidos legítimos de mudança de formato que o juiz considera injeção, 4 casos de injeção que o juiz classifica como golpe ou `seguro`, e 1 de PII. Eles **não alteram nenhum rótulo**: o critério é o da base nos dois lados. Com maioria simples na base de origem, a concordância do juiz em `ofensa_ameaca` era 58%; com consenso dos 3 anotadores, é 91%.

![Comparação e curva de limiar da cascata](reports/cascata_tradeoff.png)

![Matriz de confusão na referência](reports/matriz_confusao.png)

## Como funciona

```mermaid
flowchart LR
    M["Mensagem<br/>(chat, avaliação ou assistente)"] --> L1["L1 (código, ~0,05 ms)<br/>checksum CPF/Luhn + regex PII + assinaturas"]
    L1 -- "ataque explícito ou PII válido" --> D1["Decisão imediata<br/>(bloquear / mascarar)"]
    L1 -- "sem bloqueio" --> L2["L2 (CPU, ~1 ms)<br/>TF-IDF palavras + caracteres + LR"]
    L2 -- "confiança ≥ limiar" --> D2["Decisão em CPU"]
    L2 -- "confiança < limiar" --> L3["L3 (API)<br/>juiz LLM + Pydantic"]
    L3 --> D3["Decisão semântica final"]
```

1. **Taxonomia orientada a ação** ([`data/taxonomia.md`](data/taxonomia.md), [`dominio.py`](src/project_nov26_p5_guardrails/dominio.py)): 5 categorias (`seguro`, `pii_exposto`, `prompt_injection`, `fraude_golpe`, `ofensa_ameaca`) mapeadas para 3 ações (`permitir`, `mascarar`, `bloquear`). A mesma taxonomia é o prompt do juiz.
2. **L1, regras e redação de PII** ([`regras.py`](src/project_nov26_p5_guardrails/regras.py)): valida CPF (dígitos verificadores) e cartão (Luhn), mascara telefone, e-mail e chave Pix, e barra assinaturas inequívocas de injeção, tentativa de golpe e ameaça. Relato de golpe sofrido **não** bloqueia.
3. **L2, classificador em CPU** ([`classificador.py`](src/project_nov26_p5_guardrails/classificador.py)): TF-IDF de palavras (1–2) e caracteres (3–5) com regressão logística. Se prevê `pii_exposto` sem PII confirmado pela L1, descarta e escolhe a segunda classe.
4. **L3, juiz LLM com privacidade por construção** ([`llm.py`](src/project_nov26_p5_guardrails/llm.py), [`cascata.py`](src/project_nov26_p5_guardrails/cascata.py)): chamado só quando a confiança do L2 fica abaixo do limiar, e sempre com o texto **já redigido pela L1**.
5. **API FastAPI** ([`api.py`](src/project_nov26_p5_guardrails/api.py), [`Dockerfile`](Dockerfile)): `/health` e `/moderate` com contratos Pydantic.

## Como rodar

```bash
uv sync --group dev --group train
uv run --group train python scripts/pipeline.py      # dados → treino → avaliação → auditoria → atalho → figuras → tabelas → testes
uv run uvicorn project_nov26_p5_guardrails.api:app   # API em http://localhost:8000
```

O pipeline usa o juiz LLM (OpenRouter, modelo gratuito) em `avaliar` e `auditar`, com cache em disco: refazer sem mudar textos nem prompt não gasta requisições. `gerar_dados.py` estende os dados sem reamostrar o que já existe, então o cache continua valendo. A chave fica em `.env.shared` (nunca versionada). Para refazer só uma parte: `--de <etapa> --ate <etapa>`.

```bash
curl -s -X POST http://localhost:8000/moderate -H "Content-Type: application/json" \
  -d '{"texto": "Meu pedido atrasou, faz o estorno no CPF 529.982.247-25 por favor.", "canal": "chat_entrega"}'
```

## Decisões e trade-offs

- **Cascata em vez de um modelo só:** regras são rápidas e exatas para PII, o L2 cobre a variação de escrita, o LLM desempata o ambíguo. Custo: três componentes para manter e uma latência de cauda alta (p95 de segundos) quando a mensagem cai no juiz.
- **Limiar escolhido por orçamento, não por F1 na referência:** o critério é "maior limiar com até 15% de escalonamento na validação". A curva de limiar da referência é só análise.
- **Tamanho da classe pela menor fonte real:** 451 por classe mantém as classes equilibradas sem repetir exemplos. Custo: descarta tweets e avaliações que sobram, e a classe de ódio fica toda em tweets.
- **Critério da base nos dois lados:** o modelo é avaliado com o critério com que foi treinado. Custo: a base tem ruído de anotação conhecido (acordo baixo entre anotadores no artigo original), e esse ruído entra na métrica.
- **Validação por checksum** (CPF e Luhn) para não mascarar números de pedido, mas formatos escritos por extenso ("arroba", CPF com espaços) escapam da L1; o L2 e o juiz cobrem parte deles.
- **Redação antes do LLM:** o provedor externo recebe `[CPF]` e `[TELEFONE]`, não o dado. Custo: o juiz pode julgar pior quando o dado mascarado era o sinal principal (a L1 restaura `pii_exposto` quando detectou PII).

## Limitações e próximos passos

- **Dados:** parte sintética (golpe, injeção, PII) e rótulos por construção; testar em mensagens reais de chat de delivery e, se possível, revisar uma amostra dos rótulos com uma segunda pessoa.
- **Referência balanceada:** os números "por tráfego" dependem da prevalência (tabela acima).
- **Precisão do bloqueio em tráfego realista é baixa** para a cascata só CPU e para a completa; calibrar o L2 (ou mais dados de `seguro` difíceis) e medir em amostra de tráfego real são os próximos passos.
- **Latência de cauda** do L3 (p95 de segundos): em fluxos síncronos, usar a cascata só CPU no caminho crítico e o juiz de forma assíncrona.
- **Turno único:** golpes divididos em várias mensagens exigem janela de contexto.
- **Encoder pequeno destilado** (como o BERTimbau do [Projeto 4](https://github.com/krosct/project-nov26-p4-destilacao)) no lugar do TF-IDF para reduzir a fração que escala ao LLM.
