# Guardrails & Moderação em Tempo Real para Delivery

Pipeline de moderação em cascata (**regras determinísticas + classificador estatístico em CPU + juiz LLM**) para chats de entrega, avaliações de lojas e assistentes de IA: detecta *prompt injection*, tentativas de golpe, ofensas/ameaças e dados pessoais (PII), e **mascara PII antes de qualquer chamada a um LLM externo**. Resultado principal: a cascata completa chega a **F1 macro 0,955** (IC95 0,928–0,979) numa referência de 250 mensagens, mais que o LLM sozinho (0,854), com cerca de 80% das mensagens decididas na CPU em ~1 ms. As ressalvas pesam: dados parcialmente sintéticos, **8,8% de falso bloqueio** (IC95 3,9–14,3%, quase todo em tweets de opinião política), referência quase balanceada e precisão do bloqueio baixa em tráfego realista.

[![CI](https://github.com/krosct/project-nov26-p5-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/krosct/project-nov26-p5-guardrails/actions/workflows/ci.yml)

## O problema

Em uma plataforma com mais de 100 milhões de pedidos por mês, chats, avaliações e assistentes baseados em LLM sofrem três riscos: (1) *prompt injection*, (2) golpes e ofensas, (3) vazamento de dados pessoais (LGPD). Chamar um LLM grande para toda mensagem custa caro e leva segundos; só regras deixam passar quase tudo que não usa palavra-chave óbvia. **Como combinar latência de milissegundos, proteção determinística de PII e compreensão semântica?**

## Dados (leia antes dos números)

Detalhes e justificativas em [`data/README.md`](data/README.md).

- **Reais:** avaliações da Olist (`seguro` e base do `pii_exposto`) e tweets anotados (`ofensa_ameaca` e `seguro`; só os tweets em que os 3 anotadores concordam).
- **Sintéticos:** `prompt_injection` (dataset da comunidade gerado por LLM), `fraude_golpe` (tentativas de golpe geradas por template neste projeto) e o dado pessoal inserido em `pii_exposto` (não há PII real em bases públicas). Os números dessas classes medem a detecção de *esses* padrões, não de golpes reais.
- **`fraude_golpe` é a tentativa de golpe, não o relato.** Quem conta ter recebido produto falsificado faz uma reclamação legítima (`seguro`); as avaliações reais da Olist que citam "golpe" ou "falsificado" entram como `seguro`. Assim nenhuma palavra-chave separa as classes.
- **Sem atalho de fonte:** tweets, dataset de injeção e Olist aparecem em mais de uma classe, e o dado pessoal entra em posição e formato variados.
- **Rótulos:** treino e validação têm rótulo por construção (fonte + regra), com os tweets restritos aos de consenso dos 3 anotadores da base. **Os 250 itens da referência foram revisados um a um**; 15 rótulos mudaram (13 tweets de opinião política, que a base marcava como ódio, passaram a `seguro`; 1 nota de injeção fraca virou `seguro`; 1 isca de "clique aqui" virou `fraude_golpe`), com motivo em [`data/referencia_decisoes.csv`](data/referencia_decisoes.csv). **Treino e validação não foram revisados**, então o critério de ódio da referência é mais estrito que o do treino (ver falso bloqueio abaixo).
- 280 exemplos por classe: 200 treino, 30 validação, 50 referência, estratificados por fonte, sem duplicatas mesmo após normalizar caixa e pontuação.

## Resultados

Avaliação em **250 mensagens de referência** (64 `seguro`, 51 `fraude_golpe`, 50 `pii_exposto`, 48 `prompt_injection`, 37 `ofensa_ameaca` após a revisão), que não participam de treino nem de nenhuma escolha. O limiar da cascata (**0,65**) foi escolhido **só na validação**, por um critério fixado antes (maior limiar com até 15% de escalonamento ao LLM; critério e candidatos em [`dominio.py`](src/project_nov26_p5_guardrails/dominio.py), resultado em [`models/limiar.json`](models/limiar.json)). Tabelas geradas por `scripts/tabela_readme.py` a partir de [`reports/`](reports/); intervalos de confiança por bootstrap (2.000 reamostragens, seed 42).

<!-- TABELA:resultados -->
| Abordagem | F1 macro (IC95) | Acurácia | Kappa | Recall de bloqueio (IC95) | Falso bloqueio (IC95) | Latência p50 / p95 | Custo / 1M msgs |
|---|---|---|---|---|---|---|---|
| L1: regras (regex + checksum) | 0,400 (0,35–0,44) | 0,504 | 0,346 | 13,2% (0,08–0,19) | 0,0% (0,00–0,00) | 0,05 ms / 0,10 ms | US$ 0,01 |
| L2: TF-IDF + regressão logística (CPU) | 0,879 (0,84–0,92) | 0,880 | 0,851 | 100,0% (1,00–1,00) | 19,3% (0,12–0,27) | — | — |
| Cascata só CPU (L1 + L2) | 0,885 (0,84–0,92) | 0,884 | 0,855 | 100,0% (1,00–1,00) | 21,1% (0,14–0,29) | 1,17 ms / 1,57 ms | US$ 0,05 |
| **Cascata completa (L1 + L2 + L3)** | 0,955 (0,93–0,98) | 0,956 | 0,945 | 100,0% (1,00–1,00) | 8,8% (0,04–0,14) | 1,31 ms / 6017,71 ms | US$ 220,25 |
| LLM juiz em 100% do tráfego (medido) | 0,854 (0,81–0,89) | 0,852 | 0,813 | 98,5% (0,96–1,00) | 7,9% (0,03–0,13) | 5100,00 ms / 10534,00 ms | US$ 1079,43 |
<!-- /TABELA:resultados -->

Latência medida em um Intel Core 5 120U; custo do LLM pelo preço da versão paga do mesmo modelo (US$ 0,60 / US$ 2,40 por milhão de tokens), com os tokens médios **medidos** nas chamadas do juiz ([`latencia_custo.json`](reports/latencia_custo.json)). A linha do LLM em 100% foi medida nas mesmas 250 mensagens (texto já redigido pela L1).

**O que os números mostram**

- **A cascata supera o LLM sozinho** (0,955 contra 0,854), sobretudo por causa de PII: o LLM em 100% recebe o texto já mascarado e acerta só 50% dos exemplos de `pii_exposto` (não vê o dado), enquanto a L1 resolve 90% deles por checksum e formato. Isso é consequência do desenho (redigir antes de enviar), não defeito do modelo.
- **Falso bloqueio de 8,8% na cascata completa (IC95 3,9–14,3%) e 7,9% no LLM sozinho.** Dos 10 falsos bloqueios da cascata completa, 9 são tweets de opinião política que a revisão da referência considerou `seguro`: o L2 foi treinado com o critério da base de origem (que os marca como ódio) e bloqueia 7 deles sem chegar ao LLM. O número reflete em parte a diferença de critério entre treino e referência, e não só erro do modelo; revisar o treino com o mesmo critério é o próximo passo.
- **L1 sozinha** quase não bloqueia ataques (recall baixo), mas tem **0% de falso bloqueio** e resolve 90% dos exemplos de PII (os 10% restantes estão escritos por extenso, fora do formato).
- **L2** bloqueia todos os ataques, mas erra os exemplos seguros que vêm da mesma fonte de uma classe de ataque (tabela por fonte abaixo) e tem 19% de falso bloqueio: é aí que o juiz LLM ajuda.
- **Intervalos largos:** com poucos exemplos por classe, cada erro entre os 64 `seguro` vale ~1,6 ponto percentual.

### Atenção: o tráfego real não é balanceado

A referência tem 20% de cada classe de ataque ou PII e só 26% de `seguro`. Em produção quase tudo é seguro, e então a **precisão do bloqueio despenca**: com a prevalência ilustrativa abaixo (97% seguro; é uma suposição, não uma medida), o mesmo falso bloqueio vira muitos bloqueios indevidos para cada ataque real.

<!-- TABELA:prevalencia -->
| Abordagem | Falso bloqueio (sobre `seguro`) | Precisão do bloqueio (97% de tráfego seguro) |
|---|---|---|
| Cascata só CPU | 34,1% | 5,7% |
| Cascata completa | 15,5% | 11,7% |
| LLM em 100% | 13,9% | 12,7% |
<!-- /TABELA:prevalencia -->

Ou seja: com 97% de tráfego seguro, **nenhuma das abordagens é utilizável sem revisão do bloqueio**: a precisão do bloqueio fica em ~6% (só CPU), ~12% (cascata completa) e ~13% (LLM em 100%), porque o falso bloqueio de 8 a 21% vira muitos bloqueios indevidos para cada ataque real. A cascata completa melhora a só CPU, mas o LLM em 100% é equivalente nesse eixo, ao custo de ~5× mais caro, ~5 s de latência e de perder PII mascarado. Antes de depender do L2 em tráfego real é preciso baixar o falso bloqueio (treino revisado, mais `seguro` difícil, calibração).

### Acurácia por fonte (conteúdo × estilo)

Se o modelo aprendesse só o estilo da fonte, erraria as fontes que aparecem nas duas classes. `tweet_nao_ofensivo` e `pi_nao_injecao` são as fontes "do lado seguro" de tweets e do dataset de injeção; têm poucos exemplos na referência (4 e 5; o `tweet_odio` tem 50, dos quais 13 foram reclassificados como `seguro`), então leia como indício, não como medida.

<!-- TABELA:fontes -->
| Fonte | n | L2 | Cascata CPU | Cascata completa |
|---|---|---|---|---|
| `golpe_sintetico` | 50 | 100,0% | 100,0% | 100,0% |
| `olist_com_pii_sintetico` | 50 | 96,0% | 90,0% | 100,0% |
| `olist_elogio` | 11 | 100,0% | 100,0% | 100,0% |
| `olist_numero_inocente` | 7 | 85,7% | 85,7% | 85,7% |
| `olist_pix_link_codigo` | 5 | 100,0% | 100,0% | 100,0% |
| `olist_reclamacao` | 9 | 55,6% | 100,0% | 100,0% |
| `olist_relato_fraude` | 9 | 100,0% | 100,0% | 100,0% |
| `pi_injecao_sintetica` | 50 | 96,0% | 96,0% | 100,0% |
| `pi_nao_injecao` | 5 | 0,0% | 0,0% | 80,0% |
| `tweet_nao_ofensivo` | 4 | 25,0% | 25,0% | 100,0% |
| `tweet_odio` | 50 | 74,0% | 74,0% | 82,0% |
<!-- /TABELA:fontes -->

O L2 erra quase todos os exemplos seguros que vêm da mesma fonte de uma classe de ataque (usa o estilo como atalho em parte), e a cascata completa recupera boa parte deles via juiz LLM. A queda em `tweet_odio` vem da revisão: os 13 tweets reclassificados como `seguro` contam como erro para quem aprendeu o critério da base.

**Teste de atalho de fonte** ([`reports/teste_atalho_fonte.json`](reports/teste_atalho_fonte.json), `scripts/teste_atalho_fonte.py`): um classificador que só prevê a fonte do texto acerta **99,6%**, então o estilo é facilmente separável. Um baseline que ignora o conteúdo e prevê a classe mais comum de cada fonte chega a **74,0%** de acurácia na referência, contra **88,0%** do L2 (67,5% × 85,0% nas fontes que têm mais de uma classe). O L2 usa conteúdo, mas boa parte do resultado é explicável só pela fonte; por isso o número que importa para a qualidade real é o ganho sobre esse piso (~14 pontos), não a acurácia absoluta.

### Auditoria dos rótulos da referência

Depois da revisão dos 250 rótulos, um juiz LLM independente (rótulo sem ver a classe) concorda com **95,2%** da referência (kappa **0,94**; [`reports/auditoria_referencia.json`](reports/auditoria_referencia.json)). Sobram **12 desacordos**, listados em [`data/referencia_revisar.csv`](data/referencia_revisar.csv): 6 tweets que a revisão passou a `seguro` e o juiz ainda vê como ofensa (critério de ódio mais estrito na revisão), 2 pedidos legítimos de mudança de formato que o juiz considera injeção, 1 sarcasmo racista que a revisão manteve como ofensa, e 3 casos de injeção ou PII que o juiz classifica diferente. As 15 mudanças feitas na revisão estão em [`data/referencia_decisoes.csv`](data/referencia_decisoes.csv). Com maioria simples na base de origem, a concordância do juiz em `ofensa_ameaca` era 58%; com consenso dos 3 anotadores e revisão, é 97%.

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
uv run --group train python scripts/pipeline.py      # dados → treino → avaliação → auditoria → figuras → tabelas → testes
uv run uvicorn project_nov26_p5_guardrails.api:app   # API em http://localhost:8000
```

O pipeline usa o juiz LLM (OpenRouter, modelo gratuito) em `avaliar` e `auditar`, com cache em disco; a chave fica em `.env.shared` (nunca versionada). Para refazer só uma parte: `--de <etapa> --ate <etapa>`.

```bash
curl -s -X POST http://localhost:8000/moderate -H "Content-Type: application/json" \
  -d '{"texto": "Meu pedido atrasou, faz o estorno no CPF 529.982.247-25 por favor.", "canal": "chat_entrega"}'
```

## Decisões e trade-offs

- **Cascata em vez de um modelo só:** regras são rápidas e exatas para PII, o L2 cobre a variação de escrita, o LLM desempata o ambíguo. Custo: três componentes para manter e uma latência de cauda alta (p95 de segundos) quando a mensagem cai no juiz.
- **Limiar escolhido por orçamento, não por F1 na referência:** o critério é "maior limiar com até 15% de escalonamento na validação". Na referência o escalonamento ficou mais alto (a distribuição da validação, com 150 exemplos, não é idêntica), e a curva de limiar da referência é só análise.
- **Validação por checksum** (CPF e Luhn) para não mascarar números de pedido, mas formatos escritos por extenso ("arroba", CPF com espaços) escapam da L1; o L2 e o juiz cobrem parte deles.
- **Redação antes do LLM:** o provedor externo recebe `[CPF]` e `[TELEFONE]`, não o dado. Custo: o juiz pode julgar pior quando o dado mascarado era o sinal principal (a L1 restaura `pii_exposto` quando detectou PII).

## Limitações e próximos passos

- **Dados:** parte sintética (golpe, injeção, PII) e rótulos por construção; revisar `data/referencia_revisar.csv` e, idealmente, testar em mensagens reais de chat de delivery.
- **Referência pequena e balanceada:** intervalos largos e números "por tráfego" dependem da prevalência (tabela acima).
- **Precisão do bloqueio em tráfego realista é baixa** para a cascata só CPU; calibrar o L2 (ou mais dados de `seguro` difíceis) e medir em amostra de tráfego real são os próximos passos.
- **Latência de cauda** do L3 (p95 de segundos): em fluxos síncronos, usar a cascata só CPU no caminho crítico e o juiz de forma assíncrona.
- **Turno único:** golpes divididos em várias mensagens exigem janela de contexto.
- **Encoder pequeno destilado** (como o BERTimbau do [Projeto 4](https://github.com/krosct/project-nov26-p4-destilacao)) no lugar do TF-IDF para reduzir a fração que escala ao LLM.
