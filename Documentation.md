# Documentation — Gestao de Riscos Financeiros

> Teoria completa, formulas, exemplos resolvidos e questoes de revisao para cada modulo.

---

## Sumario

1. [Modulo 1 — Risco de Taxa de Juros](#modulo-1)
2. [Modulo 2 — Risco de Liquidez](#modulo-2)
3. [Modulo 3 — PLD / AML](#modulo-3)
4. [Modulo 4 — Produtos Financeiros](#modulo-4)
5. [Modulo 5 — Risco de Mercado](#modulo-5)
6. [Modulo 6 — Gestao de Portfolio](#modulo-6)
7. [Integracao dos Riscos](#integracao)
8. [Questoes de Revisao](#questoes)

---

## Modulo 1 — Risco de Taxa de Juros {#modulo-1}

### 1.1 Conceito

O risco de taxa de juros e a possibilidade de perda decorrente de movimentos adversos nas taxas de juros. Afeta principalmente:

- **Ativos**: titulos, emprestimos, derivativos
- **Passivos**: depositos, emissoes de divida

**Relacao fundamental**: quando a taxa sobe, o preco do titulo CAI; quando a taxa cai, o preco SOBE.

### 1.2 Precificacao de Titulos

O preco de um titulo e o valor presente de todos os seus fluxos de caixa:

```
        n      CF_t
P = sum --------
       t=1  (1+i)^t
```

Onde:
- `P` = preco do titulo
- `CF_t` = fluxo de caixa no periodo t (cupom ou principal)
- `i` = taxa de desconto (yield)
- `n` = numero de periodos

**Exemplo**: Titulo com valor de face R$ 1.000, cupom de 10% a.a., vencimento em 3 anos, yield de 12%:

```
P = 100/(1,12)^1 + 100/(1,12)^2 + 1100/(1,12)^3
P = 89,29 + 79,72 + 782,96
P = R$ 951,97
```

Como o cupom (10%) e menor que o yield (12%), o titulo e negociado com DESCONTO.

### 1.3 Duration de Macaulay

Mede o prazo medio ponderado dos fluxos de caixa (em anos). E uma medida de sensibilidade ao risco de taxa de juros.

```
        n    t * CF_t / (1+i)^t
D_Mac = sum ---------------------
       t=1         P
```

**Exemplo**: Titulo a 3 anos, cupom 10%, yield 12%:

| t | CF_t | VP(CF_t) | t * VP(CF_t) |
|---|------|----------|--------------|
| 1 | 100  | 89,29    | 89,29        |
| 2 | 100  | 79,72    | 159,44       |
| 3 | 1100 | 782,96   | 2348,88      |

```
D_Mac = (89,29 + 159,44 + 2348,88) / 951,97
D_Mac = 2597,61 / 951,97 = 2,73 anos
```

### 1.4 Duration Modificada

Ajusta a Duration de Macaulay para medir a sensibilidade percentual do preco a variacoes de taxa:

```
D_mod = D_Mac / (1 + i)
```

A variacao aproximada do preco e:

```
deltaP / P ≈ -D_mod * delta_i
```

**Interpretacao**: Se D_mod = 2,5 e a taxa sobe 1% (0,01):
```
deltaP/P ≈ -2,5 * 0,01 = -2,5%
```
O preco do titulo cai aproximadamente 2,5%.

### 1.5 Convexidade

A duration e uma aproximacao linear. A convexidade corrige o erro de segunda ordem:

```
          n    t*(t+1) * CF_t / (1+i)^(t+2)
C = sum  -----------------------------------
         t=1               P
```

Variacao do preco com convexidade:

```
deltaP/P ≈ -D_mod * delta_i + (1/2) * C * (delta_i)^2
```

**Regra pratica**: Quanto maior a convexidade, melhor para o investidor — o titulo sobe mais do que o esperado quando a taxa cai, e cai menos quando a taxa sobe.

### 1.6 Value at Risk (VaR) de Juros

```
VaR = Z * sigma * sqrt(t) * P
```

Onde:
- `Z` = quantil normal (1,65 para 95%; 2,33 para 99%)
- `sigma` = volatilidade diaria da taxa de juros
- `t` = horizonte de tempo (dias)
- `P` = valor da posicao

### 1.7 GAP de Taxas e ALM

**GAP = Ativos sensiveis a taxa - Passivos sensiveis a taxa**

- GAP > 0: banco se beneficia quando taxas sobem
- GAP < 0: banco se beneficia quando taxas caem

**ALM (Asset-Liability Management)**: gestao integrada do balanco para controlar o risco de descasamento entre ativos e passivos.

---

## Modulo 2 — Risco de Liquidez {#modulo-2}

### 2.1 Conceito

Risco de nao conseguir honrar obrigacoes financeiras no vencimento, ou de ter que vender ativos com perda expressiva para obter caixa.

**Dois tipos**:
1. **Liquidez de financiamento**: incapacidade de rolar dividas ou captar novos recursos
2. **Liquidez de mercado**: incapacidade de vender um ativo sem impactar seu preco

### 2.2 Indicadores de Basileia III

#### LCR — Liquidity Coverage Ratio

Garante que o banco tenha HQLA suficiente para sobreviver 30 dias de estresse:

```
LCR = HQLA / Saidas liquidas de caixa em 30 dias >= 100%
```

**HQLA (High Quality Liquid Assets)**: ativos de alta liquidez (caixa, titulos publicos, etc.)

**Exemplo**:
- HQLA = R$ 200 milhoes
- Saidas liquidas = R$ 150 milhoes
- LCR = 200/150 = 133% ✓ (acima de 100%)

#### NSFR — Net Stable Funding Ratio

Garante funding estavel de longo prazo (horizonte de 1 ano):

```
NSFR = Funding estavel disponivel / Funding estavel necessario >= 100%
```

### 2.3 Gap de Liquidez

Diferenca entre entradas e saidas de caixa em cada prazo:

```
Gap(t) = Entradas(t) - Saidas(t)
```

Gap acumulado negativo indica risco de liquidez no horizonte.

| Prazo   | Entradas | Saidas | Gap    | Gap Acum. |
|---------|----------|--------|--------|-----------|
| 1-7d    | 500      | 800    | -300   | -300      |
| 8-30d   | 1200     | 900    | +300   | 0         |
| 31-90d  | 2000     | 1500   | +500   | +500      |

### 2.4 Stress Testing

Simulacao de cenarios adversos:

| Cenario          | Descricao                          | Saida extra estimada |
|------------------|------------------------------------|----------------------|
| Corrida bancaria | Saque de 20% dos depositos em 5d   | R$ 500 mi            |
| Crise de mercado | Haircut de 30% em ativos liquidos  | -R$ 300 mi de HQLA   |
| Falta de credito | Nao renovacao de 50% das linhas    | R$ 200 mi            |

### 2.5 Gestao de Liquidez

- **Colchao de liquidez**: reserva minima de HQLA
- **Diversificacao de funding**: multiplas fontes de captacao
- **Plano de contingencia de liquidez (PCL)**: acoes pre-definidas para crises

---

## Modulo 3 — PLD / AML {#modulo-3}

### 3.1 Lavagem de Dinheiro — Conceito

Processo de ocultar a origem ilicita de recursos financeiros, dando-lhes aparencia de legalidade.

**As tres etapas**:

1. **Colocacao** (Placement): insercao do dinheiro ilicito no sistema financeiro
   - Depositos fracionados (smurfing)
   - Troca de notas pequenas por grandes

2. **Ocultacao** (Layering): mascaramento da origem atraves de transacoes complexas
   - Transferencias internacionais em cadeia
   - Compra e venda de ativos

3. **Integracao** (Integration): o dinheiro retorna ao sistema com aparencia legal
   - Investimentos em empresas legitimas
   - Compra de imoveis

**Legislacao brasileira**: Lei 9.613/1998 (crimes de lavagem de dinheiro)
**Orgaos**: Banco Central do Brasil, COAF, CVM

### 3.2 KYC — Know Your Customer

Processo obrigatorio de identificacao e verificacao de clientes:

**Dados coletados**:
- Identificacao: CPF/CNPJ, nome, data de nascimento
- Endereco e contato
- Renda / faturamento declarado
- Origem dos recursos
- Perfil de investidor

**Score de risco KYC**:

```
risco = (peso_pep * is_pep) + (peso_pais * risco_pais) +
        (peso_atividade * risco_atividade) + (peso_relacionamento * tempo_rel)
```

Clientes classificados como **PEP (Pessoa Exposta Politicamente)** recebem monitoramento reforçado.

### 3.3 Deteccao Estatistica — Z-score

Para identificar transacoes anomalas em relacao ao historico do cliente:

```
Z = (X - mu) / sigma
```

Onde:
- `X` = valor da transacao atual
- `mu` = media historica das transacoes do cliente
- `sigma` = desvio padrao historico

**Regra de decisao**:
- |Z| < 2: normal
- 2 <= |Z| < 3: alerta amarelo
- |Z| >= 3: alerta vermelho (suspeito)

**Exemplo**: Cliente com media de R$ 5.000 e desvio de R$ 1.500 realiza transacao de R$ 12.000:
```
Z = (12000 - 5000) / 1500 = 4,67  -->  ALERTA VERMELHO
```

### 3.4 Indicadores de Suspeicao

- Movimentacoes incompativeis com renda declarada
- Transferencias fracionadas abaixo de limites de notificacao (smurfing)
- Uso de contas de terceiros
- Operacoes com jurisdicoes de alto risco
- Estruturacoes repetitivas

### 3.5 Machine Learning para Deteccao de Fraude

**Features utilizadas**:
- Valor da transacao (absoluto e relativo a media)
- Horario da transacao (madrugada = maior risco)
- Frequencia de transacoes no dia
- Variacao em relacao ao historico (Z-score)
- Pais/regiao da transacao

**Modelos comuns**:
- **Random Forest**: robusto, resistente a overfitting, interpretavel via feature importance
- **XGBoost/LightGBM**: alta performance em dados tabulares desequilibrados
- **Isolation Forest**: deteccao de anomalias nao supervisionada
- **LSTM**: para series temporais de transacoes

**Tratamento de desbalanceamento**:
- SMOTE (Synthetic Minority Oversampling)
- Class weights
- Threshold customizado

---

## Modulo 4 — Produtos Financeiros {#modulo-4}

### 4.1 Valor do Dinheiro no Tempo

**Valor Futuro (juros compostos)**:
```
VF = VP * (1 + i)^n
```

**Valor Presente**:
```
VP = VF / (1 + i)^n
```

**Juros simples** (para prazos curtos):
```
J = VP * i * n
```

**Exemplo — Juros compostos**:
- VP = R$ 1.000
- i = 10% a.a.
- n = 5 anos
```
VF = 1000 * (1,10)^5 = R$ 1.610,51
```

### 4.2 Renda Fixa

**Tipos de titulos no Brasil**:

| Titulo | Indexador | Risco |
|--------|-----------|-------|
| Tesouro Selic | Selic | Baixissimo |
| Tesouro IPCA+ | IPCA + spread | Baixo |
| Tesouro Prefixado | Taxa fixa | Baixo (risco de mercado) |
| CDB | CDI / pre | Baixo (risco do banco) |
| LCI/LCA | CDI / IPCA | Baixo (isento IR) |
| Debentures | CDI / IPCA / pre | Medio |

**Precificacao** (YTM — Yield to Maturity):
```
        n     CF_t
0 = sum ------- - P
       t=1 (1+YTM)^t
```

O YTM e a taxa interna de retorno do titulo.

### 4.3 Renda Variavel — CAPM

**Retorno esperado de uma acao**:
```
E(Ri) = Rf + beta * (Rm - Rf)
```

Onde:
- `Rf` = taxa livre de risco (Selic, por exemplo)
- `beta` = sensibilidade ao mercado (risco sistematico)
- `Rm` = retorno esperado do mercado
- `(Rm - Rf)` = premio de risco de mercado

**Interpretacao do beta**:
- beta = 1: ativo se move com o mercado
- beta > 1: mais volatil que o mercado (agressivo)
- beta < 1: menos volatil (defensivo)
- beta < 0: se move contra o mercado (hedge natural)

**Exemplo**:
- Rf = 5%, Rm = 12%, beta = 1,2
```
E(R) = 5% + 1,2 * (12% - 5%) = 5% + 8,4% = 13,4%
```

### 4.4 Derivativos

**Tipos**:
- **Futuros**: obrigacao de comprar/vender no futuro a preco acordado
- **Opcoes**: direito (nao obrigacao) de comprar/vender
- **Swaps**: troca de fluxos de caixa entre contrapartes

**Black-Scholes — Precificacao de Opcao de Compra (Call)**:
```
C = S0 * N(d1) - K * e^(-r*T) * N(d2)

d1 = [ln(S0/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

Onde:
- `S0` = preco atual do ativo
- `K` = preco de exercicio (strike)
- `r` = taxa livre de risco
- `T` = tempo ate vencimento (em anos)
- `sigma` = volatilidade do ativo
- `N(.)` = funcao de distribuicao normal acumulada

### 4.5 Spread Bancario

```
Spread = Taxa de emprestimo - Custo de captacao
```

Componentes do spread:
- Custo de inadimplencia
- Custo operacional
- Impostos e contribuicoes
- Margem liquida do banco

---

## Modulo 5 — Risco de Mercado {#modulo-5}

### 5.1 Value at Risk (VaR)

**Definicao**: Perda maxima esperada de uma posicao/carteira em um dado horizonte de tempo, com um nivel de confianca definido.

**Exemplo**: VaR de R$ 1 milhao a 95% em 1 dia significa:
> "Ha 95% de chance de a perda nao ultrapassar R$ 1 milhao no proximo dia."

### 5.2 VaR Parametrico (Normal)

Assume distribuicao normal dos retornos:

```
VaR = Z * sigma * sqrt(t) * W
```

Onde:
- `Z` = quantil normal (1,645 para 95%; 2,326 para 99%)
- `sigma` = volatilidade (desvio padrao dos retornos)
- `t` = horizonte (dias)
- `W` = valor da posicao

**Exemplo**:
- W = R$ 1.000.000
- sigma = 2% ao dia
- Horizonte = 1 dia, confianca = 95% (Z = 1,645)
```
VaR = 1,645 * 0,02 * 1 * 1.000.000 = R$ 32.900
```

### 5.3 VaR Historico

Usa a distribuicao empirica dos retornos passados:

1. Coletar retornos historicos (ex: 500 dias)
2. Ordenar do menor para o maior
3. O VaR a 95% e o percentil 5% da distribuicao

**Vantagem**: nao assume forma da distribuicao; captura caudas gordas.

### 5.4 VaR Monte Carlo

1. Modelar os fatores de risco (ex: retornos normais ou t-Student)
2. Gerar N simulacoes de cenarios futuros (ex: 10.000)
3. Calcular o P&L em cada cenario
4. O VaR e o percentil correspondente ao nivel de confianca

**Vantagem**: flexivel, captura nao-linearidades (ex: opcoes).

### 5.5 Expected Shortfall (CVaR)

Mede a perda esperada DADO que o VaR foi superado:

```
CVaR = E[perda | perda > VaR]
```

E uma medida mais conservadora e coerente que o VaR.

### 5.6 Backtesting de VaR

Verifica se o modelo VaR e adequado:

```
taxa_excecoes = numero_de_dias_com_perda_acima_do_VaR / total_de_dias
```

**Teste de Kupiec (1995)**:
- Nivel de confianca 95%: espera-se ~5% de excecoes
- Nivel de confianca 99%: espera-se ~1% de excecoes

Muitas excecoes indicam subestimacao do risco. Poucas indicam modelo muito conservador.

---

## Modulo 6 — Gestao de Portfolio {#modulo-6}

### 6.1 Retorno Esperado de uma Carteira

```
E(Rp) = sum(wi * E(Ri))
```

Onde `wi` = peso do ativo i na carteira.

### 6.2 Risco (Variancia) de uma Carteira

Para 2 ativos:
```
sigma_p^2 = w1^2 * sigma1^2 + w2^2 * sigma2^2 + 2*w1*w2*Cov(1,2)
```

Para n ativos (forma matricial):
```
sigma_p^2 = w^T * SIGMA * w
```

Onde `SIGMA` e a matriz de covariancia dos ativos.

**Correlacao e diversificacao**:
- Correlacao = -1: diversificacao perfeita (risco zerado)
- Correlacao = 0: diversificacao parcial
- Correlacao = +1: sem beneficio de diversificacao

### 6.3 Fronteira Eficiente de Markowitz

A fronteira eficiente e o conjunto de carteiras que:
1. Maximizam retorno para dado nivel de risco, OU
2. Minimizam risco para dado nivel de retorno

**Problema de otimizacao**:
```
Minimizar:  sigma_p^2 = w^T * SIGMA * w
Sujeito a:  sum(wi) = 1
            E(Rp) = alvo
            wi >= 0  (sem venda a descoberto)
```

### 6.4 Indicadores de Performance

**Sharpe Ratio**:
```
Sharpe = (E(Rp) - Rf) / sigma_p
```
Retorno em excesso por unidade de risco total. Quanto maior, melhor.

**Sortino Ratio**:
```
Sortino = (E(Rp) - Rf) / sigma_downside
```
Usa apenas o desvio downside (so conta volatilidade negativa). Mais adequado para ativos asssimetricos.

**Indice de Treynor**:
```
Treynor = (E(Rp) - Rf) / beta_p
```
Retorno em excesso por unidade de risco sistematico (beta).

---

## Integracao dos Riscos {#integracao}

Na pratica, os riscos sao interconectados:

| Evento | Risco de Juros | Risco de Liquidez | Risco de Fraude |
|--------|---------------|-------------------|-----------------|
| Alta da SELIC | Desvaloriza carteira de titulos | Aumenta custo de funding | — |
| Corrida bancaria | — | Crise imediata | Pode ser resultado de fraude |
| Inadimplencia elevada | — | Reduz entradas de caixa | Pode indicar fraude |
| Crise cambial | Volatilidade nos ativos | Fuga de depositos | — |

**Framework integrado (ICAAP)**:
1. Identificacao dos riscos materiais
2. Quantificacao do capital necessario
3. Stress testing integrado
4. Governanca e limites
5. Relatorio periodico para o conselho

---

## Questoes de Revisao {#questoes}

### Modulo 1 — Juros

**Q1**: Se a duration modificada de um titulo e 4 anos e a taxa de juros sobe 0,5%, qual a variacao aproximada no preco?
> R: deltaP/P ≈ -4 * 0,005 = -2,0%

**Q2**: Qual a relacao entre convexidade e o preco de um titulo quando as taxas variam muito?
> R: Quanto maior a convexidade, menor a perda quando taxas sobem e maior o ganho quando taxas caem. E positivo para o detentor do titulo.

**Q3**: Um banco tem GAP de taxas negativo. O que isso significa?
> R: Os passivos sensiveis excedem os ativos sensiveis. O banco sofre quando as taxas sobem (custo de captacao aumenta mais do que a receita).

### Modulo 2 — Liquidez

**Q4**: Banco com HQLA = R$ 300 mi e saidas de 30 dias = R$ 400 mi. O banco esta em conformidade com o LCR?
> R: LCR = 300/400 = 75% — NAO esta em conformidade (minimo = 100%).

**Q5**: Qual a diferenca entre risco de liquidez de financiamento e de mercado?
> R: Financiamento = incapacidade de rolar dividas/captar; Mercado = incapacidade de vender ativos sem grande perda de preco.

### Modulo 3 — PLD/AML

**Q6**: Quais sao as tres etapas da lavagem de dinheiro?
> R: Colocacao, Ocultacao e Integracao.

**Q7**: Um cliente realiza uma transacao com Z-score = 3,8. Qual a acao recomendada?
> R: Emitir alerta vermelho e enviar para analise manual. Z >= 3 indica transacao altamente atipica.

### Modulo 4 — Produtos

**Q8**: Qual o valor presente de R$ 1.500 a receber em 4 anos, com taxa de 8% a.a.?
> R: VP = 1500 / (1,08)^4 = 1500 / 1,3605 = R$ 1.102,45

**Q9**: Uma acao tem beta = 0,8, Rf = 4%, Rm = 11%. Qual o retorno exigido?
> R: E(R) = 4% + 0,8 * (11% - 4%) = 4% + 5,6% = 9,6%

### Modulo 5 — VaR

**Q10**: Por que o Expected Shortfall (CVaR) e considerado uma medida de risco mais coerente que o VaR?
> R: O VaR nao informa o tamanho da perda quando o threshold e ultrapassado. O CVaR mede a perda media na cauda, sendo subadditivo e, portanto, satisfazendo os axiomas de medidas de risco coerentes.

### Modulo 6 — Portfolio

**Q11**: Uma carteira tem retorno de 12%, risco de 8% e taxa livre de risco de 5%. Qual o Sharpe Ratio?
> R: Sharpe = (12% - 5%) / 8% = 0,875

**Q12**: Por que adicionar um ativo com correlacao negativa a uma carteira e benefico?
> R: Reduz a variancia total da carteira sem necessariamente reduzir o retorno esperado, melhorando o Sharpe Ratio.
