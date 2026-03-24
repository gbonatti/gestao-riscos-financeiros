# Gestao de Riscos Financeiros

Projeto educacional em Python cobrindo os principais topicos de gestao de riscos em instituicoes financeiras: risco de taxa de juros, risco de liquidez, prevencao a lavagem de dinheiro (PLD/AML), produtos financeiros, risco de mercado e gestao de portfolio.

---

## Modulos

| # | Modulo | Descricao |
|---|--------|-----------|
| 01 | [Risco de Taxa de Juros](modulos/modulo_01_taxa_juros.py) | Duration, Convexidade, VaR, ALM, GAP de taxas |
| 02 | [Risco de Liquidez](modulos/modulo_02_liquidez.py) | LCR, NSFR, Gap de Liquidez, Stress Testing |
| 03 | [PLD / AML](modulos/modulo_03_pld_aml.py) | KYC, Z-score, Deteccao de Fraude com ML |
| 04 | [Produtos Financeiros](modulos/modulo_04_produtos_financeiros.py) | Renda Fixa, CAPM, Black-Scholes, Derivativos |
| 05 | [Risco de Mercado](modulos/modulo_05_risco_mercado.py) | VaR Parametrico, Historico, Monte Carlo |
| 06 | [Gestao de Portfolio](modulos/modulo_06_portfolio.py) | Markowitz, Fronteira Eficiente, Sharpe Ratio |
| -- | [Caso Pratico](exemplos/caso_pratico_banco.py) | Simulacao integrada de um banco hipotetico |

---

## Estrutura do Projeto

```
gestao-riscos-financeiros/
├── README.md
├── Documentation.md        # Teoria completa de todos os modulos
├── requirements.txt
├── modulos/
│   ├── __init__.py
│   ├── modulo_01_taxa_juros.py
│   ├── modulo_02_liquidez.py
│   ├── modulo_03_pld_aml.py
│   ├── modulo_04_produtos_financeiros.py
│   ├── modulo_05_risco_mercado.py
│   └── modulo_06_portfolio.py
└── exemplos/
    └── caso_pratico_banco.py
```

---

## Instalacao

```bash
# Clone o repositorio
git clone https://github.com/seu-usuario/gestao-riscos-financeiros.git
cd gestao-riscos-financeiros

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# Instale as dependencias
pip install -r requirements.txt
```

---

## Como Usar

Cada modulo e independente e pode ser executado diretamente:

```bash
# Modulo 1 — Risco de Taxa de Juros
python modulos/modulo_01_taxa_juros.py

# Modulo 2 — Risco de Liquidez
python modulos/modulo_02_liquidez.py

# Modulo 3 — PLD/AML
python modulos/modulo_03_pld_aml.py

# Modulo 4 — Produtos Financeiros
python modulos/modulo_04_produtos_financeiros.py

# Modulo 5 — Risco de Mercado
python modulos/modulo_05_risco_mercado.py

# Modulo 6 — Gestao de Portfolio
python modulos/modulo_06_portfolio.py

# Caso Pratico Integrado
python exemplos/caso_pratico_banco.py
```

---

## Dependencias

- `numpy` — calculos numericos
- `pandas` — manipulacao de dados
- `scipy` — estatistica e otimizacao
- `matplotlib` / `seaborn` — visualizacoes
- `scikit-learn` — modelos de machine learning

Ver [requirements.txt](requirements.txt) para versoes exatas.

---

## Conteudo por Modulo

### Modulo 01 — Risco de Taxa de Juros
- Precificacao de titulos de renda fixa
- Duration de Macaulay e Duration Modificada
- Convexidade e ajuste de segunda ordem
- Value at Risk (VaR) parametrico
- Simulacao de curva de juros (GAP de taxas)

### Modulo 02 — Risco de Liquidez
- Liquidity Coverage Ratio (LCR) — Basileia III
- Net Stable Funding Ratio (NSFR)
- Gap de Liquidez por prazo
- Stress Testing (corrida bancaria, crise de mercado)

### Modulo 03 — PLD / AML
- Conceitos de lavagem de dinheiro (colocacao, ocultacao, integracao)
- KYC (Know Your Customer) — score de risco
- Deteccao estatistica com Z-score
- Modelo de Machine Learning com Random Forest
- Alertas automaticos de transacoes suspeitas

### Modulo 04 — Produtos Financeiros
- Valor do dinheiro no tempo (VP, VF, juros compostos)
- Precificacao de titulos (YTM, duration)
- Modelo CAPM para acoes
- Precificacao de opcoes (Black-Scholes)
- Spread bancario e produtos de credito

### Modulo 05 — Risco de Mercado
- VaR Parametrico (normal)
- VaR Historico
- VaR Monte Carlo
- Expected Shortfall (CVaR)
- Backtesting de VaR

### Modulo 06 — Gestao de Portfolio
- Retorno e risco de carteiras
- Fronteira eficiente de Markowitz
- Otimizacao com restricoes (Scipy)
- Sharpe Ratio e Sortino Ratio
- Diversificacao e correlacao entre ativos

---

## Teoria

Para a teoria completa de cada modulo com formulas, exemplos resolvidos e questoes de revisao, consulte [Documentation.md](Documentation.md).

---

## Regulamentacao Brasileira Relevante

| Orgao | Norma | Tema |
|-------|-------|------|
| Banco Central do Brasil | Resolucao CMN 4.557/2017 | Gestao de Riscos |
| Banco Central do Brasil | Circular 3.749/2015 | LCR (Basileia III) |
| COAF | Lei 9.613/1998 | PLD/AML |
| CVM | Instrucao CVM 558 | Gestao de Recursos |
| B3 | Regulamento de Operacoes | Derivativos |

---

## Gilberto Ricardo Bonatti - Especialista Modelagem Numérica
