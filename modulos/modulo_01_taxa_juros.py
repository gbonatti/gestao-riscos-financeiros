"""
Modulo 01 — Risco de Taxa de Juros
====================================
Conteudo:
  - Precificacao de titulos de renda fixa
  - Duration de Macaulay e Duration Modificada
  - Convexidade
  - Value at Risk (VaR) de posicao em renda fixa
  - Simulacao de GAP de taxas (ALM simplificado)
  - Graficos de sensibilidade
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. PRECIFICACAO DE TITULOS
# ---------------------------------------------------------------------------

def preco_titulo(cupom_rate: float, face: float, yield_: float, n_periodos: int) -> float:
    """
    Calcula o preco de mercado de um titulo de renda fixa.

    Parameters
    ----------
    cupom_rate : float  - Taxa de cupom (ex: 0.10 para 10%)
    face       : float  - Valor de face / principal
    yield_     : float  - Taxa de desconto de mercado (YTM)
    n_periodos : int    - Numero de periodos ate vencimento

    Returns
    -------
    float - Preco do titulo
    """
    cupom = cupom_rate * face
    fluxos = [cupom] * n_periodos
    fluxos[-1] += face  # repagamento do principal no ultimo periodo

    preco = sum(cf / (1 + yield_) ** t for t, cf in enumerate(fluxos, start=1))
    return preco


def tabela_precificacao(cupom_rate: float, face: float, n_periodos: int,
                        yields: list) -> pd.DataFrame:
    """Calcula precos para varios cenarios de yield."""
    dados = []
    for y in yields:
        p = preco_titulo(cupom_rate, face, y, n_periodos)
        tipo = "desconto" if y > cupom_rate else ("par" if y == cupom_rate else "premio")
        dados.append({"yield (%)": round(y * 100, 2), "preco (R$)": round(p, 2), "tipo": tipo})
    return pd.DataFrame(dados)


# ---------------------------------------------------------------------------
# 2. DURATION
# ---------------------------------------------------------------------------

def duration_macaulay(cupom_rate: float, face: float, yield_: float, n_periodos: int) -> float:
    """
    Duration de Macaulay: prazo medio ponderado dos fluxos de caixa.
    """
    cupom = cupom_rate * face
    fluxos = [cupom] * n_periodos
    fluxos[-1] += face

    preco = preco_titulo(cupom_rate, face, yield_, n_periodos)
    numerador = sum(t * (cf / (1 + yield_) ** t) for t, cf in enumerate(fluxos, start=1))
    return numerador / preco


def duration_modificada(cupom_rate: float, face: float, yield_: float, n_periodos: int) -> float:
    """
    Duration Modificada: sensibilidade percentual do preco a variacao de taxa.
    D_mod = D_Mac / (1 + yield)
    """
    d_mac = duration_macaulay(cupom_rate, face, yield_, n_periodos)
    return d_mac / (1 + yield_)


def variacao_preco_duration(d_mod: float, delta_yield: float) -> float:
    """
    Aproximacao linear da variacao percentual do preco.
    deltaP/P ≈ -D_mod * delta_yield
    """
    return -d_mod * delta_yield


# ---------------------------------------------------------------------------
# 3. CONVEXIDADE
# ---------------------------------------------------------------------------

def convexidade(cupom_rate: float, face: float, yield_: float, n_periodos: int) -> float:
    """
    Convexidade de um titulo: ajuste de segunda ordem para variacao de preco.
    """
    cupom = cupom_rate * face
    fluxos = [cupom] * n_periodos
    fluxos[-1] += face

    preco = preco_titulo(cupom_rate, face, yield_, n_periodos)
    numerador = sum(
        t * (t + 1) * cf / (1 + yield_) ** (t + 2)
        for t, cf in enumerate(fluxos, start=1)
    )
    return numerador / preco


def variacao_preco_convexidade(d_mod: float, conv: float, delta_yield: float) -> float:
    """
    Variacao percentual do preco com ajuste de convexidade.
    deltaP/P ≈ -D_mod * delta_yield + (1/2) * C * delta_yield^2
    """
    return -d_mod * delta_yield + 0.5 * conv * delta_yield ** 2


# ---------------------------------------------------------------------------
# 4. VaR DE POSICAO EM RENDA FIXA
# ---------------------------------------------------------------------------

def var_renda_fixa(posicao: float, d_mod: float, sigma_yield: float,
                   horizonte: int = 1, nivel_confianca: float = 0.95) -> float:
    """
    VaR parametrico de uma posicao em renda fixa.

    Logica: VaR_retorno = Z * sigma_yield * sqrt(t)
            VaR_preco   = posicao * D_mod * VaR_retorno

    Parameters
    ----------
    posicao        : float - Valor da posicao em R$
    d_mod          : float - Duration modificada
    sigma_yield    : float - Volatilidade diaria da taxa de juros
    horizonte      : int   - Dias
    nivel_confianca: float - Ex: 0.95 ou 0.99

    Returns
    -------
    float - VaR em R$
    """
    z = norm.ppf(nivel_confianca)
    var_yield = z * sigma_yield * np.sqrt(horizonte)
    return posicao * d_mod * var_yield


# ---------------------------------------------------------------------------
# 5. GAP DE TAXAS (ALM SIMPLIFICADO)
# ---------------------------------------------------------------------------

def gap_taxas(ativos_sensiveis: float, passivos_sensiveis: float) -> dict:
    """
    Calcula o GAP de reprecificacao entre ativos e passivos sensiveis a taxa.

    GAP > 0: banco ganha quando taxa sobe (funding gap positivo)
    GAP < 0: banco perde quando taxa sobe (funding gap negativo)
    """
    gap = ativos_sensiveis - passivos_sensiveis
    interpretacao = (
        "POSITIVO — banco se beneficia quando taxas SOBEM"
        if gap > 0 else (
            "NEGATIVO — banco se beneficia quando taxas CAEM"
            if gap < 0 else "NEUTRO"
        )
    )
    impacto_por_ponto = gap * 0.01  # impacto na margem financeira com +1%
    return {
        "ativos_sensiveis": ativos_sensiveis,
        "passivos_sensiveis": passivos_sensiveis,
        "gap": gap,
        "interpretacao": interpretacao,
        "impacto_1pct_taxa (R$)": impacto_por_ponto,
    }


# ---------------------------------------------------------------------------
# 6. GRAFICO — RELACAO PRECO x YIELD
# ---------------------------------------------------------------------------

def grafico_preco_yield(cupom_rate: float, face: float, n_periodos: int,
                        titulo: str = "Titulo CDB 10% a.a.") -> None:
    """Plota a curva preco vs. yield de um titulo."""
    yields = np.linspace(0.01, 0.25, 200)
    precos = [preco_titulo(cupom_rate, face, y, n_periodos) for y in yields]

    par_yield = cupom_rate
    par_preco = preco_titulo(cupom_rate, face, par_yield, n_periodos)

    plt.figure(figsize=(10, 5))
    plt.plot(yields * 100, precos, color="steelblue", linewidth=2.5, label="Curva Preco-Yield")
    plt.axvline(par_yield * 100, color="gray", linestyle="--", alpha=0.6, label=f"Yield par ({par_yield*100:.0f}%)")
    plt.axhline(face, color="green", linestyle="--", alpha=0.5, label=f"Face = R${face:,.0f}")
    plt.scatter([par_yield * 100], [par_preco], color="red", zorder=5, s=80, label="Preco ao par")
    plt.title(f"Curva Preco vs. Yield — {titulo}", fontsize=13)
    plt.xlabel("Yield (%)")
    plt.ylabel("Preco (R$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_preco_yield.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_preco_yield.png'")


def grafico_duration_vencimento(cupom_rate: float, face: float, yield_: float,
                                max_n: int = 20) -> None:
    """Plota duration vs. prazo de vencimento."""
    prazos = range(1, max_n + 1)
    durations = [duration_macaulay(cupom_rate, face, yield_, n) for n in prazos]

    plt.figure(figsize=(9, 4))
    plt.plot(list(prazos), durations, marker="o", color="darkorange", linewidth=2)
    plt.title("Duration de Macaulay x Prazo de Vencimento", fontsize=13)
    plt.xlabel("Prazo (anos)")
    plt.ylabel("Duration (anos)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_duration.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_duration.png'")


# ---------------------------------------------------------------------------
# MAIN — demonstracao completa
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  MODULO 01 — RISCO DE TAXA DE JUROS")
    print("=" * 65)

    # Parametros do titulo ficticio
    face       = 1_000.0   # R$ 1.000
    cupom_rate = 0.10       # 10% a.a.
    yield_mer  = 0.12       # 12% a.a. (yield de mercado)
    n          = 3          # 3 anos

    # --- Precificacao ---
    preco = preco_titulo(cupom_rate, face, yield_mer, n)
    print(f"\n[1] PRECIFICACAO DO TITULO")
    print(f"    Face: R$ {face:,.2f}  |  Cupom: {cupom_rate*100:.1f}%  |  "
          f"Yield: {yield_mer*100:.1f}%  |  Prazo: {n} anos")
    print(f"    Preco de mercado: R$ {preco:,.2f}  "
          f"({'DESCONTO' if yield_mer > cupom_rate else 'PREMIO'})")

    tabela = tabela_precificacao(cupom_rate, face, n,
                                 [0.08, 0.09, 0.10, 0.11, 0.12, 0.15])
    print("\n    Tabela Preco x Yield:")
    print(tabela.to_string(index=False))

    # --- Duration ---
    d_mac = duration_macaulay(cupom_rate, face, yield_mer, n)
    d_mod = duration_modificada(cupom_rate, face, yield_mer, n)
    print(f"\n[2] DURATION")
    print(f"    Duration de Macaulay : {d_mac:.4f} anos")
    print(f"    Duration Modificada  : {d_mod:.4f}")

    delta_yield = 0.01  # +1%
    var_linear = variacao_preco_duration(d_mod, delta_yield)
    print(f"\n    Variacao do preco se yield sobe +1%:")
    print(f"    Aproximacao linear   : {var_linear * 100:.2f}%  "
          f"(R$ {preco * var_linear:,.2f})")

    # --- Convexidade ---
    conv = convexidade(cupom_rate, face, yield_mer, n)
    var_conv = variacao_preco_convexidade(d_mod, conv, delta_yield)
    print(f"\n[3] CONVEXIDADE")
    print(f"    Convexidade          : {conv:.4f}")
    print(f"    Variacao com convex. : {var_conv * 100:.2f}%  "
          f"(R$ {preco * var_conv:,.2f})")

    # --- VaR ---
    posicao       = 5_000_000.0   # R$ 5 milhoes
    sigma_yield   = 0.002         # 0.2% de volatilidade diaria da taxa
    horizonte     = 10            # 10 dias uteis
    confianca     = 0.99

    var_10d = var_renda_fixa(posicao, d_mod, sigma_yield, horizonte, confianca)
    print(f"\n[4] VaR DE POSICAO EM RENDA FIXA")
    print(f"    Posicao: R$ {posicao:,.0f}  |  D_mod: {d_mod:.2f}  |  "
          f"sigma_yield: {sigma_yield*100:.2f}%/dia")
    print(f"    VaR {int(confianca*100)}% em {horizonte} dias: R$ {var_10d:,.2f}")

    # --- GAP de Taxas ---
    print(f"\n[5] GAP DE TAXAS (ALM)")
    cenarios = [
        ("Banco Agressivo",  3_000, 1_500),
        ("Banco Conservador", 2_000, 2_500),
        ("Banco Neutro",      2_000, 2_000),
    ]
    for nome, ativos, passivos in cenarios:
        resultado = gap_taxas(ativos, passivos)
        print(f"\n    {nome}:")
        print(f"      Ativos: R$ {ativos:,} mi  |  Passivos: R$ {passivos:,} mi")
        print(f"      GAP: R$ {resultado['gap']:,} mi  |  {resultado['interpretacao']}")
        print(f"      Impacto +1% na taxa: R$ {resultado['impacto_1pct_taxa (R$)']:,} mi")

    # --- Graficos ---
    print(f"\n[6] GERANDO GRAFICOS...")
    grafico_preco_yield(cupom_rate, face, n)
    grafico_duration_vencimento(cupom_rate, face, yield_mer, max_n=15)

    print("\n" + "=" * 65)
    print("  Modulo 01 concluido com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
