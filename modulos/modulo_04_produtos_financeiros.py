"""
Modulo 04 — Produtos Financeiros
===================================
Conteudo:
  - Valor do dinheiro no tempo (VP, VF, juros compostos/simples)
  - Precificacao de titulos de renda fixa (YTM)
  - Modelo CAPM para acoes
  - Precificacao de opcoes — Black-Scholes
  - Calculadora de spread bancario
  - Comparativo de rendimentos (CDB, Tesouro, Debentures)
  - Graficos de simulacao de crescimento de capital
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq


# Diretorio de saida dos graficos
FIGURAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. VALOR DO DINHEIRO NO TEMPO
# ---------------------------------------------------------------------------

def valor_futuro(pv: float, taxa: float, n: int,
                 tipo: str = "composto") -> float:
    """
    Calcula o valor futuro de um investimento.

    Parameters
    ----------
    pv   : float - Valor presente (R$)
    taxa : float - Taxa de juros por periodo (ex: 0.10 para 10%)
    n    : int   - Numero de periodos
    tipo : str   - "composto" ou "simples"
    """
    if tipo == "composto":
        return pv * (1 + taxa) ** n
    else:
        return pv * (1 + taxa * n)


def valor_presente(fv: float, taxa: float, n: int) -> float:
    """Calcula o valor presente a partir do valor futuro."""
    return fv / (1 + taxa) ** n


def taxa_equivalente(taxa_origem: float, n_origem: int,
                     n_destino: int) -> float:
    """
    Converte taxa entre diferentes periodicidades.
    Ex: taxa anual -> taxa mensal: taxa_equivalente(0.12, 12, 1)
    """
    return (1 + taxa_origem) ** (n_destino / n_origem) - 1


def tabela_crescimento_capital(pv: float, taxa: float,
                                n_anos: int) -> pd.DataFrame:
    """Tabela de evolucao do capital ao longo dos anos."""
    dados = []
    for ano in range(n_anos + 1):
        fv_comp = valor_futuro(pv, taxa, ano, "composto")
        fv_simp = valor_futuro(pv, taxa, ano, "simples")
        dados.append({
            "Ano": ano,
            "Juros Compostos (R$)": round(fv_comp, 2),
            "Juros Simples (R$)":   round(fv_simp, 2),
            "Diferenca (R$)":       round(fv_comp - fv_simp, 2),
        })
    return pd.DataFrame(dados)


# ---------------------------------------------------------------------------
# 2. RENDA FIXA — YTM E PRECIFICACAO
# ---------------------------------------------------------------------------

def preco_titulo_rf(cupom: float, face: float, ytm: float,
                    n_periodos: int) -> float:
    """Preco de mercado dado o YTM (Yield to Maturity)."""
    fluxos = [cupom] * n_periodos
    fluxos[-1] += face
    return sum(cf / (1 + ytm) ** t for t, cf in enumerate(fluxos, start=1))


def calcular_ytm(preco: float, cupom: float, face: float,
                 n_periodos: int) -> float:
    """
    Calcula o YTM de um titulo dado o preco de mercado.
    Usa metodo numerico (Brent).
    """
    def f(ytm):
        return preco_titulo_rf(cupom, face, ytm, n_periodos) - preco

    return brentq(f, 1e-6, 10.0)


def comparar_investimentos(capital: float, prazo_anos: int,
                            taxa_selic: float, taxa_cdb: float,
                            taxa_debenture: float,
                            taxa_ipca: float, spread_ipca: float) -> pd.DataFrame:
    """
    Compara rentabilidade de diferentes instrumentos de renda fixa.
    """
    # Tesouro Selic
    tesouro_selic = valor_futuro(capital, taxa_selic, prazo_anos)

    # CDB (liquido de IR — tabela regressiva, aprox. 20% para 2+ anos)
    bruto_cdb  = valor_futuro(capital, taxa_cdb, prazo_anos)
    ir_cdb     = (bruto_cdb - capital) * 0.175  # ~17,5% IR medio
    liquido_cdb = bruto_cdb - ir_cdb

    # Tesouro IPCA+
    taxa_real_ipca = taxa_ipca + spread_ipca
    tesouro_ipca = valor_futuro(capital, taxa_real_ipca, prazo_anos)
    ir_ipca = (tesouro_ipca - capital) * 0.175
    liquido_ipca = tesouro_ipca - ir_ipca

    # LCI/LCA (isento de IR para PF)
    taxa_lci = taxa_cdb * 0.88  # LCI geralmente paga ~88% do CDI
    lci = valor_futuro(capital, taxa_lci, prazo_anos)

    # Debenture incentivada (isenta de IR)
    debenture = valor_futuro(capital, taxa_debenture, prazo_anos)

    rows = [
        {"Produto": "Tesouro Selic",       "Bruto (R$)": round(tesouro_selic, 2),  "IR (R$)": 0,                         "Liquido (R$)": round(tesouro_selic, 2),  "Obs": "Liquido (IR sobre Selic exigido no resgate)"},
        {"Produto": "CDB",                 "Bruto (R$)": round(bruto_cdb, 2),       "IR (R$)": round(ir_cdb, 2),           "Liquido (R$)": round(liquido_cdb, 2),    "Obs": "IR regressivo"},
        {"Produto": "Tesouro IPCA+",       "Bruto (R$)": round(tesouro_ipca, 2),    "IR (R$)": round(ir_ipca, 2),          "Liquido (R$)": round(liquido_ipca, 2),   "Obs": "Protege inflacao"},
        {"Produto": "LCI/LCA",             "Bruto (R$)": round(lci, 2),             "IR (R$)": 0,                         "Liquido (R$)": round(lci, 2),            "Obs": "Isento IR (PF)"},
        {"Produto": "Debenture Incentiv.", "Bruto (R$)": round(debenture, 2),        "IR (R$)": 0,                         "Liquido (R$)": round(debenture, 2),      "Obs": "Isento IR (infraestrutura)"},
    ]
    return pd.DataFrame(rows).sort_values("Liquido (R$)", ascending=False)


# ---------------------------------------------------------------------------
# 3. RENDA VARIAVEL — CAPM
# ---------------------------------------------------------------------------

def retorno_capm(rf: float, beta: float, rm: float) -> float:
    """
    Retorno esperado pelo modelo CAPM.

    E(Ri) = Rf + beta * (Rm - Rf)
    """
    return rf + beta * (rm - rf)


def security_market_line(rf: float, rm: float,
                         betas: np.ndarray) -> np.ndarray:
    """Retorna os retornos esperados ao longo da SML."""
    return rf + betas * (rm - rf)


def alpha_de_jensen(retorno_realizado: float, rf: float,
                    beta: float, rm: float) -> float:
    """
    Alpha de Jensen: excesso de retorno em relacao ao esperado pelo CAPM.
    alpha > 0: ativo gerou mais retorno que o risco implica.
    """
    retorno_esperado = retorno_capm(rf, beta, rm)
    return retorno_realizado - retorno_esperado


# ---------------------------------------------------------------------------
# 4. BLACK-SCHOLES — PRECIFICACAO DE OPCOES
# ---------------------------------------------------------------------------

def black_scholes(S: float, K: float, T: float, r: float,
                  sigma: float, tipo: str = "call") -> dict:
    """
    Precifica opcoes europeias pelo modelo Black-Scholes.

    Parameters
    ----------
    S     : float - Preco atual do ativo
    K     : float - Preco de exercicio (strike)
    T     : float - Tempo ate vencimento (anos)
    r     : float - Taxa livre de risco (continua)
    sigma : float - Volatilidade implicita do ativo
    tipo  : str   - "call" ou "put"

    Returns
    -------
    dict com preco da opcao e gregas
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == "call":
        preco = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:  # put
        preco = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    # Gregas
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2 if tipo == "call" else -d2)) / 365

    return {
        "tipo":  tipo,
        "preco": round(preco, 4),
        "d1":    round(d1, 4),
        "d2":    round(d2, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "vega":  round(vega / 100, 6),  # por 1% de volatilidade
        "theta": round(theta, 4),
    }


def paridade_put_call(call_preco: float, S: float, K: float,
                      r: float, T: float) -> float:
    """
    Calcula o preco da put pela paridade put-call.
    P = C - S + K * e^(-rT)
    """
    return call_preco - S + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# 5. SPREAD BANCARIO
# ---------------------------------------------------------------------------

def spread_bancario(taxa_emprestimo: float, custo_captacao: float,
                    inadimplencia_pct: float, custo_operacional_pct: float,
                    ir_pct: float) -> dict:
    """
    Decomposicao do spread bancario.

    Spread = Taxa de emprestimo - Custo de captacao
    """
    spread_total = taxa_emprestimo - custo_captacao
    componentes = {
        "Inadimplencia (%)":       round(inadimplencia_pct * 100, 2),
        "Custo operacional (%)":   round(custo_operacional_pct * 100, 2),
        "IR/CSLL (%)":             round(ir_pct * 100, 2),
        "Margem liquida (%)":      round((spread_total - inadimplencia_pct
                                         - custo_operacional_pct - ir_pct) * 100, 2),
    }
    return {
        "Taxa de emprestimo (%)":  round(taxa_emprestimo * 100, 2),
        "Custo de captacao (%)":   round(custo_captacao * 100, 2),
        "Spread bruto (%)":        round(spread_total * 100, 2),
        "Componentes": componentes,
    }


# ---------------------------------------------------------------------------
# 6. GRAFICOS
# ---------------------------------------------------------------------------

def grafico_juros_compostos(pv: float, taxa: float, n: int) -> None:
    """Plota crescimento com juros compostos vs. simples."""
    anos = list(range(n + 1))
    comp = [valor_futuro(pv, taxa, a, "composto") for a in anos]
    simp = [valor_futuro(pv, taxa, a, "simples")  for a in anos]

    plt.figure(figsize=(10, 5))
    plt.plot(anos, comp, marker="o", color="steelblue", label="Juros Compostos", linewidth=2)
    plt.plot(anos, simp, marker="s", color="orange",   label="Juros Simples",    linewidth=2, linestyle="--")
    plt.fill_between(anos, simp, comp, alpha=0.15, color="steelblue", label="Diferenca (poder dos juros compostos)")
    plt.title(f"Crescimento de R$ {pv:,.0f} a {taxa*100:.1f}% ao ano — {n} anos", fontsize=13)
    plt.xlabel("Anos")
    plt.ylabel("Valor (R$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "grafico_juros_compostos.png"), dpi=120)
    plt.show()
    print(f"Grafico salvo em: {os.path.join(FIGURAS_DIR, "grafico_juros_compostos.png")}")


def grafico_sml(rf: float, rm: float, ativos: list) -> None:
    """
    Plota a Security Market Line (SML) com ativos sobrevalorizados e subvalorizados.
    """
    betas = np.linspace(0, 2.5, 200)
    retornos_sml = security_market_line(rf, rm, betas)

    plt.figure(figsize=(10, 5))
    plt.plot(betas, retornos_sml * 100, color="steelblue",
             linewidth=2.5, label="SML (CAPM)")
    plt.axhline(rf * 100, color="gray", linestyle="--", alpha=0.5, label=f"Rf = {rf*100:.1f}%")

    for ativo in ativos:
        retorno_esp = retorno_capm(rf, ativo["beta"], rm)
        alpha = ativo["retorno_real"] - retorno_esp
        cor = "#27ae60" if alpha > 0 else "#e74c3c"
        label = f"{ativo['nome']} (alpha={alpha*100:+.1f}%)"
        plt.scatter(ativo["beta"], ativo["retorno_real"] * 100,
                    color=cor, s=100, zorder=5)
        plt.annotate(label, (ativo["beta"], ativo["retorno_real"] * 100),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)

    plt.title("Security Market Line — CAPM", fontsize=13)
    plt.xlabel("Beta")
    plt.ylabel("Retorno Esperado (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "grafico_sml.png"), dpi=120)
    plt.show()
    print(f"Grafico salvo em: {os.path.join(FIGURAS_DIR, "grafico_sml.png")}")


def grafico_black_scholes_superficie(S: float, K: float, r: float,
                                     sigma: float) -> None:
    """Plota o preco da call em funcao do tempo e volatilidade."""
    Ts     = np.linspace(0.05, 2.0, 60)
    sigmas = np.linspace(0.10, 0.80, 60)
    TT, SS = np.meshgrid(Ts, sigmas)

    precos = np.array([
        [black_scholes(S, K, t, r, s)["preco"] for t in Ts]
        for s in sigmas
    ])

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(TT, SS, precos, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Tempo (anos)")
    ax.set_ylabel("Volatilidade")
    ax.set_zlabel("Preco da Call (R$)")
    ax.set_title("Superficie Black-Scholes — Preco da Call", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "grafico_bs_superficie.png"), dpi=120)
    plt.show()
    print(f"Grafico salvo em: {os.path.join(FIGURAS_DIR, "grafico_bs_superficie.png")}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  MODULO 04 — PRODUTOS FINANCEIROS")
    print("=" * 65)

    # --- Valor do Dinheiro no Tempo ---
    print("\n[1] VALOR DO DINHEIRO NO TEMPO")
    pv, taxa, n = 1_000.0, 0.10, 5
    fv_comp = valor_futuro(pv, taxa, n, "composto")
    fv_simp = valor_futuro(pv, taxa, n, "simples")
    vp      = valor_presente(fv_comp, taxa, n)
    print(f"    PV = R$ {pv:,.2f}  |  Taxa = {taxa*100:.1f}% a.a.  |  n = {n} anos")
    print(f"    VF (compostos): R$ {fv_comp:,.2f}")
    print(f"    VF (simples)  : R$ {fv_simp:,.2f}")
    print(f"    VP de R$ {fv_comp:,.2f} daqui {n} anos: R$ {vp:,.2f}")

    taxa_mens = taxa_equivalente(taxa, 12, 1)
    print(f"\n    Taxa anual {taxa*100:.1f}% = taxa mensal equivalente: {taxa_mens*100:.4f}%")

    print("\n    Tabela de evolucao (5 anos):")
    tabela = tabela_crescimento_capital(pv, taxa, n)
    print(tabela.to_string(index=False))

    # --- Renda Fixa ---
    print(f"\n[2] RENDA FIXA — YTM E COMPARATIVO")
    cupom_anual = 100.0  # R$100 de cupom por ano
    face        = 1000.0
    ytm_mer     = 0.12
    preco_rf    = preco_titulo_rf(cupom_anual, face, ytm_mer, 3)
    ytm_calc    = calcular_ytm(preco_rf, cupom_anual, face, 3)
    print(f"    Titulo: cupom R$ {cupom_anual} / face R$ {face} / 3 anos")
    print(f"    Preco a YTM={ytm_mer*100:.1f}%: R$ {preco_rf:,.2f}")
    print(f"    YTM calculado a partir do preco: {ytm_calc*100:.4f}%")

    print("\n    Comparativo de Investimentos (R$ 10.000 / 3 anos):")
    df_inv = comparar_investimentos(
        capital=10_000, prazo_anos=3,
        taxa_selic=0.107, taxa_cdb=0.108,
        taxa_debenture=0.122,
        taxa_ipca=0.045, spread_ipca=0.065
    )
    print(df_inv.to_string(index=False))

    # --- CAPM ---
    print(f"\n[3] MODELO CAPM")
    rf, rm = 0.105, 0.155  # Selic = 10,5%; Ibovespa = 15,5%
    ativos = [
        {"nome": "PETR4",  "beta": 1.20, "retorno_real": 0.185},
        {"nome": "VALE3",  "beta": 1.05, "retorno_real": 0.160},
        {"nome": "ITUB4",  "beta": 0.75, "retorno_real": 0.130},
        {"nome": "WEGE3",  "beta": 0.60, "retorno_real": 0.145},
        {"nome": "RENT3",  "beta": 0.90, "retorno_real": 0.135},
    ]
    print(f"    Rf = {rf*100:.1f}%  |  Rm = {rm*100:.1f}%  |  Premio = {(rm-rf)*100:.1f}%")
    print()
    for a in ativos:
        exp = retorno_capm(rf, a["beta"], rm)
        alpha = alpha_de_jensen(a["retorno_real"], rf, a["beta"], rm)
        avaliacao = "SUBVALORIZADO" if alpha > 0 else "SOBREVALORIZADO"
        print(f"    {a['nome']:6s}  beta={a['beta']:.2f}  "
              f"E(R)={exp*100:.2f}%  real={a['retorno_real']*100:.2f}%  "
              f"alpha={alpha*100:+.2f}%  -> {avaliacao}")

    # --- Black-Scholes ---
    print(f"\n[4] BLACK-SCHOLES — PRECIFICACAO DE OPCOES")
    S, K, T, r, sigma = 50.0, 52.0, 0.50, 0.12, 0.30
    call = black_scholes(S, K, T, r, sigma, "call")
    put  = black_scholes(S, K, T, r, sigma, "put")
    print(f"    Ativo: R$ {S}  |  Strike: R$ {K}  |  T = {T*12:.0f} meses  "
          f"|  r = {r*100:.1f}%  |  sigma = {sigma*100:.0f}%")
    print(f"\n    CALL: preco=R$ {call['preco']:.4f}  |  delta={call['delta']:.4f}  "
          f"|  gamma={call['gamma']:.6f}  |  vega={call['vega']:.6f}  |  theta=R$ {call['theta']:.4f}/dia")
    print(f"    PUT : preco=R$ {put['preco']:.4f}  |  delta={put['delta']:.4f}  "
          f"|  gamma={put['gamma']:.6f}  |  vega={put['vega']:.6f}  |  theta=R$ {put['theta']:.4f}/dia")

    put_paridade = paridade_put_call(call["preco"], S, K, r, T)
    print(f"\n    Paridade Put-Call: PUT = R$ {put_paridade:.4f}  "
          f"(BS direto: R$ {put['preco']:.4f}) -- devem ser iguais")

    # --- Spread bancario ---
    print(f"\n[5] SPREAD BANCARIO")
    spread = spread_bancario(
        taxa_emprestimo=0.32,
        custo_captacao=0.11,
        inadimplencia_pct=0.08,
        custo_operacional_pct=0.04,
        ir_pct=0.03
    )
    print(f"    Taxa de emprestimo: {spread['Taxa de emprestimo (%)']:.2f}%")
    print(f"    Custo de captacao : {spread['Custo de captacao (%)']:.2f}%")
    print(f"    Spread bruto      : {spread['Spread bruto (%)']:.2f}%")
    print(f"    Decomposicao:")
    for k, v in spread["Componentes"].items():
        print(f"      {k}: {v:.2f}%")

    # --- Graficos ---
    print(f"\n[6] GERANDO GRAFICOS...")
    grafico_juros_compostos(1_000, 0.10, 20)
    grafico_sml(rf, rm, ativos)
    grafico_black_scholes_superficie(50, 52, 0.12, 0.30)

    print("\n" + "=" * 65)
    print("  Modulo 04 concluido com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
