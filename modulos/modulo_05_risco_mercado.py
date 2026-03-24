"""
Modulo 05 — Risco de Mercado
================================
Conteudo:
  - VaR Parametrico (distribuicao normal)
  - VaR Historico (percentil empirico)
  - VaR Monte Carlo (simulacao)
  - Expected Shortfall (CVaR / ES)
  - Backtesting de VaR (teste de Kupiec)
  - DV01 — sensibilidade a 1 basis point
  - Graficos de distribuicao de retornos e comparacao de metodos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# 1. VaR PARAMETRICO
# ---------------------------------------------------------------------------

def var_parametrico(posicao: float, sigma: float, horizonte: int = 1,
                    nivel_confianca: float = 0.99,
                    distribuicao: str = "normal",
                    df_t: float = 5.0) -> float:
    """
    VaR parametrico para distribuicao normal ou t-Student.

    VaR_normal = Z * sigma * sqrt(t) * posicao
    VaR_t      = t_quantil(df) * sigma * sqrt(t) * posicao

    Parameters
    ----------
    posicao         : float  - Valor da posicao (R$)
    sigma           : float  - Volatilidade diaria dos retornos
    horizonte       : int    - Horizonte em dias
    nivel_confianca : float  - Ex: 0.99
    distribuicao    : str    - "normal" ou "t"
    df_t            : float  - Graus de liberdade (para dist. t)
    """
    alpha = 1 - nivel_confianca
    if distribuicao == "normal":
        z = norm.ppf(nivel_confianca)
    else:
        z = t_dist.ppf(nivel_confianca, df=df_t) * np.sqrt((df_t - 2) / df_t)

    return z * sigma * np.sqrt(horizonte) * posicao


def var_scaling_rule(var_1d: float, horizonte: int) -> float:
    """
    Extrapola VaR de 1 dia para horizonte maior pela regra da raiz do tempo.
    Valida apenas para retornos i.i.d.
    """
    return var_1d * np.sqrt(horizonte)


# ---------------------------------------------------------------------------
# 2. VaR HISTORICO
# ---------------------------------------------------------------------------

def simular_retornos_historicos(n_dias: int = 500,
                                 mu: float = 0.0005,
                                 sigma: float = 0.018,
                                 seed: int = 42) -> np.ndarray:
    """Gera serie historica de retornos diarios (dados ficticios)."""
    rng = np.random.default_rng(seed)
    retornos = rng.normal(mu, sigma, n_dias)
    # Adiciona alguns eventos de cauda
    n_choques = int(n_dias * 0.02)
    idx_choques = rng.choice(n_dias, n_choques, replace=False)
    retornos[idx_choques] *= rng.choice([-3, -4, -5], n_choques)
    return retornos


def var_historico(retornos: np.ndarray, posicao: float,
                  nivel_confianca: float = 0.99) -> float:
    """
    VaR Historico: percentil empirico da distribuicao de retornos.
    """
    alpha = 1 - nivel_confianca
    percentil = np.percentile(retornos, alpha * 100)
    return -percentil * posicao  # VaR e positivo (perda)


# ---------------------------------------------------------------------------
# 3. VaR MONTE CARLO
# ---------------------------------------------------------------------------

def var_monte_carlo(posicao: float, mu: float, sigma: float,
                    horizonte: int = 1, nivel_confianca: float = 0.99,
                    n_simulacoes: int = 100_000,
                    seed: int = 42) -> dict:
    """
    VaR por simulacao de Monte Carlo.

    Simula N cenarios de retorno e calcula o percentil correspondente.
    """
    rng = np.random.default_rng(seed)
    retornos_sim = rng.normal(mu * horizonte,
                               sigma * np.sqrt(horizonte),
                               n_simulacoes)
    pls = retornos_sim * posicao
    alpha = 1 - nivel_confianca
    var = -np.percentile(pls, alpha * 100)
    es  = -pls[pls <= -var].mean()

    return {
        "VaR": round(var, 2),
        "ES (CVaR)": round(es, 2),
        "retornos_simulados": retornos_sim,
        "pls": pls,
    }


# ---------------------------------------------------------------------------
# 4. EXPECTED SHORTFALL (CVaR)
# ---------------------------------------------------------------------------

def expected_shortfall(retornos: np.ndarray, posicao: float,
                        nivel_confianca: float = 0.99) -> float:
    """
    Expected Shortfall (CVaR): media das perdas alem do VaR.
    ES = E[perda | perda > VaR]
    """
    alpha = 1 - nivel_confianca
    threshold = np.percentile(retornos, alpha * 100)
    tail_losses = retornos[retornos <= threshold]
    return -tail_losses.mean() * posicao


# ---------------------------------------------------------------------------
# 5. BACKTESTING — TESTE DE KUPIEC
# ---------------------------------------------------------------------------

def backtesting_var(retornos: np.ndarray, posicao: float,
                     var_diario: float, nivel_confianca: float = 0.99) -> dict:
    """
    Backtest de VaR pelo metodo de Kupiec (1995).

    Calcula taxa de excecoes e testa se e compativel com o nivel de confianca.

    H0: taxa de excecoes = alpha (modelo correto)
    Ha: taxa de excecoes != alpha
    """
    alpha_esperado = 1 - nivel_confianca
    n = len(retornos)
    perdas = -retornos * posicao

    excecoes = (perdas > var_diario).sum()
    taxa_excecoes = excecoes / n

    # Estatistica LR de Kupiec
    p_hat = taxa_excecoes if taxa_excecoes > 0 else 1e-10
    p_0   = alpha_esperado

    # Evitar log(0)
    if p_hat <= 0 or p_hat >= 1:
        lr_stat = np.nan
        p_valor  = np.nan
    else:
        lr_stat = -2 * (
            n * np.log(1 - p_0) + excecoes * np.log(p_0) -
            (n - excecoes) * np.log(1 - p_hat) - excecoes * np.log(p_hat)
        )
        p_valor = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "n_dias": n,
        "var_diario": round(var_diario, 2),
        "excecoes": int(excecoes),
        "taxa_excecoes (%)": round(taxa_excecoes * 100, 2),
        "alpha_esperado (%)": round(alpha_esperado * 100, 2),
        "LR_stat": round(lr_stat, 4) if not np.isnan(lr_stat) else "N/A",
        "p_valor": round(p_valor, 4) if not np.isnan(p_valor) else "N/A",
        "resultado": (
            "APROVADO (modelo adequado)"
            if (isinstance(p_valor, float) and p_valor > 0.05)
            else "REPROVADO (modelo inadequado)"
        ),
    }


# ---------------------------------------------------------------------------
# 6. DV01 — SENSIBILIDADE A 1 BASIS POINT
# ---------------------------------------------------------------------------

def dv01(posicao: float, d_mod: float) -> float:
    """
    DV01 (Dollar Value of 1 Basis Point): variacao em R$ para +1bp de taxa.

    DV01 = posicao * D_mod * 0.0001
    """
    return posicao * d_mod * 0.0001


# ---------------------------------------------------------------------------
# 7. COMPARACAO DE METODOS VaR
# ---------------------------------------------------------------------------

def comparar_metodos_var(posicao: float, retornos: np.ndarray,
                          sigma: float, mu: float,
                          nivel_confianca: float = 0.99) -> pd.DataFrame:
    """Compara os tres metodos de VaR para o mesmo ativo."""
    v_param  = var_parametrico(posicao, sigma, 1, nivel_confianca)
    v_hist   = var_historico(retornos, posicao, nivel_confianca)
    mc       = var_monte_carlo(posicao, mu, sigma, 1, nivel_confianca)
    v_mc     = mc["VaR"]
    es_hist  = expected_shortfall(retornos, posicao, nivel_confianca)

    dados = [
        {"Metodo": "VaR Parametrico (Normal)",  "VaR (R$)": round(v_param, 2), "Pressupostos": "Retornos normais"},
        {"Metodo": "VaR Historico",              "VaR (R$)": round(v_hist, 2),  "Pressupostos": "Distribuicao empirica"},
        {"Metodo": "VaR Monte Carlo",            "VaR (R$)": round(v_mc, 2),    "Pressupostos": "Simulacao normal"},
        {"Metodo": "ES/CVaR (Historico)",        "VaR (R$)": round(es_hist, 2), "Pressupostos": "Media das perdas na cauda"},
    ]
    return pd.DataFrame(dados)


# ---------------------------------------------------------------------------
# 8. GRAFICOS
# ---------------------------------------------------------------------------

def grafico_distribuicao_retornos(retornos: np.ndarray, posicao: float,
                                   nivel_confianca: float = 0.99) -> None:
    """Histograma dos retornos com VaR marcado."""
    v_hist = var_historico(retornos, posicao, nivel_confianca) / posicao
    v_param = -norm.ppf(1 - nivel_confianca) * retornos.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Retornos
    axes[0].hist(retornos * 100, bins=60, density=True,
                 color="steelblue", edgecolor="white", alpha=0.7, label="Retornos (%)")
    x_range = np.linspace(retornos.min() * 100, retornos.max() * 100, 300)
    axes[0].plot(x_range, norm.pdf(x_range, retornos.mean()*100, retornos.std()*100),
                 "r-", linewidth=2, label="Normal ajustada")
    axes[0].axvline(-v_hist * 100, color="red", linewidth=2.5,
                    linestyle="--", label=f"VaR Hist {int(nivel_confianca*100)}%")
    axes[0].axvline(-v_param * 100, color="orange", linewidth=2,
                    linestyle="-.", label=f"VaR Param {int(nivel_confianca*100)}%")
    axes[0].set_title("Distribuicao de Retornos Diarios", fontsize=12)
    axes[0].set_xlabel("Retorno (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Serie historica
    axes[1].plot(retornos * 100, color="steelblue", linewidth=0.8, alpha=0.8)
    axes[1].axhline(-v_hist * 100, color="red", linewidth=1.5, linestyle="--",
                    label=f"VaR Hist {int(nivel_confianca*100)}%")
    perdas_extremas = retornos < -v_hist
    axes[1].scatter(np.where(perdas_extremas)[0],
                    retornos[perdas_extremas] * 100,
                    color="red", s=25, zorder=5, label="Excecoes ao VaR")
    axes[1].set_title("Serie de Retornos — Excecoes ao VaR", fontsize=12)
    axes[1].set_xlabel("Dias")
    axes[1].set_ylabel("Retorno (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("grafico_var_retornos.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_var_retornos.png'")


def grafico_monte_carlo_var(mc_result: dict, posicao: float,
                             nivel_confianca: float = 0.99) -> None:
    """Histograma das simulacoes Monte Carlo com VaR e ES."""
    pls = mc_result["pls"]
    var = mc_result["VaR"]
    es  = mc_result["ES (CVaR)"]

    plt.figure(figsize=(10, 5))
    plt.hist(pls / 1000, bins=100, density=True,
             color="steelblue", edgecolor="white", alpha=0.7)
    plt.axvline(-var / 1000, color="red", linewidth=2.5, linestyle="--",
                label=f"VaR {int(nivel_confianca*100)}% = R$ {var:,.0f}")
    plt.axvline(-es / 1000, color="darkred", linewidth=2, linestyle="-.",
                label=f"ES/CVaR = R$ {es:,.0f}")
    plt.fill_betweenx([0, 0.0001], -es / 1000, plt.xlim()[0],
                      alpha=0.1, color="red")
    plt.title(f"Simulacao Monte Carlo — Distribuicao de P&L\n"
              f"(Posicao: R$ {posicao:,.0f}  |  {len(pls):,} simulacoes)", fontsize=12)
    plt.xlabel("P&L (R$ mil)")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_monte_carlo.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_monte_carlo.png'")


def grafico_backtesting(retornos: np.ndarray, posicao: float,
                         var_diario: float, nivel_confianca: float = 0.99) -> None:
    """Plota o backtest do VaR ao longo do tempo."""
    perdas = -retornos * posicao / 1000  # em mil R$
    var_mil = var_diario / 1000
    excecoes = perdas > var_mil

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(perdas, color="steelblue", linewidth=0.8, alpha=0.8, label="Perda diaria (R$ mil)")
    ax.axhline(var_mil, color="red", linewidth=2, linestyle="--",
               label=f"VaR {int(nivel_confianca*100)}% = R$ {var_diario/1000:,.0f} mil")
    ax.scatter(np.where(excecoes)[0], perdas[excecoes],
               color="red", s=40, zorder=5,
               label=f"Excecoes ({excecoes.sum()} dias = {excecoes.mean()*100:.1f}%)")
    ax.set_title("Backtest de VaR — Perdas Realizadas vs. VaR", fontsize=12)
    ax.set_xlabel("Dias")
    ax.set_ylabel("Perda (R$ mil)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_backtesting.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_backtesting.png'")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  MODULO 05 — RISCO DE MERCADO")
    print("=" * 65)

    POSICAO = 10_000_000.0   # R$ 10 milhoes
    SIGMA   = 0.018          # 1,8% de volatilidade diaria
    MU      = 0.0003         # Retorno medio diario esperado
    NC_99   = 0.99
    NC_95   = 0.95

    # --- Retornos historicos ---
    retornos = simular_retornos_historicos(n_dias=500, mu=MU, sigma=SIGMA)
    print(f"\n    Retornos historicos simulados: {len(retornos)} dias")
    print(f"    Mu diario: {retornos.mean()*100:.4f}%  |  "
          f"Sigma diario: {retornos.std()*100:.4f}%")

    # --- VaR Parametrico ---
    print(f"\n[1] VaR PARAMETRICO")
    for nc in [0.95, 0.99]:
        v_1d  = var_parametrico(POSICAO, SIGMA, 1, nc)
        v_10d = var_parametrico(POSICAO, SIGMA, 10, nc)
        print(f"    NC={int(nc*100)}% | VaR 1d = R$ {v_1d:,.0f} | "
              f"VaR 10d = R$ {v_10d:,.0f}")

    v_t = var_parametrico(POSICAO, SIGMA, 1, NC_99, distribuicao="t", df_t=5)
    v_n = var_parametrico(POSICAO, SIGMA, 1, NC_99)
    print(f"\n    VaR 99% Normal     : R$ {v_n:,.0f}")
    print(f"    VaR 99% t-Student  : R$ {v_t:,.0f}  (cauda mais pesada)")

    # --- VaR Historico ---
    print(f"\n[2] VaR HISTORICO")
    v_hist_99 = var_historico(retornos, POSICAO, NC_99)
    v_hist_95 = var_historico(retornos, POSICAO, NC_95)
    print(f"    VaR Hist 99% = R$ {v_hist_99:,.0f}")
    print(f"    VaR Hist 95% = R$ {v_hist_95:,.0f}")

    # --- Monte Carlo ---
    print(f"\n[3] VaR MONTE CARLO (100.000 simulacoes)")
    mc = var_monte_carlo(POSICAO, MU, SIGMA, 1, NC_99, n_simulacoes=100_000)
    print(f"    VaR MC 99%  = R$ {mc['VaR']:,.0f}")
    print(f"    ES/CVaR 99% = R$ {mc['ES (CVaR)']:,.0f}")

    # --- Comparacao ---
    print(f"\n[4] COMPARACAO DOS METODOS")
    df_comp = comparar_metodos_var(POSICAO, retornos, SIGMA, MU, NC_99)
    print(df_comp.to_string(index=False))

    # --- Expected Shortfall ---
    print(f"\n[5] EXPECTED SHORTFALL")
    es_99 = expected_shortfall(retornos, POSICAO, NC_99)
    es_95 = expected_shortfall(retornos, POSICAO, NC_95)
    print(f"    ES 99% = R$ {es_99:,.0f}  (vs VaR Hist 99% = R$ {v_hist_99:,.0f})")
    print(f"    ES 95% = R$ {es_95:,.0f}  (vs VaR Hist 95% = R$ {v_hist_95:,.0f})")
    print(f"    ES/VaR ratio: {es_99/v_hist_99:.2f}x  (ES sempre >= VaR)")

    # --- Backtesting ---
    print(f"\n[6] BACKTESTING DE VaR")
    var_diario = var_parametrico(POSICAO, SIGMA, 1, NC_99)
    bt = backtesting_var(retornos, POSICAO, var_diario, NC_99)
    for k, v in bt.items():
        print(f"    {k}: {v}")

    # --- DV01 ---
    print(f"\n[7] DV01")
    d_mod = 4.2
    dv = dv01(POSICAO, d_mod)
    print(f"    Posicao: R$ {POSICAO:,.0f}  |  Duration Mod: {d_mod}")
    print(f"    DV01 = R$ {dv:,.0f}  (perda para +1bp na taxa)")

    # --- Graficos ---
    print(f"\n[8] GERANDO GRAFICOS...")
    grafico_distribuicao_retornos(retornos, POSICAO, NC_99)
    grafico_monte_carlo_var(mc, POSICAO, NC_99)
    grafico_backtesting(retornos, POSICAO, var_diario, NC_99)

    print("\n" + "=" * 65)
    print("  Modulo 05 concluido com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
