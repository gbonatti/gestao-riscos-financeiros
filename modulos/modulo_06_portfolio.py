"""
Modulo 06 — Gestao de Portfolio
==================================
Conteudo:
  - Retorno e risco de carteiras com multiplos ativos
  - Fronteira eficiente de Markowitz
  - Carteira de minima variancia e carteira otima
  - Sharpe Ratio, Sortino Ratio, Treynor
  - Simulacao de portfolios aleatorios (Monte Carlo)
  - Analise de correlacao e diversificacao
  - Graficos da fronteira eficiente
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from scipy.stats import norm


# Diretorio de saida dos graficos
FIGURAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DADOS FICTICIOS DE ATIVOS BRASILEIROS
# ---------------------------------------------------------------------------

ATIVOS = {
    "PETR4": {"retorno_anual": 0.18, "vol_anual": 0.38, "beta": 1.20},
    "VALE3": {"retorno_anual": 0.15, "vol_anual": 0.35, "beta": 1.05},
    "ITUB4": {"retorno_anual": 0.14, "vol_anual": 0.28, "beta": 0.75},
    "WEGE3": {"retorno_anual": 0.20, "vol_anual": 0.30, "beta": 0.60},
    "ABEV3": {"retorno_anual": 0.10, "vol_anual": 0.22, "beta": 0.50},
    "RENT3": {"retorno_anual": 0.16, "vol_anual": 0.32, "beta": 0.90},
}

# Matriz de correlacao ficticia
CORRELACOES = np.array([
    [1.00, 0.55, 0.40, 0.25, 0.20, 0.35],  # PETR4
    [0.55, 1.00, 0.35, 0.20, 0.15, 0.30],  # VALE3
    [0.40, 0.35, 1.00, 0.45, 0.50, 0.60],  # ITUB4
    [0.25, 0.20, 0.45, 1.00, 0.35, 0.40],  # WEGE3
    [0.20, 0.15, 0.50, 0.35, 1.00, 0.30],  # ABEV3
    [0.35, 0.30, 0.60, 0.40, 0.30, 1.00],  # RENT3
])


def montar_parametros(ativos: dict,
                       correlacoes: np.ndarray) -> tuple:
    """
    Monta vetores de retorno e matriz de covariancia.

    Returns
    -------
    (nomes, retornos, vols, cov_matrix)
    """
    nomes    = list(ativos.keys())
    retornos = np.array([v["retorno_anual"] for v in ativos.values()])
    vols     = np.array([v["vol_anual"]     for v in ativos.values()])
    cov      = np.outer(vols, vols) * correlacoes
    return nomes, retornos, vols, cov


# ---------------------------------------------------------------------------
# 2. METRICAS DE CARTEIRA
# ---------------------------------------------------------------------------

def retorno_carteira(pesos: np.ndarray, retornos: np.ndarray) -> float:
    """E(Rp) = sum(wi * Ri)"""
    return float(pesos @ retornos)


def risco_carteira(pesos: np.ndarray, cov: np.ndarray) -> float:
    """sigma_p = sqrt(w^T * SIGMA * w)"""
    return float(np.sqrt(pesos @ cov @ pesos))


def sharpe_ratio(pesos: np.ndarray, retornos: np.ndarray,
                  cov: np.ndarray, rf: float) -> float:
    """Sharpe = (E(Rp) - Rf) / sigma_p"""
    rp = retorno_carteira(pesos, retornos)
    sp = risco_carteira(pesos, cov)
    return (rp - rf) / sp if sp > 0 else 0.0


def sortino_ratio(pesos: np.ndarray, retornos_historicos: np.ndarray,
                   pesos_ativos: np.ndarray, rf: float) -> float:
    """
    Sortino = (E(Rp) - Rf) / sigma_downside
    Usa apenas a volatilidade dos retornos negativos.
    """
    retornos_port = retornos_historicos @ pesos_ativos
    rp = retornos_port.mean() * 252
    downside = retornos_port[retornos_port < 0].std() * np.sqrt(252)
    return (rp - rf) / downside if downside > 0 else 0.0


def treynor_ratio(rp: float, rf: float, beta: float) -> float:
    """Treynor = (E(Rp) - Rf) / beta"""
    return (rp - rf) / beta if beta != 0 else 0.0


# ---------------------------------------------------------------------------
# 3. OTIMIZACAO DE PORTFOLIO (MARKOWITZ)
# ---------------------------------------------------------------------------

def carteira_minima_variancia(retornos: np.ndarray, cov: np.ndarray,
                               allow_short: bool = False) -> dict:
    """
    Encontra a carteira de minima variancia global.
    """
    n = len(retornos)
    pesos_ini = np.ones(n) / n

    restricoes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n

    resultado = minimize(
        fun=lambda w: risco_carteira(w, cov),
        x0=pesos_ini,
        method="SLSQP",
        bounds=bounds,
        constraints=restricoes,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    pesos_otimos = resultado.x
    return {
        "pesos": pesos_otimos,
        "retorno": retorno_carteira(pesos_otimos, retornos),
        "risco":   risco_carteira(pesos_otimos, cov),
        "convergiu": resultado.success,
    }


def carteira_maximo_sharpe(retornos: np.ndarray, cov: np.ndarray,
                            rf: float,
                            allow_short: bool = False) -> dict:
    """
    Encontra a carteira com maximo Sharpe Ratio (carteira tangente).
    """
    n = len(retornos)
    pesos_ini = np.ones(n) / n

    restricoes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n

    resultado = minimize(
        fun=lambda w: -sharpe_ratio(w, retornos, cov, rf),
        x0=pesos_ini,
        method="SLSQP",
        bounds=bounds,
        constraints=restricoes,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    pesos_otimos = resultado.x
    return {
        "pesos": pesos_otimos,
        "retorno": retorno_carteira(pesos_otimos, retornos),
        "risco":   risco_carteira(pesos_otimos, cov),
        "sharpe":  sharpe_ratio(pesos_otimos, retornos, cov, rf),
        "convergiu": resultado.success,
    }


def fronteira_eficiente(retornos: np.ndarray, cov: np.ndarray,
                         n_pontos: int = 80,
                         allow_short: bool = False) -> pd.DataFrame:
    """
    Calcula a fronteira eficiente de Markowitz.

    Minimiza a variancia para cada nivel de retorno alvo.
    """
    n   = len(retornos)
    ret_min = carteira_minima_variancia(retornos, cov, allow_short)["retorno"]
    ret_max = retornos.max()
    alvos   = np.linspace(ret_min, ret_max, n_pontos)

    pontos = []
    for alvo in alvos:
        restricoes = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, a=alvo: retorno_carteira(w, retornos) - a},
        ]
        bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n

        res = minimize(
            fun=lambda w: risco_carteira(w, cov),
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=restricoes,
            options={"ftol": 1e-12},
        )
        if res.success:
            pontos.append({
                "retorno": alvo,
                "risco":   risco_carteira(res.x, cov),
                "pesos":   res.x,
            })

    return pd.DataFrame(pontos)


# ---------------------------------------------------------------------------
# 4. SIMULACAO MONTE CARLO DE PORTFOLIOS
# ---------------------------------------------------------------------------

def simular_portfolios_aleatorios(retornos: np.ndarray, cov: np.ndarray,
                                   rf: float, n_portfolios: int = 5_000,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Gera portfolios com pesos aleatorios e calcula Sharpe de cada um.
    Util para visualizar o espaco risco-retorno.
    """
    rng = np.random.default_rng(seed)
    n   = len(retornos)
    resultados = []

    for _ in range(n_portfolios):
        pesos = rng.random(n)
        pesos /= pesos.sum()
        rp = retorno_carteira(pesos, retornos)
        sp = risco_carteira(pesos, cov)
        sh = (rp - rf) / sp
        resultados.append({"retorno": rp, "risco": sp, "sharpe": sh})

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# 5. ANALISE DE CORRELACAO
# ---------------------------------------------------------------------------

def analise_correlacao(nomes: list, correlacoes: np.ndarray) -> pd.DataFrame:
    """Retorna a matriz de correlacao formatada."""
    return pd.DataFrame(correlacoes, index=nomes, columns=nomes)


def beneficio_diversificacao(pesos: np.ndarray, vols: np.ndarray,
                              cov: np.ndarray) -> dict:
    """
    Calcula o beneficio da diversificacao comparando a volatilidade
    ponderada vs. a volatilidade real da carteira.
    """
    vol_ponderada = float(pesos @ vols)   # sem diversificacao
    vol_real      = risco_carteira(pesos, cov)
    reducao       = vol_ponderada - vol_real
    return {
        "volatilidade_ponderada": round(vol_ponderada, 4),
        "volatilidade_real":      round(vol_real, 4),
        "reducao_absoluta":       round(reducao, 4),
        "reducao_percentual":     round(reducao / vol_ponderada * 100, 2),
    }


# ---------------------------------------------------------------------------
# 6. GRAFICOS
# ---------------------------------------------------------------------------

def grafico_fronteira_eficiente(df_fronteira: pd.DataFrame,
                                 df_sim: pd.DataFrame,
                                 cart_min_var: dict,
                                 cart_max_sharpe: dict,
                                 ativos_ind: tuple,
                                 rf: float) -> None:
    """Plota a fronteira eficiente com portfolios simulados."""
    nomes_at, retornos_at, vols_at = ativos_ind

    fig, ax = plt.subplots(figsize=(11, 7))

    # Portfolios aleatorios
    sc = ax.scatter(df_sim["risco"] * 100, df_sim["retorno"] * 100,
                    c=df_sim["sharpe"], cmap="viridis",
                    alpha=0.35, s=12, label="Portfolios aleatorios")
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Fronteira eficiente
    ax.plot(df_fronteira["risco"] * 100, df_fronteira["retorno"] * 100,
            "b-", linewidth=3, label="Fronteira Eficiente")

    # Carteira de minima variancia
    ax.scatter(cart_min_var["risco"] * 100, cart_min_var["retorno"] * 100,
               color="green", s=200, zorder=10, marker="*",
               label=f"Min. Variancia (risco={cart_min_var['risco']*100:.1f}%)")

    # Carteira otima (max Sharpe)
    ax.scatter(cart_max_sharpe["risco"] * 100, cart_max_sharpe["retorno"] * 100,
               color="red", s=200, zorder=10, marker="*",
               label=f"Max. Sharpe = {cart_max_sharpe['sharpe']:.2f}")

    # Ativos individuais
    for nome, r, v in zip(nomes_at, retornos_at, vols_at):
        ax.scatter(v * 100, r * 100, color="black", s=70, zorder=8)
        ax.annotate(nome, (v * 100, r * 100), textcoords="offset points",
                    xytext=(6, 3), fontsize=9, color="black")

    # Capital Market Line (CML)
    riscos_cml = np.linspace(0, df_fronteira["risco"].max() * 1.2, 100)
    sharpe_max = cart_max_sharpe["sharpe"]
    retornos_cml = rf + sharpe_max * riscos_cml
    ax.plot(riscos_cml * 100, retornos_cml * 100, "r--",
            linewidth=1.5, alpha=0.7, label="Capital Market Line")

    ax.set_xlabel("Risco — Desvio Padrao (%)", fontsize=12)
    ax.set_ylabel("Retorno Esperado (%)", fontsize=12)
    ax.set_title("Fronteira Eficiente de Markowitz\n"
                 "(Ativos: PETR4, VALE3, ITUB4, WEGE3, ABEV3, RENT3)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "grafico_fronteira_eficiente.png"), dpi=130)
    plt.show()
    print(f"Grafico salvo em: {os.path.join(FIGURAS_DIR, "grafico_fronteira_eficiente.png")}")


def grafico_composicao_carteira(nomes: list, cart_min_var: dict,
                                 cart_max_sharpe: dict) -> None:
    """Grafico de pizza para composicao de cada carteira."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    cores = plt.cm.Set3(np.linspace(0, 1, len(nomes)))

    for ax, cart, titulo in [
        (axes[0], cart_min_var, "Carteira Minima Variancia"),
        (axes[1], cart_max_sharpe, f"Carteira Max. Sharpe ({cart_max_sharpe['sharpe']:.2f})"),
    ]:
        pesos = cart["pesos"]
        idx   = pesos > 0.005
        wedges, texts, autotexts = ax.pie(
            pesos[idx], labels=np.array(nomes)[idx],
            autopct="%1.1f%%", startangle=90,
            colors=cores[idx], pctdistance=0.8
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title(f"{titulo}\nRetorno={cart['retorno']*100:.1f}%  "
                     f"Risco={cart['risco']*100:.1f}%", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "grafico_composicao_carteiras.png"), dpi=120)
    plt.show()
    print(f"Grafico salvo em: {os.path.join(FIGURAS_DIR, "grafico_composicao_carteiras.png")}")


def grafico_heatmap_correlacao(nomes: list, correlacoes: np.ndarray) -> None:
    """Heatmap da matriz de correlacao."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlacoes, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(nomes)))
    ax.set_yticks(range(len(nomes)))
    ax.set_xticklabels(nomes, rotation=45, ha="right")
    ax.set_yticklabels(nomes)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(j, i, f"{correlacoes[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if abs(correlacoes[i, j]) < 0.7 else "white",
                    fontsize=10)
    ax.set_title("Matriz de Correlacao entre Ativos", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "grafico_correlacao.png"), dpi=120)
    plt.show()
    print(f"Grafico salvo em: {os.path.join(FIGURAS_DIR, "grafico_correlacao.png")}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  MODULO 06 — GESTAO DE PORTFOLIO")
    print("=" * 65)

    RF = 0.105  # Selic = 10,5% a.a.
    nomes, retornos, vols, cov = montar_parametros(ATIVOS, CORRELACOES)

    # --- Ativos individuais ---
    print("\n[1] ATIVOS INDIVIDUAIS")
    df_at = pd.DataFrame({
        "Ativo":  nomes,
        "Retorno (%)": [f"{r*100:.1f}" for r in retornos],
        "Vol (%)":     [f"{v*100:.1f}" for v in vols],
        "Sharpe":      [f"{(r-RF)/v:.3f}" for r, v in zip(retornos, vols)],
    })
    print(df_at.to_string(index=False))

    # --- Carteira igualitaria ---
    print(f"\n[2] CARTEIRA IGUALITARIA (1/N)")
    pesos_iguais = np.ones(len(nomes)) / len(nomes)
    rp_ig = retorno_carteira(pesos_iguais, retornos)
    sp_ig = risco_carteira(pesos_iguais, cov)
    sh_ig = sharpe_ratio(pesos_iguais, retornos, cov, RF)
    print(f"    Retorno: {rp_ig*100:.2f}%  |  Risco: {sp_ig*100:.2f}%  |  Sharpe: {sh_ig:.3f}")

    div = beneficio_diversificacao(pesos_iguais, vols, cov)
    print(f"\n    Beneficio de Diversificacao:")
    print(f"      Vol ponderada (sem diversif.): {div['volatilidade_ponderada']*100:.2f}%")
    print(f"      Vol real da carteira:          {div['volatilidade_real']*100:.2f}%")
    print(f"      Reducao:                       {div['reducao_absoluta']*100:.2f}pp  "
          f"({div['reducao_percentual']:.1f}%)")

    # --- Carteira de minima variancia ---
    print(f"\n[3] CARTEIRA DE MINIMA VARIANCIA")
    cart_min = carteira_minima_variancia(retornos, cov)
    print(f"    Retorno: {cart_min['retorno']*100:.2f}%  |  "
          f"Risco: {cart_min['risco']*100:.2f}%")
    print("    Composicao:")
    for nome, p in zip(nomes, cart_min["pesos"]):
        if p > 0.001:
            print(f"      {nome}: {p*100:.1f}%")

    # --- Carteira de maximo Sharpe ---
    print(f"\n[4] CARTEIRA DE MAXIMO SHARPE (TANGENTE)")
    cart_sharpe = carteira_maximo_sharpe(retornos, cov, RF)
    print(f"    Retorno: {cart_sharpe['retorno']*100:.2f}%  |  "
          f"Risco: {cart_sharpe['risco']*100:.2f}%  |  "
          f"Sharpe: {cart_sharpe['sharpe']:.4f}")
    print("    Composicao:")
    for nome, p in zip(nomes, cart_sharpe["pesos"]):
        if p > 0.001:
            print(f"      {nome}: {p*100:.1f}%")

    # --- Indicadores de performance ---
    print(f"\n[5] INDICADORES DE PERFORMANCE")
    rp_sh = cart_sharpe["retorno"]
    beta_approx = sum(p * ATIVOS[n]["beta"] for n, p in zip(nomes, cart_sharpe["pesos"]))
    treynor = treynor_ratio(rp_sh, RF, beta_approx)
    print(f"    Carteira Max. Sharpe:")
    print(f"      Sharpe Ratio : {cart_sharpe['sharpe']:.4f}")
    print(f"      Treynor Ratio: {treynor:.4f}  (beta={beta_approx:.2f})")
    print(f"\n    Interpretacao: Sharpe > 1 e considerado excelente.")
    print(f"    Sharpe > 0.5 e bom; < 0 indica abaixo do ativo livre de risco.")

    # --- Fronteira eficiente ---
    print(f"\n[6] CALCULANDO FRONTEIRA EFICIENTE...")
    df_fronteira = fronteira_eficiente(retornos, cov, n_pontos=80)
    print(f"    {len(df_fronteira)} pontos calculados na fronteira.")

    # --- Monte Carlo de portfolios ---
    print(f"\n[7] SIMULANDO PORTFOLIOS ALEATORIOS (Monte Carlo)...")
    df_sim = simular_portfolios_aleatorios(retornos, cov, RF, n_portfolios=6_000)
    print(f"    Melhor Sharpe encontrado: {df_sim['sharpe'].max():.4f}")
    print(f"    vs. Otimizacao:           {cart_sharpe['sharpe']:.4f}")

    # --- Correlacao ---
    print(f"\n[8] ANALISE DE CORRELACAO")
    df_corr = analise_correlacao(nomes, CORRELACOES)
    print(df_corr.round(2).to_string())

    # --- Graficos ---
    print(f"\n[9] GERANDO GRAFICOS...")
    grafico_heatmap_correlacao(nomes, CORRELACOES)
    grafico_fronteira_eficiente(
        df_fronteira, df_sim, cart_min, cart_sharpe,
        (nomes, retornos, vols), RF
    )
    grafico_composicao_carteira(nomes, cart_min, cart_sharpe)

    print("\n" + "=" * 65)
    print("  Modulo 06 concluido com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
