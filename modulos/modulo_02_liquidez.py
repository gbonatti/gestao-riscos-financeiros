"""
Modulo 02 — Risco de Liquidez
================================
Conteudo:
  - Liquidity Coverage Ratio (LCR) — Basileia III
  - Net Stable Funding Ratio (NSFR)
  - Gap de Liquidez por prazo
  - Stress Testing (cenarios adversos)
  - Colchao de liquidez
  - Graficos de projecao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# 1. LIQUIDITY COVERAGE RATIO (LCR)
# ---------------------------------------------------------------------------

def calcular_lcr(hqla: float, saidas_liquidas_30d: float) -> dict:
    """
    Calcula o LCR (Liquidity Coverage Ratio) conforme Basileia III.

    LCR = HQLA / Saidas Liquidas de Caixa em 30 dias >= 100%

    Parameters
    ----------
    hqla             : float - High Quality Liquid Assets (R$ milhoes)
    saidas_liquidas  : float - Saidas liquidas projetadas em 30 dias (R$ milhoes)

    Returns
    -------
    dict com resultado e classificacao
    """
    if saidas_liquidas_30d <= 0:
        raise ValueError("Saidas liquidas devem ser positivas.")
    lcr = hqla / saidas_liquidas_30d * 100  # em porcentagem
    adequado = lcr >= 100
    return {
        "HQLA (R$ mi)": hqla,
        "Saidas_30d (R$ mi)": saidas_liquidas_30d,
        "LCR (%)": round(lcr, 2),
        "Minimo exigido (%)": 100.0,
        "Status": "ADEQUADO" if adequado else "INSUFICIENTE",
        "Folga / Deficit (R$ mi)": round(hqla - saidas_liquidas_30d, 2),
    }


def componentes_hqla(nivel1: float, nivel2a: float, nivel2b: float) -> dict:
    """
    Calcula o HQLA total com base nos tres niveis de ativos.

    Nivel 1 : caixa, reservas no BC, titulos soberanos — 100% do valor
    Nivel 2A: titulos de alta qualidade com haircut de 15%
    Nivel 2B: titulos de menor liquidez com haircut de 50%
    """
    hqla_n1  = nivel1
    hqla_n2a = nivel2a * 0.85
    hqla_n2b = nivel2b * 0.50
    total    = hqla_n1 + hqla_n2a + hqla_n2b
    return {
        "Nivel 1 (100%)": round(hqla_n1, 2),
        "Nivel 2A (85%)": round(hqla_n2a, 2),
        "Nivel 2B (50%)": round(hqla_n2b, 2),
        "HQLA Total": round(total, 2),
    }


# ---------------------------------------------------------------------------
# 2. NET STABLE FUNDING RATIO (NSFR)
# ---------------------------------------------------------------------------

def calcular_nsfr(funding_estavel_disponivel: float, funding_estavel_necessario: float) -> dict:
    """
    Calcula o NSFR para avaliacao de liquidez estrutural de longo prazo.

    NSFR = Funding Estavel Disponivel / Funding Estavel Necessario >= 100%
    """
    nsfr = funding_estavel_disponivel / funding_estavel_necessario * 100
    return {
        "Funding Disponivel (R$ mi)": funding_estavel_disponivel,
        "Funding Necessario (R$ mi)": funding_estavel_necessario,
        "NSFR (%)": round(nsfr, 2),
        "Status": "ADEQUADO" if nsfr >= 100 else "INSUFICIENTE",
    }


# ---------------------------------------------------------------------------
# 3. GAP DE LIQUIDEZ
# ---------------------------------------------------------------------------

def gap_liquidez(entradas: list, saidas: list, prazos: list) -> pd.DataFrame:
    """
    Calcula o Gap de Liquidez por faixa de prazo.

    Parameters
    ----------
    entradas : list - Entradas de caixa projetadas por prazo (R$ mi)
    saidas   : list - Saidas de caixa projetadas por prazo (R$ mi)
    prazos   : list - Descricao dos prazos

    Returns
    -------
    DataFrame com gap simples e acumulado
    """
    if not (len(entradas) == len(saidas) == len(prazos)):
        raise ValueError("Listas devem ter o mesmo tamanho.")

    gaps = [e - s for e, s in zip(entradas, saidas)]
    gaps_acum = np.cumsum(gaps).tolist()

    df = pd.DataFrame({
        "Prazo": prazos,
        "Entradas (R$ mi)": entradas,
        "Saidas (R$ mi)": saidas,
        "Gap (R$ mi)": gaps,
        "Gap Acumulado (R$ mi)": [round(g, 2) for g in gaps_acum],
    })
    return df


# ---------------------------------------------------------------------------
# 4. STRESS TESTING
# ---------------------------------------------------------------------------

CENARIOS_ESTRESSE = {
    "Corrida Bancaria": {
        "descricao": "20% dos depositos sacados em 5 dias",
        "reducao_hqla_pct": 0.00,
        "saidas_extras_pct": 0.20,
    },
    "Crise de Mercado": {
        "descricao": "Haircut adicional de 30% nos ativos de nivel 2",
        "reducao_hqla_pct": 0.30,
        "saidas_extras_pct": 0.05,
    },
    "Falta de Credito": {
        "descricao": "50% das linhas comprometidas nao renovadas",
        "reducao_hqla_pct": 0.10,
        "saidas_extras_pct": 0.15,
    },
    "Crise Combinada": {
        "descricao": "Combinacao de corrida bancaria + crise de mercado",
        "reducao_hqla_pct": 0.20,
        "saidas_extras_pct": 0.30,
    },
}


def stress_test_liquidez(hqla_base: float, saidas_base: float,
                          depositos_totais: float) -> pd.DataFrame:
    """
    Executa stress tests de liquidez em multiplos cenarios.

    Parameters
    ----------
    hqla_base        : float - HQLA em condicoes normais (R$ mi)
    saidas_base      : float - Saidas de caixa em 30 dias em condicoes normais
    depositos_totais : float - Total de depositos do banco

    Returns
    -------
    DataFrame com LCR em cada cenario
    """
    resultados = []
    lcr_base = hqla_base / saidas_base * 100

    for nome, params in CENARIOS_ESTRESSE.items():
        hqla_stress = hqla_base * (1 - params["reducao_hqla_pct"])
        saidas_stress = saidas_base + depositos_totais * params["saidas_extras_pct"]
        lcr_stress = hqla_stress / saidas_stress * 100
        resultados.append({
            "Cenario": nome,
            "Descricao": params["descricao"],
            "HQLA (R$ mi)": round(hqla_stress, 2),
            "Saidas (R$ mi)": round(saidas_stress, 2),
            "LCR (%)": round(lcr_stress, 2),
            "Adequado?": "SIM" if lcr_stress >= 100 else "NAO",
            "Variacao vs Base (pp)": round(lcr_stress - lcr_base, 2),
        })

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# 5. COLCHAO DE LIQUIDEZ MINIMO
# ---------------------------------------------------------------------------

def colchao_minimo(saidas_projetadas_30d: float, margem_seguranca: float = 0.20) -> float:
    """
    Calcula o HQLA minimo necessario para cumprir LCR com margem de seguranca.

    colchao = saidas_30d * (1 + margem_seguranca)
    """
    return saidas_projetadas_30d * (1 + margem_seguranca)


# ---------------------------------------------------------------------------
# 6. GRAFICOS
# ---------------------------------------------------------------------------

def grafico_gap_liquidez(df_gap: pd.DataFrame) -> None:
    """Plota o gap de liquidez por prazo (barras)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cores = ["#27ae60" if g >= 0 else "#e74c3c" for g in df_gap["Gap (R$ mi)"]]
    axes[0].bar(df_gap["Prazo"], df_gap["Gap (R$ mi)"], color=cores, edgecolor="white")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Gap de Liquidez por Prazo", fontsize=13)
    axes[0].set_xlabel("Faixa de Prazo")
    axes[0].set_ylabel("Gap (R$ mi)")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(df_gap["Gap (R$ mi)"]):
        axes[0].text(i, v + (2 if v >= 0 else -8), f"{v:+.0f}", ha="center", fontsize=9)

    cores_acum = ["#27ae60" if g >= 0 else "#e74c3c" for g in df_gap["Gap Acumulado (R$ mi)"]]
    axes[1].bar(df_gap["Prazo"], df_gap["Gap Acumulado (R$ mi)"], color=cores_acum, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Gap de Liquidez Acumulado", fontsize=13)
    axes[1].set_xlabel("Faixa de Prazo")
    axes[1].set_ylabel("Gap Acumulado (R$ mi)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("grafico_gap_liquidez.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_gap_liquidez.png'")


def grafico_stress_test(df_stress: pd.DataFrame, lcr_base: float) -> None:
    """Plota LCR em cada cenario de estresse."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cenarios = ["Base"] + df_stress["Cenario"].tolist()
    lcrs = [lcr_base] + df_stress["LCR (%)"].tolist()
    cores = ["#3498db"] + ["#27ae60" if l >= 100 else "#e74c3c" for l in df_stress["LCR (%)"]]

    bars = ax.barh(cenarios, lcrs, color=cores, edgecolor="white", height=0.5)
    ax.axvline(100, color="black", linewidth=2, linestyle="--", label="Minimo (100%)")
    for bar, val in zip(bars, lcrs):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    verde  = mpatches.Patch(color="#27ae60", label="Adequado")
    vermelho = mpatches.Patch(color="#e74c3c", label="Insuficiente")
    azul   = mpatches.Patch(color="#3498db", label="Base (normal)")
    ax.legend(handles=[verde, vermelho, azul], loc="lower right")
    ax.set_title("Stress Test de Liquidez — LCR por Cenario", fontsize=13)
    ax.set_xlabel("LCR (%)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_stress_liquidez.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_stress_liquidez.png'")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  MODULO 02 — RISCO DE LIQUIDEZ")
    print("=" * 65)

    # Dados fictícios do banco hipotetico "BancoAlpha"
    HQLA_N1  = 120.0   # R$ 120 mi — caixa + reservas BC
    HQLA_N2A = 80.0    # R$ 80 mi — titulos soberanos nivel 2A
    HQLA_N2B = 40.0    # R$ 40 mi — titulos corporativos nivel 2B
    SAIDAS_30D     = 170.0  # R$ 170 mi saidas liquidas em 30 dias
    DEPOSITOS      = 600.0  # R$ 600 mi total de depositos
    FUND_DISPONIVEL = 450.0
    FUND_NECESSARIO = 380.0

    # --- HQLA ---
    print("\n[1] COMPOSICAO DO HQLA")
    hqla_comp = componentes_hqla(HQLA_N1, HQLA_N2A, HQLA_N2B)
    for k, v in hqla_comp.items():
        print(f"    {k}: R$ {v:,.2f} mi")
    hqla_total = hqla_comp["HQLA Total"]

    # --- LCR ---
    print(f"\n[2] LIQUIDITY COVERAGE RATIO (LCR)")
    lcr_result = calcular_lcr(hqla_total, SAIDAS_30D)
    for k, v in lcr_result.items():
        print(f"    {k}: {v}")

    # --- NSFR ---
    print(f"\n[3] NET STABLE FUNDING RATIO (NSFR)")
    nsfr_result = calcular_nsfr(FUND_DISPONIVEL, FUND_NECESSARIO)
    for k, v in nsfr_result.items():
        print(f"    {k}: {v}")

    # --- Gap de Liquidez ---
    print(f"\n[4] GAP DE LIQUIDEZ POR PRAZO")
    prazos   = ["1-7 dias", "8-30 dias", "31-90 dias", "91-180 dias", "181-365 dias"]
    entradas = [80,  150, 220, 180, 160]
    saidas   = [120, 130, 180, 160, 140]
    df_gap = gap_liquidez(entradas, saidas, prazos)
    print(df_gap.to_string(index=False))

    # --- Stress Test ---
    print(f"\n[5] STRESS TESTING")
    df_stress = stress_test_liquidez(hqla_total, SAIDAS_30D, DEPOSITOS)
    print(df_stress[["Cenario", "HQLA (R$ mi)", "Saidas (R$ mi)", "LCR (%)", "Adequado?"]].to_string(index=False))

    lcr_base = lcr_result["LCR (%)"]

    # --- Colchao minimo ---
    print(f"\n[6] COLCHAO DE LIQUIDEZ MINIMO")
    colchao = colchao_minimo(SAIDAS_30D, margem_seguranca=0.20)
    print(f"    Saidas projetadas 30d : R$ {SAIDAS_30D:,.2f} mi")
    print(f"    Colchao minimo (120%) : R$ {colchao:,.2f} mi")
    print(f"    HQLA atual            : R$ {hqla_total:,.2f} mi")
    print(f"    Situacao              : {'ADEQUADO' if hqla_total >= colchao else 'ABAIXO DO COLCHAO'}")

    # --- Graficos ---
    print(f"\n[7] GERANDO GRAFICOS...")
    grafico_gap_liquidez(df_gap)
    grafico_stress_test(df_stress, lcr_base)

    print("\n" + "=" * 65)
    print("  Modulo 02 concluido com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
