"""
Caso Pratico Integrado — Banco BetaFinance S.A.
=================================================
Simulacao completa de um banco hipotetico enfrentando tres desafios
simultaneos de gestao de riscos:

  1. Alta da taxa de juros -> perda na carteira de titulos
  2. Saques elevados -> pressao sobre liquidez (LCR cai)
  3. Transacoes suspeitas -> risco de PLD/AML

Este script importa funcoes dos seis modulos e produz um relatorio
integrado de riscos com recomendacoes.

Estrutura:
  - Balanco simplificado do banco
  - Choque de taxa de juros (+2 p.p.)
  - Analise de liquidez pos-choque
  - Deteccao de transacoes suspeitas no mesmo periodo
  - Carteira de investimentos e VaR
  - Relatorio final de riscos
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# --- importacoes dos modulos ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modulos.modulo_01_taxa_juros import (
    preco_titulo, duration_modificada, convexidade,
    variacao_preco_convexidade, var_renda_fixa, gap_taxas
)
from modulos.modulo_02_liquidez import (
    calcular_lcr, componentes_hqla, gap_liquidez, stress_test_liquidez
)
from modulos.modulo_03_pld_aml import (
    gerar_dataset_transacoes, treinar_modelo_fraude,
    zscore_transacao, score_kyc
)
from modulos.modulo_05_risco_mercado import (
    var_parametrico, var_historico,
    simular_retornos_historicos, expected_shortfall
)
from modulos.modulo_06_portfolio import (
    montar_parametros, ATIVOS, CORRELACOES,
    carteira_maximo_sharpe, retorno_carteira, risco_carteira
)


# Diretorio de saida dos graficos
FIGURAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)


# ===========================================================================
# BALANCO DO BANCO BETAFINANCE S.A. (valores fictícios em R$ milhoes)
# ===========================================================================

BALANCO = {
    # ATIVOS
    "Caixa e reservas no BC":     150.0,
    "Titulos publicos (LFT/NTN)": 300.0,
    "CDB emitidos por outros":     80.0,
    "Carteira de credito":        800.0,
    "Acoes e fundos":             120.0,
    "Imoveis e outros":           100.0,
    # PASSIVOS
    "Depositos a vista":          250.0,
    "Depositos a prazo (CDB)":    600.0,
    "Captacao interbancaria":     200.0,
    "Patrimonio liquido":         500.0,
}

# Parametros dos titulos na carteira
CARTEIRA_TITULOS = [
    # (descricao, face, cupom_rate, yield_atual, n_anos, volume_milhoes)
    ("NTN-F 2027",  1000, 0.12, 0.115, 3, 100.0),
    ("NTN-F 2030",  1000, 0.12, 0.115, 6, 120.0),
    ("LTN 2026",    1000, 0.00, 0.115, 2,  80.0),
]

RF = 0.105   # Taxa livre de risco (Selic)
CHOQUE_TAXA = 0.02  # +2 p.p. de alta na taxa de juros


# ===========================================================================
# 1. RELATORIO DO BALANCO
# ===========================================================================

def exibir_balanco():
    print("\n" + "=" * 70)
    print("  BANCO BETAFINANCE S.A. — BALANCO PATRIMONIAL SIMPLIFICADO")
    print("=" * 70)

    total_ativos = sum(v for k, v in BALANCO.items() if k in [
        "Caixa e reservas no BC", "Titulos publicos (LFT/NTN)",
        "CDB emitidos por outros", "Carteira de credito",
        "Acoes e fundos", "Imoveis e outros"
    ])
    print(f"\n  {'ATIVOS':50s}  {'R$ mi':>8}")
    print("  " + "-" * 62)
    for k, v in list(BALANCO.items())[:6]:
        print(f"  {k:50s}  {v:>8,.1f}")
    print(f"  {'TOTAL ATIVO':50s}  {total_ativos:>8,.1f}")

    total_passivos = sum(v for k, v in BALANCO.items() if k in [
        "Depositos a vista", "Depositos a prazo (CDB)",
        "Captacao interbancaria", "Patrimonio liquido"
    ])
    print(f"\n  {'PASSIVOS + PL':50s}  {'R$ mi':>8}")
    print("  " + "-" * 62)
    for k, v in list(BALANCO.items())[6:]:
        print(f"  {k:50s}  {v:>8,.1f}")
    print(f"  {'TOTAL PASSIVO + PL':50s}  {total_passivos:>8,.1f}")


# ===========================================================================
# 2. CHOQUE DE TAXA DE JUROS
# ===========================================================================

def analise_choque_juros():
    print("\n" + "=" * 70)
    print(f"  MODULO 1 — CHOQUE DE TAXA DE JUROS: +{CHOQUE_TAXA*100:.0f} p.p.")
    print("=" * 70)

    impacto_total = 0.0
    dados = []

    for desc, face, cpn, ytm, n, vol in CARTEIRA_TITULOS:
        preco_antes = preco_titulo(cpn, face, ytm, n)
        preco_depois = preco_titulo(cpn, face, ytm + CHOQUE_TAXA, n)

        d_mod = duration_modificada(cpn, face, ytm, n)
        conv  = convexidade(cpn, face, ytm, n)
        var_pct = variacao_preco_convexidade(d_mod, conv, CHOQUE_TAXA)
        var_reais = var_pct * preco_antes / face * vol * 1_000_000

        impacto_total += var_reais / 1_000_000  # em milhoes

        dados.append({
            "Titulo":        desc,
            "D_mod":         round(d_mod, 2),
            "Convexidade":   round(conv, 2),
            "Preco antes":   round(preco_antes, 2),
            "Preco depois":  round(preco_depois, 2),
            "Var preco (%)": round(var_pct * 100, 2),
            "Impacto (R$mi)": round(var_reais / 1_000_000, 2),
        })

    df = pd.DataFrame(dados)
    print(df.to_string(index=False))
    print(f"\n  PERDA TOTAL ESTIMADA NA CARTEIRA: R$ {abs(impacto_total):,.2f} milhoes")

    # GAP de taxas
    gap = gap_taxas(ativos_sensiveis=380.0, passivos_sensiveis=600.0)
    print(f"\n  GAP DE TAXAS:")
    print(f"    {gap['interpretacao']}")
    print(f"    Impacto no NII (+2%): R$ {gap['gap']*0.02:,.0f} mi")

    return impacto_total


# ===========================================================================
# 3. ANALISE DE LIQUIDEZ
# ===========================================================================

def analise_liquidez(perda_carteira: float):
    print("\n" + "=" * 70)
    print("  MODULO 2 — RISCO DE LIQUIDEZ (pos-choque)")
    print("=" * 70)

    # HQLA apos perda de mercado nos titulos
    hqla_n1  = BALANCO["Caixa e reservas no BC"]
    hqla_n2a = BALANCO["Titulos publicos (LFT/NTN)"] + perda_carteira  # desconta perda
    hqla_n2b = BALANCO["CDB emitidos por outros"] * 0.5

    hqla_comp = componentes_hqla(hqla_n1, hqla_n2a, hqla_n2b)
    hqla_total = hqla_comp["HQLA Total"]

    # Saidas estimadas: saques elevados (5% dos depositos)
    saidas_normais = (BALANCO["Depositos a vista"] * 0.05 +
                      BALANCO["Depositos a prazo (CDB)"] * 0.03 +
                      BALANCO["Captacao interbancaria"] * 0.10)
    saidas_estresse = saidas_normais * 2.5  # cenario adverso

    lcr_normal  = calcular_lcr(hqla_total, saidas_normais)
    lcr_estresse = calcular_lcr(hqla_total, saidas_estresse)

    print(f"\n  HQLA apos choque:")
    for k, v in hqla_comp.items():
        print(f"    {k}: R$ {v:,.2f} mi")

    print(f"\n  LCR em condicoes normais:")
    print(f"    Saidas 30d: R$ {saidas_normais:,.1f} mi  |  "
          f"LCR: {lcr_normal['LCR (%)']:.1f}%  |  {lcr_normal['Status']}")

    print(f"\n  LCR em cenario de estresse (saques 2.5x):")
    print(f"    Saidas 30d: R$ {saidas_estresse:,.1f} mi  |  "
          f"LCR: {lcr_estresse['LCR (%)']:.1f}%  |  {lcr_estresse['Status']}")

    # Gap de liquidez
    prazos   = ["1-7d", "8-30d", "31-90d", "91-180d", "180-365d"]
    entradas = [60, 140, 200, 180, 160]
    saidas   = [90, 160, 170, 150, 140]
    df_gap = gap_liquidez(entradas, saidas, prazos)
    print(f"\n  Gap de Liquidez por Prazo:")
    print(df_gap.to_string(index=False))

    return lcr_estresse


# ===========================================================================
# 4. DETECCAO DE TRANSACOES SUSPEITAS
# ===========================================================================

def analise_pld():
    print("\n" + "=" * 70)
    print("  MODULO 3 — PLD/AML — DETECCAO DE TRANSACOES SUSPEITAS")
    print("=" * 70)

    # Gerar transacoes do periodo de crise
    df = gerar_dataset_transacoes(n_legitimas=800, n_suspeitas=80, seed=2024)
    print(f"\n  Transacoes no periodo: {len(df)}")
    print(f"  Suspeitas reais (ground truth): {(df['suspeita']==1).sum()}")

    # Treinar modelo
    resultado = treinar_modelo_fraude(df)
    print(f"\n  Modelo RF treinado — AUC: {resultado['auc']:.4f}")

    # Alertas de alto risco (prob > 0.7)
    probs = resultado["modelo"].predict_proba(resultado["X_test"])[:, 1]
    alertas_criticos = (probs > 0.7).sum()
    print(f"  Alertas criticos (prob > 70%): {alertas_criticos}")

    # Z-score de transacoes especificas do periodo
    print(f"\n  Analise Z-score de transacoes selecionadas:")
    casos_suspeitos = [
        ("Cliente Offshore Corp",    95_000, 8_000,  15_000),
        ("Transf. fracionada 1",      9_200, 4_000,   2_000),
        ("Transf. fracionada 2",      9_400, 4_000,   2_000),
        ("Transf. fracionada 3",      9_100, 4_000,   2_000),
        ("Saque elevado — Agencia 5", 45_000, 5_000,  8_000),
    ]
    for nome, val, media, dp in casos_suspeitos:
        r = zscore_transacao(val, media, dp)
        print(f"    {nome:<35s}  R$ {val:>8,.0f}  "
              f"Z={r['z_score']:>6.2f}  {r['nivel']}")

    return resultado


# ===========================================================================
# 5. VaR DA CARTEIRA DE INVESTIMENTOS
# ===========================================================================

def analise_var_carteira():
    print("\n" + "=" * 70)
    print("  MODULO 5 — VaR DA CARTEIRA DE INVESTIMENTOS")
    print("=" * 70)

    POSICAO_ACOES = BALANCO["Acoes e fundos"] * 1_000_000  # em R$
    SIGMA         = 0.020   # 2% vol. diaria
    MU            = 0.0003

    retornos = simular_retornos_historicos(n_dias=250, mu=MU, sigma=SIGMA, seed=99)

    v_param = var_parametrico(POSICAO_ACOES, SIGMA, 1,  0.99)
    v_hist  = var_historico(retornos, POSICAO_ACOES, 0.99)
    es      = expected_shortfall(retornos, POSICAO_ACOES, 0.99)
    v_10d   = var_parametrico(POSICAO_ACOES, SIGMA, 10, 0.99)

    print(f"\n  Carteira de Acoes e Fundos: R$ {POSICAO_ACOES/1e6:,.0f} milhoes")
    print(f"  Volatilidade diaria: {SIGMA*100:.1f}%")
    print(f"\n  VaR Parametrico 99% 1d : R$ {v_param/1e6:,.3f} mi")
    print(f"  VaR Historico   99% 1d : R$ {v_hist/1e6:,.3f} mi")
    print(f"  Expected Shortfall 99% : R$ {es/1e6:,.3f} mi")
    print(f"  VaR Parametrico 99% 10d: R$ {v_10d/1e6:,.3f} mi")

    return v_10d


# ===========================================================================
# 6. OTIMIZACAO DO PORTFOLIO DE ACOES
# ===========================================================================

def analise_portfolio():
    print("\n" + "=" * 70)
    print("  MODULO 6 — OTIMIZACAO DO PORTFOLIO DE ACOES")
    print("=" * 70)

    nomes, retornos, vols, cov = montar_parametros(ATIVOS, CORRELACOES)

    cart = carteira_maximo_sharpe(retornos, cov, RF)
    print(f"\n  Carteira otima (max Sharpe):")
    print(f"    Retorno anual: {cart['retorno']*100:.2f}%")
    print(f"    Risco anual:   {cart['risco']*100:.2f}%")
    print(f"    Sharpe Ratio:  {cart['sharpe']:.4f}")
    print(f"\n  Composicao recomendada:")
    for nome, p in zip(nomes, cart["pesos"]):
        if p > 0.005:
            print(f"    {nome}: {p*100:.1f}%  "
                  f"(R$ {p * BALANCO['Acoes e fundos']:,.1f} mi)")
    return cart


# ===========================================================================
# 7. RELATORIO FINAL INTEGRADO
# ===========================================================================

def relatorio_final(perda_juros: float, lcr_estresse: dict,
                     var_10d: float, cart: dict):
    print("\n" + "=" * 70)
    print("  RELATORIO INTEGRADO DE RISCOS — BANCO BETAFINANCE S.A.")
    print("=" * 70)

    pl = BALANCO["Patrimonio liquido"]

    print(f"""
  DATA REFERENCIA: Cenario de Estresse Q2/2026
  PATRIMONIO LIQUIDO: R$ {pl:,.0f} mi

  RESUMO EXECUTIVO
  ----------------
  O banco enfrenta um cenario de estresse combinado:
  alta de taxa de juros (+2 p.p.), saques elevados e transacoes
  suspeitas identificadas. O relatorio consolida os impactos.

  RISCO DE TAXA DE JUROS
  ----------------------
  Perda estimada na carteira de titulos:  R$ {abs(perda_juros):,.2f} mi
  Impacto sobre o PL:                     {abs(perda_juros)/pl*100:.2f}%
  Severidade:                             {'ALTA' if abs(perda_juros)/pl > 0.02 else 'MODERADA'}
  Acao recomendada:                       Reduzir duration da carteira; ampliar hedge com swaps DI

  RISCO DE LIQUIDEZ
  -----------------
  LCR em cenario de estresse:            {lcr_estresse['LCR (%)']:.1f}%
  Status:                                 {lcr_estresse['Status']}
  Folga/Deficit vs. minimo (100%):       R$ {lcr_estresse['Folga / Deficit (R$ mi)']:+,.1f} mi
  Acao recomendada:                       {'Acionar plano de contingencia de liquidez (PCL)' if lcr_estresse['LCR (%)'] < 100 else 'Monitorar; ampliar colchao de HQLA para margem adicional'}

  RISCO PLD/AML
  -------------
  Transacoes em alerta critico identificadas no periodo.
  Padroes detectados: smurfing, volumes incompativeis com perfil, horario atipico.
  Acao recomendada:   Enviar comunicado ao COAF; congelar contas sob investigacao; reforcar monitoramento

  RISCO DE MERCADO (CARTEIRA DE ACOES)
  -------------------------------------
  VaR 99% em 10 dias:                    R$ {var_10d/1e6:,.2f} mi
  Capital em risco / PL:                 {var_10d/1e6/pl*100:.2f}%
  Portfolio atual otimizado:
    Retorno esperado:                     {cart['retorno']*100:.2f}% a.a.
    Volatilidade:                         {cart['risco']*100:.2f}% a.a.
    Sharpe Ratio:                         {cart['sharpe']:.3f}

  SEMAFORO DE RISCO GERAL
  -----------------------
  Risco de Juros   : {'[VERMELHO]' if abs(perda_juros)/pl > 0.02 else '[AMARELO]'}
  Risco de Liquidez: {'[VERMELHO]' if lcr_estresse['LCR (%)'] < 100 else '[AMARELO]'}
  Risco PLD/AML    : [VERMELHO] (transacoes criticas identificadas)
  Risco de Mercado : {'[AMARELO]' if var_10d/1e6/pl < 0.05 else '[VERMELHO]'}

  ACOES PRIORITARIAS
  ------------------
  1. Convocar comite de crise de liquidez
  2. Reduzir duration da carteira de titulos (venda de LTN longas)
  3. Acionar PCL: emissao de LCI/CDB para reforco de funding
  4. Bloquear contas com alertas PLD criticos e notificar COAF
  5. Revisar limites de VaR e executar hedge cambial se aplicavel
""")


# ===========================================================================
# 8. DASHBOARD GRAFICO
# ===========================================================================

def gerar_dashboard(perda_juros: float, lcr_estresse: dict, var_10d: float):
    """Gera painel visual com os principais indicadores de risco."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Dashboard de Riscos — Banco BetaFinance S.A.\n"
                 "Cenario de Estresse Q2/2026", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Painel 1: Impacto nos titulos por choque de taxa ---
    ax1 = fig.add_subplot(gs[0, 0])
    titulos = [t[0] for t in CARTEIRA_TITULOS]
    volumes = [t[5] for t in CARTEIRA_TITULOS]
    impactos_pct = []
    for desc, face, cpn, ytm, n, vol in CARTEIRA_TITULOS:
        from modulos.modulo_01_taxa_juros import duration_modificada as dm, convexidade as cv
        d = dm(cpn, face, ytm, n)
        c = cv(cpn, face, ytm, n)
        impactos_pct.append(variacao_preco_convexidade(d, c, CHOQUE_TAXA) * 100)
    ax1.barh(titulos, impactos_pct, color=["#e74c3c"] * 3)
    ax1.set_title("Variacao no Preco dos Titulos\n(Choque +2 p.p.)", fontsize=10)
    ax1.set_xlabel("Variacao (%)")
    ax1.axvline(0, color="black")
    ax1.grid(axis="x", alpha=0.3)

    # --- Painel 2: LCR normal vs estresse ---
    ax2 = fig.add_subplot(gs[0, 1])
    lcr_values = [lcr_estresse["LCR (%)"] * 1.8, lcr_estresse["LCR (%)"]]  # normal estimado
    labels = ["Normal", "Estresse"]
    cores = ["#27ae60" if v >= 100 else "#e74c3c" for v in lcr_values]
    bars = ax2.bar(labels, lcr_values, color=cores, edgecolor="white", width=0.5)
    ax2.axhline(100, color="red", linewidth=2, linestyle="--", label="Minimo 100%")
    for bar, val in zip(bars, lcr_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("LCR — Normal vs. Estresse", fontsize=10)
    ax2.set_ylabel("LCR (%)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # --- Painel 3: VaR por horizonte ---
    ax3 = fig.add_subplot(gs[0, 2])
    POSICAO = BALANCO["Acoes e fundos"] * 1e6
    horizontes = [1, 5, 10, 22]
    vars_h = [var_parametrico(POSICAO, 0.020, h, 0.99) / 1e6 for h in horizontes]
    ax3.plot(horizontes, vars_h, marker="o", color="steelblue", linewidth=2)
    ax3.fill_between(horizontes, vars_h, alpha=0.15, color="steelblue")
    ax3.set_title("VaR 99% por Horizonte\n(Carteira de Acoes)", fontsize=10)
    ax3.set_xlabel("Horizonte (dias)")
    ax3.set_ylabel("VaR (R$ mi)")
    ax3.grid(alpha=0.3)

    # --- Painel 4: Composicao do HQLA ---
    ax4 = fig.add_subplot(gs[1, 0])
    hqla_vals = [BALANCO["Caixa e reservas no BC"],
                 BALANCO["Titulos publicos (LFT/NTN)"] * 0.85,
                 BALANCO["CDB emitidos por outros"] * 0.50]
    hqla_labels = ["Nivel 1\n(Caixa/BC)", "Nivel 2A\n(Titulos Sov.)", "Nivel 2B\n(Outros)"]
    cores_hqla = ["#27ae60", "#f39c12", "#e67e22"]
    ax4.bar(hqla_labels, hqla_vals, color=cores_hqla, edgecolor="white")
    ax4.set_title("Composicao do HQLA (R$ mi)", fontsize=10)
    ax4.set_ylabel("R$ milhoes")
    ax4.grid(axis="y", alpha=0.3)

    # --- Painel 5: Semaforo de riscos ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(0, 4)
    ax5.set_ylim(-0.5, 4)
    ax5.axis("off")
    ax5.set_title("Semaforo de Riscos", fontsize=11, fontweight="bold")
    riscos = ["Taxa de Juros", "Liquidez", "PLD/AML", "Mercado"]
    cores_sem = ["#e74c3c", "#e74c3c", "#e74c3c", "#f39c12"]
    for i, (risco, cor) in enumerate(zip(riscos, cores_sem)):
        circle = plt.Circle((0.5, i * 0.9), 0.25, color=cor, zorder=5)
        ax5.add_patch(circle)
        ax5.text(1.0, i * 0.9, risco, va="center", fontsize=11)

    # --- Painel 6: Balanco ---
    ax6 = fig.add_subplot(gs[1, 2])
    itens_at = ["Caixa/BC", "Titulos", "Credito", "Acoes", "Outros"]
    vals_at  = [150, 380, 800, 120, 100]
    ax6.pie(vals_at, labels=itens_at, autopct="%1.0f%%",
            startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(itens_at))),
            pctdistance=0.75)
    ax6.set_title("Composicao dos Ativos\nBanco BetaFinance S.A.", fontsize=10)

    plt.savefig(os.path.join(FIGURAS_DIR, "dashboard_riscos_betafinance.png"), dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Dashboard salvo em: {os.path.join(FIGURAS_DIR, "dashboard_riscos_betafinance.png")}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "#" * 70)
    print("#   CASO PRATICO INTEGRADO — BANCO BETAFINANCE S.A.            #")
    print("#   Gestao de Riscos Financeiros                                #")
    print("#" * 70)

    exibir_balanco()

    perda_juros = analise_choque_juros()
    lcr_estresse = analise_liquidez(perda_juros)
    analise_pld()
    var_10d = analise_var_carteira()
    cart = analise_portfolio()

    relatorio_final(perda_juros, lcr_estresse, var_10d, cart)

    print("[Gerando dashboard visual...]")
    gerar_dashboard(perda_juros, lcr_estresse, var_10d)

    print("\n" + "#" * 70)
    print("#   CASO PRATICO CONCLUIDO                                      #")
    print("#" * 70)


if __name__ == "__main__":
    main()
