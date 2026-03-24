"""
Modulo 03 — Prevencao a Lavagem de Dinheiro (PLD/AML)
========================================================
Conteudo:
  - Score de risco KYC (Know Your Customer)
  - Deteccao estatistica com Z-score
  - Regras heuristicas de monitoramento
  - Modelo de Machine Learning (Random Forest) para deteccao de fraude
  - Geracao de dataset sintetico de transacoes
  - Graficos e relatorio de alertas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. KYC — SCORE DE RISCO DO CLIENTE
# ---------------------------------------------------------------------------

PAISES_ALTO_RISCO = {
    "Cayman Islands", "Ilhas Virgens Britanicas", "Panama",
    "Bahamas", "Liechtenstein", "Seychelles"
}

ATIVIDADES_ALTO_RISCO = {
    "cambio", "cassino", "revendedor_veiculos",
    "joias_metais_preciosos", "imoveis"
}


def score_kyc(is_pep: bool, pais_origem: str, atividade: str,
              anos_relacionamento: int, renda_declarada: float,
              num_ocorrencias: int = 0) -> dict:
    """
    Calcula o score de risco KYC de um cliente.

    Pontuacao de 0 a 100 (quanto maior, maior o risco).

    Classificacao:
      0-30  : Baixo risco
      31-60 : Medio risco
      61-100: Alto risco (monitoramento reforçado)
    """
    score = 0

    # PEP: Person Politically Exposed — peso alto
    if is_pep:
        score += 35

    # Pais de origem
    if pais_origem in PAISES_ALTO_RISCO:
        score += 20

    # Atividade economica
    if atividade.lower() in ATIVIDADES_ALTO_RISCO:
        score += 15

    # Tempo de relacionamento (quanto menor, mais suspeito)
    if anos_relacionamento < 1:
        score += 15
    elif anos_relacionamento < 3:
        score += 8

    # Renda declarada muito baixa (possivel smurfing)
    if renda_declarada < 2_000:
        score += 10

    # Ocorrencias anteriores
    score += min(num_ocorrencias * 5, 20)

    score = min(score, 100)

    if score <= 30:
        classificacao = "BAIXO"
        monitoramento = "Padrao"
    elif score <= 60:
        classificacao = "MEDIO"
        monitoramento = "Elevado"
    else:
        classificacao = "ALTO"
        monitoramento = "Reforçado (due diligence aprofundada)"

    return {
        "score_kyc": score,
        "classificacao": classificacao,
        "monitoramento": monitoramento,
        "is_pep": is_pep,
        "pais_origem": pais_origem,
        "atividade": atividade,
    }


# ---------------------------------------------------------------------------
# 2. Z-SCORE PARA DETECCAO DE ANOMALIAS
# ---------------------------------------------------------------------------

def zscore_transacao(valor: float, media_historica: float,
                     desvio_historico: float) -> dict:
    """
    Calcula o Z-score de uma transacao em relacao ao historico do cliente.

    Z = (X - mu) / sigma

    Classificacao:
      |Z| < 2  : Normal
      2 <= |Z| < 3 : Alerta amarelo
      |Z| >= 3 : Alerta vermelho (suspeito)
    """
    if desvio_historico == 0:
        return {"z": 0, "nivel": "NORMAL", "probabilidade_normal": 1.0}

    z = (valor - media_historica) / desvio_historico
    prob_normal = 2 * norm.cdf(-abs(z))  # probabilidade bi-caudal

    if abs(z) < 2:
        nivel = "NORMAL"
    elif abs(z) < 3:
        nivel = "ALERTA AMARELO"
    else:
        nivel = "ALERTA VERMELHO"

    return {
        "valor": valor,
        "media": media_historica,
        "desvio": desvio_historico,
        "z_score": round(z, 3),
        "nivel": nivel,
        "prob_valor_normal": f"{prob_normal * 100:.2f}%",
    }


def monitorar_carteira_zscore(df_transacoes: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Z-score a todas as transacoes de um DataFrame agrupando por cliente.

    DataFrame deve ter colunas: cliente_id, valor
    """
    stats = df_transacoes.groupby("cliente_id")["valor"].agg(["mean", "std"]).reset_index()
    stats.columns = ["cliente_id", "media", "desvio"]

    df = df_transacoes.merge(stats, on="cliente_id")
    df["z_score"] = (df["valor"] - df["media"]) / df["desvio"].replace(0, np.nan)
    df["z_score"] = df["z_score"].fillna(0)
    df["nivel_alerta"] = df["z_score"].abs().apply(
        lambda z: "VERMELHO" if z >= 3 else ("AMARELO" if z >= 2 else "NORMAL")
    )
    return df


# ---------------------------------------------------------------------------
# 3. REGRAS HEURISTICAS
# ---------------------------------------------------------------------------

def verificar_regras_aml(valor: float, horario: int, frequencia_dia: int,
                          pais: str, pais_origem_cliente: str,
                          valor_medio_cliente: float) -> list:
    """
    Aplica regras heuristicas de deteccao de atividades suspeitas.

    Retorna lista de alertas identificados.
    """
    alertas = []

    # Smurfing: transacoes abaixo do limite de declaracao automatica (R$ 10.000)
    if 8_000 <= valor < 10_000:
        alertas.append("SMURFING: valor proximo ao limite de declaracao compulsoria")

    # Transacao em horario incomum
    if 0 <= horario < 6:
        alertas.append("HORARIO ATIPICO: transacao entre 00h e 06h")

    # Alta frequencia no mesmo dia
    if frequencia_dia > 5:
        alertas.append(f"ALTA FREQUENCIA: {frequencia_dia} transacoes no mesmo dia")

    # Jurisdicao de alto risco
    if pais in PAISES_ALTO_RISCO or pais_origem_cliente in PAISES_ALTO_RISCO:
        alertas.append(f"JURISDICAO DE RISCO: {pais}")

    # Volume incompativel com renda
    if valor > valor_medio_cliente * 10:
        alertas.append(f"VOLUME INCOMPATIVEL: {valor:.0f} vs media {valor_medio_cliente:.0f}")

    return alertas if alertas else ["SEM ALERTAS"]


# ---------------------------------------------------------------------------
# 4. DATASET SINTETICO DE TRANSACOES
# ---------------------------------------------------------------------------

def gerar_dataset_transacoes(n_legitimas: int = 2000,
                              n_suspeitas: int = 200,
                              seed: int = 42) -> pd.DataFrame:
    """
    Gera dataset sintetico de transacoes bancarias para treinamento de modelo ML.

    Features:
      - valor           : valor da transacao
      - horario         : hora do dia (0-23)
      - frequencia_dia  : numero de transacoes no dia pelo mesmo cliente
      - z_score         : desvio em relacao a media do cliente
      - dias_cliente    : dias de relacionamento
      - pais_risco      : 1 se pais de alto risco
      - valor_log       : log do valor
    """
    rng = np.random.default_rng(seed)

    # Transacoes legitimas
    val_leg  = rng.lognormal(mean=8.5, sigma=1.2, size=n_legitimas)   # media ~R$ 5.000
    hor_leg  = rng.integers(6, 22, n_legitimas)
    freq_leg = rng.integers(1, 4, n_legitimas)
    z_leg    = rng.normal(0, 1, n_legitimas)
    dias_leg = rng.integers(180, 3600, n_legitimas)
    risco_leg = rng.choice([0, 1], n_legitimas, p=[0.97, 0.03])

    # Transacoes suspeitas
    val_sus  = rng.choice([
        rng.uniform(8_000, 9_999, n_suspeitas // 2),    # smurfing
        rng.lognormal(mean=11, sigma=0.8, size=n_suspeitas // 2),  # volumes altos
    ]).flatten()[:n_suspeitas]
    hor_sus  = rng.choice(list(range(0, 6)) + list(range(22, 24)), n_suspeitas)
    freq_sus = rng.integers(6, 20, n_suspeitas)
    z_sus    = rng.uniform(3, 8, n_suspeitas)
    dias_sus = rng.integers(1, 90, n_suspeitas)
    risco_sus = rng.choice([0, 1], n_suspeitas, p=[0.30, 0.70])

    n_total  = n_legitimas + n_suspeitas
    df = pd.DataFrame({
        "valor":          np.concatenate([val_leg, val_sus]),
        "horario":        np.concatenate([hor_leg, hor_sus]),
        "frequencia_dia": np.concatenate([freq_leg, freq_sus]),
        "z_score":        np.concatenate([np.abs(z_leg), z_sus]),
        "dias_cliente":   np.concatenate([dias_leg, dias_sus]),
        "pais_risco":     np.concatenate([risco_leg, risco_sus]),
        "suspeita":       [0] * n_legitimas + [1] * n_suspeitas,
    })
    df["valor_log"] = np.log1p(df["valor"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. MODELO ML — RANDOM FOREST
# ---------------------------------------------------------------------------

def treinar_modelo_fraude(df: pd.DataFrame) -> dict:
    """
    Treina e avalia um modelo Random Forest para deteccao de fraude.

    Parameters
    ----------
    df : DataFrame com colunas de features e coluna 'suspeita' (label)

    Returns
    -------
    dict com modelo treinado e metricas
    """
    FEATURES = ["valor_log", "horario", "frequencia_dia",
                "z_score", "dias_cliente", "pais_risco"]

    X = df[FEATURES]
    y = df["suspeita"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train_sc, y_train)

    y_pred  = modelo.predict(X_test_sc)
    y_proba = modelo.predict_proba(X_test_sc)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return {
        "modelo":      modelo,
        "scaler":      scaler,
        "features":    FEATURES,
        "X_test":      X_test_sc,
        "y_test":      y_test,
        "y_pred":      y_pred,
        "y_proba":     y_proba,
        "auc":         auc,
        "report":      classification_report(y_test, y_pred, target_names=["Legitima", "Suspeita"]),
        "importancias": importancias,
        "confusion":   confusion_matrix(y_test, y_pred),
    }


def isolation_forest_anomalias(df: pd.DataFrame, contaminacao: float = 0.09) -> pd.DataFrame:
    """
    Deteccao de anomalias nao supervisionada com Isolation Forest.
    Util quando nao ha labels historicos de fraude.
    """
    FEATURES = ["valor_log", "horario", "frequencia_dia", "z_score", "dias_cliente"]
    X = df[FEATURES]
    iso = IsolationForest(contamination=contaminacao, random_state=42, n_jobs=-1)
    df = df.copy()
    df["anomalia"] = iso.fit_predict(X)  # -1 = anomalia, +1 = normal
    df["score_anomalia"] = -iso.score_samples(X)  # quanto maior, mais anomalo
    return df


# ---------------------------------------------------------------------------
# 6. GRAFICOS
# ---------------------------------------------------------------------------

def grafico_distribuicao_valores(df: pd.DataFrame) -> None:
    """Histograma de valores por tipo de transacao."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, cor in [(0, "#3498db"), (1, "#e74c3c")]:
        nome = "Legitima" if label == 0 else "Suspeita"
        axes[0].hist(df[df["suspeita"] == label]["valor_log"], bins=40,
                     alpha=0.6, color=cor, label=nome, edgecolor="white")

    axes[0].set_title("Distribuicao de Valores (log) por Tipo", fontsize=12)
    axes[0].set_xlabel("log(valor)")
    axes[0].set_ylabel("Frequencia")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(df[df["suspeita"] == 0]["horario"], bins=24,
                 alpha=0.6, color="#3498db", label="Legitima", edgecolor="white")
    axes[1].hist(df[df["suspeita"] == 1]["horario"], bins=24,
                 alpha=0.6, color="#e74c3c", label="Suspeita", edgecolor="white")
    axes[1].set_title("Distribuicao por Horario", fontsize=12)
    axes[1].set_xlabel("Hora do Dia")
    axes[1].set_ylabel("Frequencia")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("grafico_distribuicao_transacoes.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_distribuicao_transacoes.png'")


def grafico_feature_importance(importancias: pd.Series) -> None:
    """Plota importancia das features do modelo."""
    fig, ax = plt.subplots(figsize=(8, 4))
    cores = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importancias)))
    importancias.plot(kind="barh", ax=ax, color=cores[::-1], edgecolor="white")
    ax.set_title("Importancia das Features — Random Forest", fontsize=12)
    ax.set_xlabel("Importancia")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(importancias):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("grafico_feature_importance.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_feature_importance.png'")


def grafico_roc(resultado: dict) -> None:
    """Plota a curva ROC do modelo."""
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        resultado["y_test"], resultado["y_proba"],
        name=f"Random Forest (AUC = {resultado['auc']:.3f})",
        ax=ax, color="steelblue"
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
    ax.set_title("Curva ROC — Deteccao de Fraude", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_roc_fraude.png", dpi=120)
    plt.show()
    print("Grafico salvo como 'grafico_roc_fraude.png'")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  MODULO 03 — PLD / AML — PREVENCAO A LAVAGEM DE DINHEIRO")
    print("=" * 65)

    # --- KYC ---
    print("\n[1] SCORE KYC — EXEMPLOS DE CLIENTES")
    clientes = [
        {"nome": "Carlos Silva",   "pep": False, "pais": "Brasil",             "ativ": "comercio",          "anos": 5, "renda": 8000,  "ocorr": 0},
        {"nome": "Ana Rodrigues",  "pep": True,  "pais": "Brasil",             "ativ": "funcao_publica",    "anos": 2, "renda": 15000, "ocorr": 0},
        {"nome": "Offshore Corp",  "pep": False, "pais": "Cayman Islands",     "ativ": "cambio",            "anos": 0, "renda": 50000, "ocorr": 2},
        {"nome": "Maria Santos",   "pep": False, "pais": "Brasil",             "ativ": "servidora_publica", "anos": 10,"renda": 5000,  "ocorr": 0},
    ]
    for c in clientes:
        r = score_kyc(c["pep"], c["pais"], c["ativ"], c["anos"], c["renda"], c["ocorr"])
        print(f"\n    {c['nome']}")
        print(f"      Score KYC: {r['score_kyc']}  |  Risco: {r['classificacao']}  |  Monitoramento: {r['monitoramento']}")

    # --- Z-score ---
    print(f"\n[2] Z-SCORE DE TRANSACOES ANOMALAS")
    casos = [
        ("Transacao normal",    5200, 5000, 1500),
        ("Alerta amarelo",      8500, 5000, 1500),
        ("Alerta VERMELHO",    16000, 5000, 1500),
        ("Transferencia enorme",50000, 5000, 1500),
    ]
    for nome, val, media, dp in casos:
        r = zscore_transacao(val, media, dp)
        print(f"\n    {nome}: R$ {val:,.0f}")
        print(f"      Z-score: {r['z_score']:.2f}  |  Nivel: {r['nivel']}  |  "
              f"Prob. normal: {r['prob_valor_normal']}")

    # --- Regras heuristicas ---
    print(f"\n[3] REGRAS HEURISTICAS AML")
    transacoes_teste = [
        {"valor": 9_500, "horario": 14, "freq": 2, "pais": "Brasil",         "pais_cli": "Brasil",         "media": 4000},
        {"valor": 9_200, "horario": 2,  "freq": 8, "pais": "Brasil",         "pais_cli": "Cayman Islands", "media": 3000},
        {"valor": 1_200, "horario": 10, "freq": 1, "pais": "Brasil",         "pais_cli": "Brasil",         "media": 1500},
        {"valor": 85_000,"horario": 23, "freq": 7, "pais": "Panama",         "pais_cli": "Brasil",         "media": 4000},
    ]
    for i, t in enumerate(transacoes_teste, 1):
        alertas = verificar_regras_aml(t["valor"], t["horario"], t["freq"],
                                       t["pais"], t["pais_cli"], t["media"])
        print(f"\n    Transacao {i} (R$ {t['valor']:,.0f} | {t['horario']}h):")
        for a in alertas:
            print(f"      -> {a}")

    # --- Dataset e ML ---
    print(f"\n[4] GERANDO DATASET SINTETICO...")
    df = gerar_dataset_transacoes(n_legitimas=3000, n_suspeitas=300)
    print(f"    Total: {len(df)} transacoes  |  "
          f"Legitimas: {(df['suspeita']==0).sum()}  |  "
          f"Suspeitas: {(df['suspeita']==1).sum()}")

    print(f"\n[5] TREINANDO MODELO RANDOM FOREST...")
    resultado = treinar_modelo_fraude(df)
    print(f"\n    AUC-ROC: {resultado['auc']:.4f}")
    print(f"\n    Relatorio de Classificacao:\n")
    print(resultado["report"])
    print(f"\n    Matriz de Confusao:")
    print(resultado["confusion"])
    print(f"\n    Top Features:")
    print(resultado["importancias"].to_string())

    # --- Isolation Forest ---
    print(f"\n[6] ISOLATION FOREST (nao supervisionado)...")
    df_iso = isolation_forest_anomalias(df)
    n_anomalos = (df_iso["anomalia"] == -1).sum()
    print(f"    Anomalias detectadas: {n_anomalos}  ({n_anomalos/len(df)*100:.1f}% do total)")
    real_suspeitas = df_iso[df_iso["suspeita"] == 1]["anomalia"].eq(-1).mean()
    print(f"    Recall de suspeitas reais: {real_suspeitas*100:.1f}%")

    # --- Graficos ---
    print(f"\n[7] GERANDO GRAFICOS...")
    grafico_distribuicao_valores(df)
    grafico_feature_importance(resultado["importancias"])
    grafico_roc(resultado)

    print("\n" + "=" * 65)
    print("  Modulo 03 concluido com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
