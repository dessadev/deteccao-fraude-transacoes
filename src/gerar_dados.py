"""Gera uma base sintética de transações de cartão com fraude rara.

Os dados são simulados: nenhum dado real de cliente ou instituição é usado.
"""
import numpy as np
import pandas as pd


def gerar_transacoes(n=60_000, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.date_range("2025-01-01", periods=180, freq="D")
    df = pd.DataFrame({
        "data": rng.choice(data, n),
        "hora": rng.integers(0, 24, n),
        "valor": np.round(rng.lognormal(4.0, 1.0, n), 2),
        "distancia_casa_km": np.round(rng.exponential(8, n), 1),
        "dispositivo_novo": rng.binomial(1, 0.05, n),
        "transacoes_ultima_hora": rng.poisson(0.6, n),
        "pais_diferente": rng.binomial(1, 0.02, n),
        "canal": rng.choice(["presencial", "online", "app"], n, p=[0.45, 0.35, 0.20]),
    })
    media_cliente = rng.lognormal(4.0, 0.6, n)
    df["valor_vs_media_cliente"] = np.round(df["valor"] / media_cliente, 2)

    madrugada = df["hora"].between(0, 5).astype(int)
    online = (df["canal"] == "online").astype(int)
    valor_alto = (df["valor_vs_media_cliente"] > 3).astype(int)
    logit = (
        -9.3
        + 1.6 * np.log1p(df["valor_vs_media_cliente"])
        + 0.05 * df["distancia_casa_km"].clip(upper=80)
        + 1.5 * df["dispositivo_novo"]
        + 0.7 * df["transacoes_ultima_hora"]
        + 2.4 * df["pais_diferente"]
        + 1.0 * madrugada
        + 0.6 * online
        + 2.2 * df["dispositivo_novo"] * valor_alto
        + 1.4 * madrugada * online
        + rng.normal(0, 0.3, n)
    )
    df["fraude"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df.sort_values("data").reset_index(drop=True)


if __name__ == "__main__":
    df = gerar_transacoes()
    df.to_csv("transacoes_sinteticas.csv", index=False)
    print(df.shape, df["fraude"].mean())
