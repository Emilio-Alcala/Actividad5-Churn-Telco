# datos_prep.py
"""Limpieza y estandarización del dataset Telco Customer Churn."""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_INI = RAIZ / "datos" / "datos_ini" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RUTA_LIMP = RAIZ / "datos" / "datos_limp" / "telco_churn_limpio.csv"
RUTA_REPORTE = RAIZ / "datos" / "datos_limp" / "reporte_calidad.md"

COLS_INTERNET = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]


def cargar_datos_originales(ruta=RUTA_INI):
    return pd.read_csv(ruta)


def metricas_calidad(df):
    """Métricas de calidad para comparar antes y después."""
    vacios = df.replace(r"^\s*$", pd.NA, regex=True)
    
    # Cálculo seguro de la tasa de churn según el tipo de dato (numérico o texto)
    if pd.api.types.is_numeric_dtype(df["Churn"]):
        tasa_churn = float(df["Churn"].mean())
    else:
        tasa_churn = float((df["Churn"] == "Yes").mean())

    return {
        "Filas": len(df),
        "Columnas": df.shape[1],
        "customerID duplicados": int(df["customerID"].duplicated().sum()),
        "Filas duplicadas completas": int(df.duplicated().sum()),
        "Valores nulos o vacíos": int(vacios.isna().sum().sum()),
        "Tipo de TotalCharges": str(df["TotalCharges"].dtype),
        "Valores únicos en MultipleLines": int(df["MultipleLines"].nunique()),
        "Valores únicos en OnlineSecurity": int(df["OnlineSecurity"].nunique()),
        "Tasa de churn": round(tasa_churn, 4),
    }


def limpiar_datos(df):
    df = df.copy()
    # 1. Quitar espacios sobrantes en texto (compatible con versiones modernas de pandas)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    # 2. Deduplicar por identificador
    df = df.drop_duplicates(subset="customerID", keep="first")
    # 3. TotalCharges: texto -> número; vacíos con tenure 0 -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    sin_facturar = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[sin_facturar, "TotalCharges"] = 0.0
    assert df["TotalCharges"].isna().sum() == 0, "Quedan nulos sin explicar"
    # 4. SeniorCitizen 0/1 -> No/Yes (consistente con el resto)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # 5. Categorías redundantes
    for c in COLS_INTERNET:
        df[c] = df[c].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    # 6. Variable objetivo a 0/1 de forma explícita y numérica
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").astype("int8")
    return df.reset_index(drop=True)


def escribir_reporte(antes, despues, ruta=RUTA_REPORTE):
    lineas = ["# Reporte de calidad de datos - Telco Customer Churn", "",
              "| Métrica | Antes | Después |", "|---|---|---|"]
    for k in antes:
        lineas.append(f"| {k} | {antes[k]} | {despues[k]} |")
    lineas += ["", "## Transformaciones aplicadas",
               "- Eliminación de espacios sobrantes en columnas de texto.",
               "- Deduplicación por customerID.",
               "- TotalCharges convertido a numérico; 11 vacíos (tenure = 0) "
               "imputados con 0 por regla de negocio (cliente aún sin facturar).",
               "- SeniorCitizen estandarizado de 0/1 a No/Yes.",
               "- 'No internet service' y 'No phone service' colapsados a 'No'.",
               "- Churn convertido a 0/1 (1 = el cliente se dio de baja).",
               "- Escalado y codificación: se aplican dentro del Pipeline de "
               "entrenamiento (train.py) para evitar fuga de datos."]
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def main():
    crudo = cargar_datos_originales()
    limpio = limpiar_datos(crudo)
    RUTA_LIMP.parent.mkdir(parents=True, exist_ok=True)
    limpio.to_csv(RUTA_LIMP, index=False)
    antes, despues = metricas_calidad(crudo), metricas_calidad(limpio)
    escribir_reporte(antes, despues)
    for k in antes:
        print(f"{k}: {antes[k]} -> {despues[k]}")
    print(f"Guardado: {RUTA_LIMP}")


if __name__ == "__main__":
    main()