"""Entrenamiento, ajuste de hiperparámetros y registro en MLflow."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "datos" / "datos_limp" / "telco_churn_limpio.csv"
EVIDENCIAS = RAIZ / "evidencias"
GRAFICAS = EVIDENCIAS / "graficas"

SEMILLA = 42
K_TOP = 0.15                      # fracción de mayor riesgo (Fase I)
EXPERIMENTO = "Churn_Telco_Sierra_Movil"
DATA_VERSION = "v1.0-datos"       # etiqueta de Git de los datos usados
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

MODELOS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=SEMILLA),
        {"clf__C": [0.01, 0.1, 1, 10],
         "clf__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEMILLA),
        {"clf__n_estimators": [200, 400],
         "clf__max_depth": [5, 10, None],
         "clf__min_samples_leaf": [1, 5]},
    ),
}


def cargar_datos():
    df = pd.read_csv(RUTA_DATOS)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y,
                            random_state=SEMILLA)


def crear_pipeline(clf, X):
    categoricas = [c for c in X.columns if c not in NUMERICAS]
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def precision_recall_at_k(y_true, proba, k=K_TOP):
    """Precisión y recall dentro del k% de clientes con mayor probabilidad."""
    y_true = np.asarray(y_true)
    n = int(np.ceil(len(y_true) * k))
    top = np.argsort(proba)[::-1][:n]
    tp = y_true[top].sum()
    return tp / n, tp / y_true.sum()


def evaluar(modelo, X_test, y_test):
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    p_k, r_k = precision_recall_at_k(y_test, proba)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_precision_at_15": p_k,
        "test_recall_at_15": r_k,
    }


def entrenar_y_registrar(nombre, clf, grid, X_train, X_test, y_train, y_test):
    pipe = crear_pipeline(clf, X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)

    with mlflow.start_run(run_name=nombre):
        mlflow.set_tags({"dataset": "Telco Customer Churn (IBM)",
                         "data_version": DATA_VERSION,
                         "actividad": "Actividad 5"})
        gs.fit(X_train, y_train)
        mejor = gs.best_estimator_

        # Parámetros
        mlflow.log_params({k.replace("clf__", ""): v
                            for k, v in gs.best_params_.items()})
        mlflow.log_params({"cv_folds": 5, "scoring": "roc_auc",
                           "semilla": SEMILLA, "top_k": K_TOP,
                           "n_train": len(X_train), "n_test": len(X_test),
                           "espacio_busqueda": str(grid)})

        # Métricas
        mlflow.log_metric("cv_roc_auc_mean", gs.best_score_)
        mlflow.log_metric(
            "cv_roc_auc_std",
            gs.cv_results_["std_test_score"][gs.best_index_])
        metricas = evaluar(mejor, X_test, y_test)
        mlflow.log_metrics(metricas)

        # 15 puntajes de AUC (5-fold x 3 repeticiones) para pruebas estadísticas
        cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                         random_state=SEMILLA)
        scores = cross_val_score(mejor, X_train, y_train, cv=cv_rep,
                                 scoring="roc_auc", n_jobs=-1)
        EVIDENCIAS.mkdir(exist_ok=True)
        ruta_cv = EVIDENCIAS / f"cv_scores_{nombre}.csv"
        pd.DataFrame({"roc_auc": scores}).to_csv(ruta_cv, index=False)
        mlflow.log_artifact(str(ruta_cv))

        # Resultados completos del Grid Search
        res = pd.DataFrame(gs.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        res["params"] = res["params"].astype(str)
        mlflow.log_table(res, "grid_search_results.json")

        # Gráficas
        GRAFICAS.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_estimator(mejor, X_test, y_test, ax=ax)
        ax.set_title(f"Matriz de confusión - {nombre}")
        fig.tight_layout()
        fig.savefig(GRAFICAS / f"cm_{nombre}.png", dpi=150)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(mejor, X_test, y_test, ax=ax)
        ax.set_title(f"Curva ROC - {nombre}")
        fig.tight_layout()
        fig.savefig(GRAFICAS / f"roc_{nombre}.png", dpi=150)
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        # Reporte de calidad de datos y modelo entrenado
        mlflow.log_artifact(str(RAIZ / "datos" / "datos_limp"
                                    / "reporte_calidad.md"))
        firma = infer_signature(X_test, mejor.predict_proba(X_test))
        
        # Corrección aplicada aquí: se añade skops_trusted_types y se usa artifact_path
        mlflow.sklearn.log_model(
            mejor, 
            artifact_path="model", 
            signature=firma,
            input_example=X_test.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

    return {"modelo": nombre, **gs.best_params_,
            "cv_roc_auc_mean": gs.best_score_, **metricas}

def main():
    mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENTO)
    X_train, X_test, y_train, y_test = cargar_datos()
    filas = []
    for nombre, (clf, grid) in MODELOS.items():
        print(f"Entrenando {nombre}...")
        filas.append(entrenar_y_registrar(nombre, clf, grid,
                                          X_train, X_test, y_train, y_test))
    resultados = pd.DataFrame(filas)
    resultados.to_csv(EVIDENCIAS / "resultados_modelos.csv", index=False)
    pd.set_option("display.width", 200)
    print(resultados.round(4).T)


if __name__ == "__main__":
    main()