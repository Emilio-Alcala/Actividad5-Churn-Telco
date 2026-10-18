# CHANGELOG

## [v1.1-modelos] - 2026-10-18
### Agregado
- `fuentes/train.py`: Grid Search (5-fold estratificado, AUC) para regresión logística y random forest, con registro en MLflow.
- Notebook `entrena.ipynb`: EDA, pruebas estadísticas (t pareada, Wilcoxon, Nadeau-Bengio) y comparación de modelos.
- `evidencias/`: gráficas, puntajes de validación cruzada y tabla comparativa.

## [v1.0-datos] - 2026-10-17
### Agregado
- Dataset original Telco Customer Churn en `datos/datos_ini/` (sin modificar).
- `fuentes/datos_prep.py`: limpieza y estandarización reproducibles.
- `datos/datos_limp/telco_churn_limpio.csv` y `reporte_calidad.md`.
### Cambiado (respecto al original)
- TotalCharges a numérico (11 vacíos con tenure = 0 imputados con 0).
- SeniorCitizen 0/1 a No/Yes; categorías "No internet/phone service" a "No".
- Churn a 0/1.