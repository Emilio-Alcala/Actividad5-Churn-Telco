# Actividad 5 – Predicción de churn (Telco Customer Churn)

Proyecto de la materia Gestión de Proyectos de Inteligencia Artificial (Tecmilenio). 

Continúa el business case de la Fase I (Sierra Móvil): clasificar clientes con alta probabilidad de baja y priorizar el 15% de mayor riesgo.

## Dataset
- Fuente: IBM Telco Customer Churn (Kaggle, `blastchar`).
- 7,043 clientes, 21 columnas; objetivo `Churn` (26.5% de bajas).
- Original en `datos/datos_ini/` (no se modifica)
- limpio en `datos/datos_limp/`

## Estructura
```
datos/datos_ini/     datos originales
datos/datos_limp/    datos limpios + reporte_calidad.md
fuentes/datos_prep.py  limpieza
fuentes/train.py       entrenamiento, Grid Search y registro en MLflow
fuentes/entrena.ipynb  EDA, pruebas estadísticas y comparación
evidencias/            gráficas, resultados, capturas de MLflow
CHANGELOG.md           versiones del proyecto y de los datos
```

## Cómo reproducir
```powershell
(terminal de powershell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python fuentes\datos_prep.py
python fuentes\train.py
```
```powershell
(terminal adicional de powershell)
mlflow ui --backend-store-uri sqlite:///mlflow.db   

# En el navegador --->  http://localhost:5000
```

Luego ejecutar `fuentes/entrena.ipynb` (de arriba abajo).

## Decisiones principales
- Limpieza: TotalCharges a numérico (11 vacíos con tenure=0 imputados con 0), categorías redundantes colapsadas, Churn a 0/1.
- Escalado y one-hot dentro de un Pipeline (evita fuga de datos).
- Partición 80/20 estratificada, semilla 42.
- Modelos: regresión logística y random forest; Grid Search con 5-fold estratificado y métrica ROC-AUC.
- Métricas: accuracy, precision, recall, F1, ROC-AUC, precision@15% y recall@15%.
- Comparación: prueba t pareada, Wilcoxon y t corregida (Nadeau-Bengio) sobre 15 puntajes de validación cruzada.

## Resultados (conjunto de prueba)
| Métrica | Reg. logística | Random forest |
|---|---|---|
| ROC-AUC | 0.8415 | 0.8429 |
| Precision@15% | 0.7123 | 0.6745 |
| Recall | 0.5588 | 0.4144 |

Sin diferencia significativa en AUC (p > 0.05). Se elige regresión logística por interpretabilidad y simplicidad.

## Versiones
Ver `CHANGELOG.md`. 

Etiquetas Git: 
- `v1.0-datos` 
- `v1.1-modelos`