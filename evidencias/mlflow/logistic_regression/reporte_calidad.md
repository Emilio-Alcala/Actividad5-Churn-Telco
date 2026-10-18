# Reporte de calidad de datos - Telco Customer Churn

| Métrica | Antes | Después |
|---|---|---|
| Filas | 7043 | 7043 |
| Columnas | 21 | 21 |
| customerID duplicados | 0 | 0 |
| Filas duplicadas completas | 0 | 0 |
| Valores nulos o vacíos | 11 | 0 |
| Tipo de TotalCharges | str | float64 |
| Valores únicos en MultipleLines | 3 | 2 |
| Valores únicos en OnlineSecurity | 3 | 2 |
| Tasa de churn | 0.2654 | 0.2654 |

## Transformaciones aplicadas
- Eliminación de espacios sobrantes en columnas de texto.
- Deduplicación por customerID.
- TotalCharges convertido a numérico; 11 vacíos (tenure = 0) imputados con 0 por regla de negocio (cliente aún sin facturar).
- SeniorCitizen estandarizado de 0/1 a No/Yes.
- 'No internet service' y 'No phone service' colapsados a 'No'.
- Churn convertido a 0/1 (1 = el cliente se dio de baja).
- Escalado y codificación: se aplican dentro del Pipeline de entrenamiento (train.py) para evitar fuga de datos.