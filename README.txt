# Garden Coffee — Identificación y priorización de mercados de exportación
### Reto técnico ML Engineering · Metodología CRISP-DM

| Fase CRISP-DM | Estado |
|---|---|
| 1. Entendimiento del negocio | ✅ Desarrollada |
| 2. Entendimiento de los datos | ✅ Desarrollada |
| 3. Preparación de los datos | ✅ Implementada (tabla analítica lista) |
| 4. Modelado | ⏳ Pendiente (diseño propuesto) |
| 5. Evaluación | ⏳ Pendiente (criterios definidos) |
| 6. Despliegue | ⏳ Pendiente (incluye propuesta GenAI) |

**Archivos de entrada** (en `DATA_DIR`):
- `coffee_db.parquet`: consumo doméstico (entregado con el reto).
- `coffee-production-by-exporting-countries-1991-2020-ico.csv`: producción total (Kaggle, datos ICO).

**Archivo de salida:** `coffee_balance_long.parquet`, la tabla analítica que usan las fases 4 y 5.