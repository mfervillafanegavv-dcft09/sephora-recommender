# Preparación Nexa v2
Separación global por fecha: primeros ~80% train, siguientes ~10% validación, últimos ~10% test. Cortes iguales permanecen en un período. Por período se conserva la última reseña por usuario-producto.
Perfiles y estadísticas calculados únicamente en train. Validación permite elegir parámetros; test queda reservado. No se utilizan helpfulness, rating/reviews del catálogo ni loves_count en las estadísticas.
Los objetivos de evaluación son productos nuevos para usuarios conocidos en train. Se conservan productos sin historial para contabilizar fallos de cobertura. Los usuarios nuevos se contabilizan por separado y necesitan evaluación de cold-start.
Propuesta de ranking: media ajustada (n*media + m*media_global)/(n+m). Versión por piel: combinar media del grupo con el ranking general, peso n_grupo/(n_grupo+m_piel). Elegir m y m_piel en validación.
Filtrar candidatos y objetivos con la misma categoría y presupuesto; excluir productos ya reseñados en train. Precio del catálogo en USD; presupuestos son escenarios simulados. No condicionar el presupuesto al precio del producto objetivo. Reportar cuántos objetivos quedan fuera por presupuesto, categoría o falta de soporte.
Para la prueba con presupuesto, solo objetivos positivos (4-5) que cumplen el filtro forman el denominador de Recall@10. Medir también NDCG@10, cobertura y tamaño de cohortes; productos no reseñados tienen relevancia desconocida, no negativa confirmada.
No usar stock de la captura para simular stock histórico. Una comparación con catálogo actual es retrospectiva y condicionada a esa captura, no un backtest fiel del catálogo histórico.
Entrenamiento y evaluación no deben reutilizar conclusiones del EDA como parámetros optimizados en test. El EDA previo vio el conjunto completo: el test no es evidencia completamente independiente de la selección del enfoque.
Estos archivos preparan la comparación; todavía no demuestran mejora del modelo ni ventas.
