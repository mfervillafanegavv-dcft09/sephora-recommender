# Sephora Skincare Recommender

**NEXA Data Solutions**

Sistema de recomendación de productos de skincare para **usuarios nuevos sin historial (cold start)**, construido sobre el dataset público de productos y reseñas de Sephora. La persona consulta aportando solo su perfil (tipo de piel, tono de piel, color de ojos, color de cabello) y el sistema devuelve un Top-10 de productos (planteamiento de la DEMO 1). La **solución final** del proyecto es **Nexa v4**, que recomienda por categoría y presupuesto y personaliza con el historial del usuario cuando existe. El modelo **A2** (perfil + popularidad) se conserva como **línea alternativa previa**.

> **Estado del documento:** cubre el antecedente en Power BI, la **DEMO 1** completa y la **DEMO 2** (Etapas 2 y 3, Nexa v4 con su API y monitoreo, y la línea alternativa A2), según el contenido de las ramas `Fernanda` y `main` del repositorio al 15 de octubre de 2026. Los puntos abiertos están en [Pendientes por confirmar](#pendientes-por-confirmar).

---

## Tabla de contenidos

1. [Antecedente — Exploración del dataset en Power BI](#antecedente--exploración-del-dataset-en-power-bi)
2. [Resumen de la DEMO 1](#resumen-de-la-demo-1)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Datos](#datos)
5. [Requisitos e instalación](#requisitos-e-instalación)
6. [Cómo ejecutar el pipeline](#cómo-ejecutar-el-pipeline)
7. [Pipeline paso a paso](#pipeline-paso-a-paso)
8. [Modelos y protocolo de evaluación](#modelos-y-protocolo-de-evaluación)
9. [Resultados](#resultados)
10. [Limitaciones detectadas al cierre de la DEMO 1](#limitaciones-detectadas-al-cierre-de-la-demo-1)
11. [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
12. [DEMO 2](#demo-2)
13. [Changelog DEMO 1 → DEMO 2](#changelog-demo-1--demo-2)
14. [Pendientes por confirmar](#pendientes-por-confirmar)
15. [Equipo](#equipo)

---

## Antecedente — Exploración del dataset en Power BI

Antes de la DEMO 1, el equipo exploró el dataset en un tablero de Power BI (`Sephora_PBI.pbix`) con dos páginas. Sus cifras coinciden con las del EDA de la DEMO 1 y fundamentan el planteamiento del proyecto.

### Página 1 — Catálogo y comportamiento de productos

![Tablero de Power BI: catálogo y comportamiento de productos](reports/figures/powerbi_catalogo_productos.jpg)

- **8.494 productos** en el catálogo y **1.093.855 interacciones** (reseñas).
- **Cobertura de reseñas: 27,68 %.** 2,35 mil productos tienen reseñas y 6,14 mil (**72,32 %**) no tienen ninguna.
- **Categorías:** Skincare y Makeup son las más grandes (~2,4 mil productos cada una), seguidas de Hair y Fragrance (~1,4 mil).
- **Marcas con más productos:** SEPHORA COLLECTION, Dior, Bumble and bumble, Kérastase, tarte, TOM FORD, CLINIQUE, NEST New York y Charlotte Tilbury, entre otras.
- **Precio:** la mayoría de los productos cuesta entre USD 25 y 50 (~3,9 mil), seguidos de los de 0 a 25 (~2 mil); los de más de USD 100 son unos 0,95 mil.
- **Valoración promedio de Skincare:** ~4,3.
- Los filtros disponibles son rango de precio, marca y categoría primaria.

### Página 2 — Usuarios y recomendación

![Tablero de Power BI: usuarios y recomendación](reports/figures/powerbi_usuarios_recomendacion.jpg)

- **503.196 usuarios únicos** (el EDA de la DEMO 1 reporta 503.216; ver [Pendientes](#pendientes-por-confirmar)).
- **Escasez de la matriz usuario–producto: 99,9075 %.** Promedio de valoración: **4,30**.
- **Tipo de piel:** combination 49,3 %, dry 16,2 %, normal 12,2 % y oily 11,3 %.
- **Tono de piel:** light 23,2 %, fair 18,8 % y lightMedium 18,0 %, entre otros.
- **Color de ojos:** brown 40,5 %, blue 15,3 % y hazel, entre otros. **Color de cabello:** brown 36,4 %, black 23,3 % y blonde 17,8 %.
- **Valoraciones:** concentradas en 4 y 5 estrellas. **Recomendación:** 71,1 % de las reseñas recomienda el producto, 13,6 % no lo recomienda y 15,3 % no tiene dato.

**Hallazgos clave del tablero:**

1. El **72,32 % del catálogo carece de reseñas**, lo que plantea un desafío importante para los productos.
2. La **escasez de datos** en la matriz usuario–producto (99,9075 %) limita la eficacia de los enfoques puramente colaborativos.
3. Aproximadamente el **58,19 % de los usuarios registra una única interacción**, lo que da información histórica limitada para la personalización.

Estos tres hallazgos son el punto de partida del planteamiento de cold start de la DEMO 1.

---

## Resumen de la DEMO 1

| Etapa | Qué hace | Script / notebook |
|---|---|---|
| 1. EDA y limpieza | Audita, limpia y analiza productos y reseñas; genera tablas, figuras y un reporte HTML | `Sephora_EDA.py` |
| 2. Feature engineering | TF-IDF de productos, One-Hot de perfiles, filtrado de interacciones positivas | `Sephora_Features.py` |
| 3. Modelado | Cuatro recomendadores + baseline de popularidad, evaluados con el mismo protocolo | `notebooks/01` a `04` + `sephora_common.py` |
| 4. Comparación | Tabla final, bootstrap de la diferencia contra el baseline | `notebooks/05_comparacion_final.ipynb` |

**Cifras clave del dataset (post-limpieza):**

- 8.494 productos en catálogo; solo 2.351 (27,7 %) tienen reseñas.
- 1.093.855 reseñas de 503.216 usuarios únicos.
- Sparsity de la matriz usuario–producto: **99,91 %**.
- 58,2 % de los usuarios tiene una sola reseña; 75,1 % tiene el perfil completo.

---

## Estructura del repositorio

### Ramas

| Rama | Contenido | Estado |
|---|---|---|
| `Fernanda` | Proyecto completo: notebooks de las Etapas 1 y 2, scripts de la Etapa 3 y resultados, EDA y Feature Engineering, modelo y motor Nexa v4, app Streamlit, artefactos y validación automática (GitHub Actions) | La más completa; no incluye la API ni el monitoreo |
| `main` | API de Nexa v4 (FastAPI), monitoreo y registro de inferencias, `Sephora_Deploy.py` (CLI de A2) y `Sephora_EDA.py` | Incluye la integración del PR #1; **no incluye** el motor, el modelo ni los notebooks |
| `martinpa-spec-patch-1` | API y procesamiento de datos históricos (monitoreo) | Integrada en `main` mediante el PR #1 |
| `martinpa-spec-upload-1` | Misma base que la anterior, con el `README.md` reemplazado por el texto de la carpeta `app/` y sin `nexa_streamlit.py` | **No integrada**; ver [Pendientes](#pendientes-por-confirmar) |

`main` y `Fernanda` aún no están unificadas: la API vive en `main` y el modelo, el motor y la interfaz que la consume, en `Fernanda`.

### Estructura (rama `Fernanda`, con la API de `main`)

```
sephora-recommender/
├── .github/workflows/validacion.yml   # Verifica la sintaxis de src/ y app/ en cada push
├── api/main.py                        # API FastAPI de Nexa v4 (rama main)
├── app/
│   ├── nexa_deploy.py                 # CLI de Nexa v4
│   ├── nexa_streamlit.py              # Interfaz Streamlit (consume la API)
│   └── assets/                        # Logo e ícono de NEXA
├── artifacts/Etapa_01/, Etapa_02/     # Resultados exportados por los notebooks (JSON y CSV)
├── data/
│   ├── raw/                           # Datos originales (no versionados)
│   └── processed/                     # Datos procesados (no versionados)
├── models/nexa_v4/                    # Modelo Nexa v4 entrenado y guardado
├── notebooks/
│   ├── Etapa_01/                      # DEMO 1: modelos A–D y comparación (+ sephora_common.py)
│   └── Etapa_02/                      # DEMO 2: experimentos A1–A5 y diagnóstico (+ sephora_common.py)
├── reports/
│   ├── Etapa_03/                      # Resultados de la Etapa 3 (diagnóstico, M1, M2, auditoría de Nexa)
│   └── registro.md
├── src/
│   ├── Sephora_EDA.py                 # EDA y limpieza (DEMO 1)
│   ├── Sephora_EDA_complementario.py  # EDA complementario: asociaciones descriptivas, sin entrenar modelos
│   ├── Sephora_Features.py            # Feature Engineering (DEMO 1)
│   ├── E3_preparacion_datos.py        # Etapa 3: split cronológico, catálogo y estadísticas
│   ├── E3_diagnostico_inicial.py      # Etapa 3: ranking general vs. ranking por tipo de piel
│   ├── E3_M1_hibrido.py               # Etapa 3: Nexa v3 (ranking general + contenido, peso fijo)
│   ├── E3_M2_adaptativo.py            # Etapa 3: Nexa v4 (peso de contenido adaptativo)
│   └── nexa_recommender.py            # Clase NexaRecommender (build, save, load, recommend)
├── Sephora_Monitoring.py              # Monitoreo de producción (rama main)
├── README_Monitoring.md               # Guía de monitoreo (rama main)
├── Sephora_Deploy.py                  # CLI de A2 (rama main)
├── tests/
└── requirements.txt
```

`.gitignore` excluye `data/raw/*`, `data/processed/*` y los registros de ejecución de Nexa (`artifacts/nexa_deploy/*.log` y `*.jsonl`). El tablero `Sephora_PBI.pbix` (26 MB) vive en la carpeta local del proyecto y **aún no está en el repositorio**.

---

## Datos

**Fuente:** dataset de Sephora (productos y reseñas), 6 archivos CSV (en la carpeta local de la DEMO 1, `Sephora_database/`; en el repositorio, `data/raw/`):

| Archivo | Tamaño aprox. | Filas |
|---|---|---|
| `product_info.csv` | 7,9 MB | 8.494 |
| `reviews_0-250.csv` | 282 MB | 602.130 |
| `reviews_250-500.csv` | 100 MB | 206.725 |
| `reviews_500-750.csv` | 56 MB | 116.262 |
| `reviews_750-1250.csv` | 58 MB | 119.317 |
| `reviews_1250-end.csv` | 24 MB | 49.977 |

> Los CSV de reseñas pesan ~520 MB y **no están en el repositorio**. Descárgalos de la fuente original y colócalos en `data/raw/`. **TODO: añadir enlace de descarga.**

**Datasets procesados** (generados por `Sephora_EDA.py` en `data/processed/`, en CSV y Parquet): `products_clean`, `reviews_clean`, `interactions`, `user_profiles`, `product_review_coverage`, `category_analysis`.

**Datasets de features** (generados por `Sephora_Features.py` en `data/features/`): `products_id_index.parquet`, `users_features.parquet`, `positive_interactions.parquet`.

---

## Requisitos e instalación

- Python 3.14 (el protocolo guardó estas versiones: numpy 2.4.6, pandas 3.0.3, scipy 1.18.1, scikit-learn 1.9.0)
- Además: `matplotlib`, `pyarrow`, `jupyter`

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib pyarrow jupyter
```

**TODO:** generar un `requirements.txt` con `pip freeze` desde el entorno que usó el equipo.

---

## Cómo ejecutar el pipeline

**DEMO 1** (desde la raíz del proyecto, en este orden):

```bash
python src/Sephora_EDA.py        # 1. limpia datos y genera data/processed
python src/Sephora_Features.py   # 2. genera data/features
# 3. ejecutar los notebooks de notebooks/Etapa_01 de arriba a abajo (modelos A–D y comparación)
```

(En la carpeta local de la DEMO 1 los scripts están en la raíz: `python Sephora_EDA.py`.)

**DEMO 2:**

```bash
# Etapa 2: ejecutar los notebooks de notebooks/Etapa_02 (A1 → A5 y diagnóstico)
python src/E3_preparacion_datos.py --project-root .   # Etapa 3: prepara datos
python src/E3_diagnostico_inicial.py --project-root .
python src/E3_M1_hibrido.py --project-root .
python src/E3_M2_adaptativo.py --project-root .
```

Notas:

- Los notebooks leen `data/processed/` (`reviews_clean.parquet`, `products_clean.parquet`) y localizan la raíz del proyecto buscando esa carpeta; `sephora_common.py` debe estar junto a ellos.
- Todos los notebooks de un mismo protocolo deben usar los mismos valores de `N_VALIDACION`, `N_TEST` y `SEMILLA` (500, 500 y 42).
- Los scripts de la Etapa 3 requieren `pandas`, `numpy`, `scipy`, `scikit-learn` y `pyarrow` (si se leen Parquet), y los datos de `data/processed/` generados por `src/Sephora_EDA.py`.
- `Sephora_Features.py` no es requisito de los notebooks (ver [Limitaciones](#limitaciones-y-trabajo-futuro)).

---

## Pipeline paso a paso

### 1. EDA y limpieza — `Sephora_EDA.py`

1. **Carga** del catálogo y de los 5 archivos de reseñas (acepta nombres alternativos tipo `archivo(1).csv`).
2. **Auditoría inicial** antes de transformar: resumen del dataset, estadísticos descriptivos, valores faltantes.
3. **Limpieza**
   - Productos: eliminación de duplicados exactos (0 encontrados).
   - Reseñas: se eliminan 224 duplicados exactos y 332 reseñas con mismo autor, producto y texto; se normaliza `grey` → `gray` en color de ojos. Los valores faltantes **no se imputan**: quedan explícitos.
4. **Datasets derivados:** cobertura producto–reseña, análisis por categoría, perfiles de usuario, tabla de interacciones.
5. **Análisis:** outliers, distribución de ratings y recomendaciones, long tail de productos y usuarios, sparsity, cold start.
6. **Salidas:** 21 figuras, tablas CSV, diccionario de datos, `eda_report.html` y `pipeline_summary.json`.

### 2. Feature engineering — `Sephora_Features.py`

1. **Productos → TF-IDF:** se concatenan `primary_category`, `highlights` e `ingredients` y se vectorizan en 500 palabras clave (inglés, sin stop words). Se guarda el orden de `product_id` en `products_id_index.parquet`.
2. **Usuarios → One-Hot:** `skin_type`, `skin_tone`, `eye_color`, `hair_color` (nulos como `Unknown`) → `users_features.parquet`.
3. **Interacciones positivas:** reseñas con `rating >= 4` **o** `is_recommended == True`, deduplicadas por (autor, producto) → `positive_interactions.parquet`.

> Nota: la matriz TF-IDF se calcula pero **no se guarda en disco**; solo se guarda el índice de productos.

---

## Modelos y protocolo de evaluación

### Protocolo común (`cold-start-users-v2.0`, semilla 42)

- **Catálogo candidato:** solo productos con `primary_category = Skincare` (2.420).
- **Positivo:** `rating >= 4` o `is_recommended = True`.
- **Split por usuario:** 500 usuarios de validación y 500 de test se excluyen por completo del entrenamiento (502.216 usuarios y 1.084.636 reseñas en train). Solo son elegibles usuarios con perfil completo (tipo y tono de piel) y al menos 2 positivos (161.396 elegibles).
- **Consulta:** únicamente el perfil. Ni ratings ni compras del usuario de test se usan como entrada.
- **Evaluación:** se oculta un producto positivo y se mide si aparece en el Top-10.
- **Métricas:** Hit Rate@10, NDCG@10, cobertura de usuarios, Top-10 completo, cobertura de catálogo.
- **Reproducibilidad:** `protocolo.json` guarda el ID del protocolo (hash SHA-256), versiones de librerías y los casos de validación/test.

### Los cuatro modelos y el baseline

| Modelo | Idea | Qué usa del perfil | Hiperparámetro ajustado |
|---|---|---|---|
| **Baseline** — Popularidad | Productos con más reseñas positivas | Nada | – |
| **A. Vecinos por perfil** | One-Hot de los 4 atributos → coseno → K vecinos → votos positivos | Los 4 atributos | `k` |
| **B. Satisfacción por segmento** | Rating medio suavizado por producto dentro del mismo tipo de piel | Tipo de piel | `m` (suavizado) |
| **C. Similitud ponderada** | Cada reseña pesa según coincidencia de tipo y tono de piel; rating ponderado y suavizado | Tipo y tono de piel | pesos `a`/`b` y `m` |
| **D. Afinidad supervisada** | Regresión logística con perfil, marca, categorías, precio, TF-IDF y cruces piel–categoría | Perfil + atributos de producto | `C` |

Los hiperparámetros se eligen en validación; el test se evalúa una sola vez.

---

## Resultados

Evaluación en test (500 usuarios, un producto oculto por usuario):

| Modelo | Config. elegida | Hits | HR@10 | NDCG@10 | Cobertura de catálogo |
|---|---|---|---|---|---|
| **Baseline popularidad** | – | 37 | **7,4 %** | **0,0332** | 0,4 % |
| A. Vecinos por perfil | k = 250 | 29 | 5,8 % | 0,0275 | 10,6 % |
| B. Satisfacción por segmento | m = 100 | 9 | 1,8 % | 0,0074 | 0,7 % |
| C. Similitud ponderada | 0,5 / 0,5, m = 50 | 8 | 1,6 % | 0,0068 | 1,4 % |
| D. Afinidad supervisada | C = 0,1 | 3 | 0,6 % | 0,0021 | 1,0 % |

Los cinco devuelven Top-10 completo al 100 % de los usuarios.

**Lectura de los resultados**

- **Ningún modelo basado en perfil supera al baseline de popularidad** en HR@10 ni NDCG@10. El mejor (A) queda cerca (5,8 % vs 7,4 %), pero con solo 500 usuarios de test la diferencia es poco estable (≈ 8 hits de diferencia).
- El modelo A recomienda mucha más variedad de catálogo (10,6 % vs 0,4 %), lo que puede ser valioso aunque cueste algo de acierto.
- Con 4 atributos de perfil, la señal para predecir qué producto va a consumir alguien es débil; la popularidad ya captura gran parte de lo que se puede acertar.
- El HR@10 mide si se recupera un producto que la persona *efectivamente reseñó*, no si el producto es adecuado para su piel.

**Artefactos:** `artifacts/comparacion_4_modelos/<protocol_id>/` contiene, por modelo, el JSON de validación y test, el detalle por usuario, un ejemplo de Top-10 y `comparacion_final.csv`.

---

## Limitaciones detectadas al cierre de la DEMO 1

Al evaluar los resultados, el equipo de NEXA Data Solutions identificó que el sistema aún no alcanza el desempeño esperado para un recomendador de productos:

- **Las métricas de los 4 modelos (HR@10 y NDCG@10) quedaron por debajo del baseline de popularidad** y por debajo del rango esperable para este tipo de sistema (baseline 7,4 % de HR@10 frente a 5,8 % del mejor modelo y 0,6 % del peor; ver [Resultados](#resultados)).
- **La señal del perfil es débil:** con solo 4 atributos de perfil y una matriz con 99,91 % de sparsity, los modelos basados únicamente en perfil aportan poca información frente a la popularidad.

**Punto de partida para la DEMO 2:** mejorar estas métricas es el objetivo principal de la siguiente etapa. Los cambios aplicados y sus resultados se documentan en la sección [DEMO 2](#demo-2).

---

## Limitaciones y trabajo futuro

- **Sesgo de selección:** solo existen reseñas de productos que alguien compró y opinó; 72,3 % del catálogo no tiene reseñas y no puede recomendarse con modelos basados en evidencia.
- **Sparsity extrema** (99,91 %) y mayoría de usuarios con una sola reseña.
- **El perfil es una moda de lo declarado** por cada usuario en sus reseñas; se asume disponible al consultar.
- **Muestra de test pequeña** (500 usuarios): diferencias entre modelos pueden no ser significativas.
- **El resultado no es una recomendación clínica**; no se evalúan ingredientes ni alergias.
- **Pendiente:** conectar `Sephora_Features.py` al flujo de modelado, o documentar que fue un experimento exploratorio previo (TF-IDF + K-Means).

---

## DEMO 2

Punto de partida: las limitaciones detectadas al cierre de la DEMO 1 (todos los modelos por debajo del baseline).

> **Solución final: Nexa v4.** El modelo **A2** (perfil + popularidad), desarrollado en la Etapa 2, se conserva como **línea alternativa previa**.

La DEMO 2 avanza en tres pasos:

| Paso | Qué se hizo | Dónde |
|---|---|---|
| **Etapa 2** (línea alternativa previa) | Cinco variantes del modelo A (A1–A5), con **A2** como la mejor, y un diagnóstico del baseline frente a A2 | `notebooks/Etapa_02/` |
| **Etapa 3** | Nuevo planteamiento (categoría + presupuesto + historial), nuevo protocolo de evaluación y los modelos Nexa v3 y Nexa v4 | `src/E3_*.py` |
| **Solución final y despliegue** | Nexa v4 con CLI y aplicación web; CLI y paquete portable de A2 | `nexa_deploy.py`, `nexa_streamlit.py`, `Sephora_Deploy.py` |

### Etapa 2 — Línea alternativa previa: mejora del modelo de perfil (A2)

**Protocolo.** Idéntico al de la DEMO 1 (`cold-start-users-v2.0`, semilla 42, 500 usuarios de validación y 500 de test, `protocol_id` `99fc64b0ea63`). `sephora_common.py` no cambió su lógica y los datos procesados son los mismos, por lo que los resultados son directamente comparables con los de la DEMO 1. Los hiperparámetros se eligen en validación y el test se evalúa una sola vez.

**Resultados en test (500 usuarios):**

| Modelo | Idea | Config. elegida | Hits | HR@10 | NDCG@10 | Cobertura de catálogo |
|---|---|---|---|---|---|---|
| Baseline popularidad | Productos con más reseñas positivas | – | 37 | **7,4 %** | 0,0332 | 0,4 % |
| A (DEMO 1) | K vecinos por perfil, voto simple | k=250 | 29 | 5,8 % | 0,0275 | 10,6 % |
| A1 | Votos ponderados por la similitud del vecino | k=250 | 30 | 6,0 % | 0,0281 | 10,9 % |
| **A2** | A1 + popularidad global | K=500, alpha=0,5 | 36 | 7,2 % | **0,0342** | 3,3 % |
| A3 | A1 + popularidad segmentada por perfil | k=250, alpha=1,0 | 30 | 6,0 % | 0,0281 | 10,9 % |
| A4 | A2 + similitud de contenido (TF-IDF) | K=500, alpha=0,25, beta=0,10 | 36 | 7,2 % | 0,0330 | 1,5 % |
| A5 | Top-10 con posiciones protegidas para los más populares | k=250, 2 protegidos | 29 | 5,8 % | 0,0268 | 9,5 % |

**Cómo se calculan A1 y A2.**

- **A1:** cada vecino aporta un voto ponderado por su similitud coseno con el perfil consultado.
- **A2:** combina ese puntaje con la popularidad del producto, ambos normalizados entre 0 y 1:

```
score_final = alpha * score_perfil + (1 - alpha) * score_popularidad
```

`K` es la cantidad de vecinos y `alpha` el peso del perfil frente a la popularidad (alpha = 1 equivale a usar solo el perfil).

**Lectura.**

- **A2 es la mejor variante:** sube de 5,8 % a 7,2 % de HR@10 y queda a **un solo acierto** del baseline (36 frente a 37). Su NDCG@10 es 0,001 superior al del baseline, una diferencia que con 500 usuarios **no es concluyente**: no se puede afirmar que supere al baseline.
- **A3** eligió alpha = 1,0 en validación, es decir, la popularidad segmentada no aportó información; se conserva como experimento descartado.
- **A4:** incorporar contenido (TF-IDF) no mejoró a A2 (mismo HR@10 y menor NDCG@10); se conserva como experimento descartado.
- **A5:** reservar posiciones para productos populares no mejoró: quedó por debajo de A2 y del baseline.

**Ejemplo de recomendación de A2** (perfil: piel `oily`, tono `dark`, ojos `blue`, cabello `brown`), Top-10:

| # | Producto | Marca | Categoría | Precio (USD) |
|---|---|---|---|---|
| 1 | Lip Sleeping Mask Intense Hydration with Vitamin C | LANEIGE | Lip Balms & Treatments | 24 |
| 2 | Niacinamide 10% + Zinc 1% Oil Control Serum | The Ordinary | Treatments | 6 |
| 3 | Cicapair Tiger Grass Color Correcting Treatment | Dr. Jart+ | Moisturizers | 52 |
| 4 | Soy Hydrating Gentle Face Cleanser | fresh | Cleansers | 39 |
| 5 | Mini Cicapair Tiger Grass Color Correcting Treatment | Dr. Jart+ | Mini Size | 21 |
| 6 | Protini Polypeptide Firming Refillable Moisturizer | Drunk Elephant | Moisturizers | 68 |
| 7 | Daily Microfoliant Exfoliator | Dermalogica | Cleansers | 65 |
| 8 | Mini Daily Microfoliant Exfoliator | Dermalogica | Mini Size | 18 |
| 9 | Green Clean Makeup Removing Cleansing Balm | Farmacy | Cleansers | 36 |
| 10 | Green Clean Makeup Meltaway Cleansing Balm Limited Edition | Farmacy | Cleansers | 60 |

#### Diagnóstico del baseline y de A2 (`E2_diagnostico_modelos.ipynb`)

Antes de seguir, el equipo analizó por qué el baseline es tan difícil de superar:

- **Las interacciones están muy concentradas:** los 10 productos más populares reúnen el 7,3 % de las interacciones positivas, y los 100 más populares, el 33,9 %. El 7,4 % de los productos objetivo del test está en el Top-10 global, y el 40,2 % en el Top-100.
- **A2 y el baseline aciertan en usuarios distintos:** 26 aciertos compartidos, 10 exclusivos de A2, 11 exclusivos del baseline, y 453 usuarios sin acierto en ninguno. Las señales son **complementarias**.
- **A2 no copia la popularidad:** en promedio, 6,37 de sus 10 recomendaciones pertenecen al Top-10 global; el resto cambia según el perfil. Recomienda 81 productos distintos (3,35 % del catálogo), frente a 10 del baseline.
- **El perfil aporta señal en algunos casos:** por ejemplo, un limpiador ocupa el puesto 77 del ranking global pero el 11 entre usuarios de piel `oily` y tono `mediumTan`. En otros segmentos ocurre lo contrario (por ejemplo, del 77 al 118 en piel `combination` y tono `fair`).

**Conclusión de la Etapa 2.** Con solo cuatro atributos de perfil, el techo del enfoque está cerca del baseline. A2 se conserva como línea alternativa previa y esto motiva el cambio de planteamiento de la Etapa 3, que da origen a la solución final.

---

### Etapa 3 — Nuevo planteamiento: categoría + presupuesto + historial

**Qué cambia.** En lugar de recomendar solo desde el perfil físico del usuario, el sistema recibe una **categoría** (`secondary_category`, por ejemplo `Moisturizers`) y un **presupuesto máximo en USD**, y opcionalmente un **usuario con historial**. Esto acerca la evaluación al uso real de un catálogo.

**Protocolo de evaluación** (`src/E3_preparacion_datos.py`). Es **distinto** al de la DEMO 1 y de la Etapa 2:

- **Split global cronológico** por fecha de reseña: ~80 % train, ~10 % validación y ~10 % test. Se conserva la última reseña por par usuario–producto dentro de cada período.
- **Positivo:** `rating >= 4`.
- **Perfiles y estadísticas** calculados únicamente con train.
- **Objetivos de evaluación:** pares nuevos de usuarios conocidos en train, solo positivos. Los productos sin historial se mantienen para contabilizar los fallos de cobertura.
- **Consulta simulada:** la categoría se toma del producto objetivo y el presupuesto se evalúa en cuatro escenarios (**USD 30, 60, 100 y sin límite**). No se recomiendan productos que el usuario ya reseñó.
- **Métricas:** Recall@10, NDCG@10 y Hit Rate@10, separadas por cohortes (todos los objetivos, producto conocido en train y producto nuevo), más cantidad de productos distintos recomendados.
- **Selección de parámetros** en validación por NDCG@10 promedio entre escenarios de presupuesto; el test se evalúa una vez, con un bootstrap pareado por usuario (500 remuestreos, intervalo del 95 %).

> **Importante:** el baseline de la Etapa 3 es un **ranking por valoración media ajustada por número de reseñas** dentro de la categoría y el presupuesto, `(n · media + m · media_global) / (n + m)`. **No es** el baseline de popularidad de las Etapas 1 y 2, y las métricas de la Etapa 3 **no son comparables** con las de la DEMO 1 ni con las de A2.

**Scripts de la Etapa 3:**

| Script | Qué hace |
|---|---|
| `E3_preparacion_datos.py` | Genera el split cronológico, el catálogo Skincare, los perfiles, las estadísticas de train y los objetivos de validación y test. Guarda `resumen_preparacion.json` |
| `E3_diagnostico_inicial.py` | Compara el ranking general con un ranking ajustado por tipo de piel (`m` ∈ {10, 50, 100}; `m_piel` ∈ {10, 50, 100, 500}) |
| `E3_M1_hibrido.py` (**Nexa v3**) | Mezcla el ranking general con la similitud de contenido de los productos que el usuario valoró bien, con un peso de contenido fijo (0 a 1) elegido en validación |
| `E3_M2_adaptativo.py` (**Nexa v4**) | Igual que M1, pero el peso del contenido **se adapta por producto**: `w = t / (n_reseñas + t)`, con `t` ∈ {0, 10, 50, 100}. Incluye una referencia de contenido puro y las diferencias pareadas con el baseline |

**Datos del split.** Train hasta el 2022-02-01 (870.693 pares usuario–producto), validación hasta el 2022-08-15 (109.487) y test posterior (108.911). El test tiene **14.632 usuarios** y 42.514 objetivos positivos; el **53,7 %** de ellos son productos sin historial en train.

**Resultados en test.** NDCG@10 por escenario de presupuesto (cohorte "todos los objetivos"):

| Presupuesto (USD) | Consultas | Baseline | Popularidad¹ | Contenido puro (M1) | **Nexa v4 (M2)** |
|---|---|---|---|---|---|
| 30 | 8.877 | 0,1514 | 0,1206 | 0,1140 | **0,2036** |
| 60 | 20.096 | 0,0740 | 0,0479 | 0,0659 | **0,1113** |
| 100 | 26.121 | 0,0674 | 0,0369 | 0,0557 | **0,1031** |
| Sin límite | 27.650 | 0,0624 | 0,0359 | 0,0532 | **0,0980** |

¹ Medida en la auditoría de Nexa (`reports/Etapa_03/auditoria_nexa/`).

Recall@10 del baseline y de Nexa v4 (M2): 0,263 → 0,358 (USD 30), 0,122 → 0,198 (USD 60), 0,125 → 0,187 (USD 100) y 0,117 → 0,177 (sin límite). Nexa v4 recomienda además más variedad de catálogo: entre 462 y 1.165 productos distintos según el presupuesto, frente a 149–171 del baseline.

**Diferencia pareada de NDCG@10 respecto del baseline** (bootstrap por usuario, intervalo del 95 %):

| Modelo | Δ NDCG@10 | Intervalo del 95 % | Lectura |
|---|---|---|---|
| Diagnóstico (ranking por tipo de piel) | −0,0056 | −0,0063 a −0,0048 | Peor que el baseline |
| M1 — Nexa v3 (contenido, peso 1,0 elegido en validación) | −0,0128 | −0,0165 a −0,0092 | Peor que el baseline |
| **M2 — Nexa v4 (peso adaptativo)** | **+0,0378** | **+0,0355 a +0,0402** | Mejor que el baseline en los 4 presupuestos |
| Auditoría de Nexa v4 (parámetros congelados m=100, t=100) | +0,0377 | +0,0353 a +0,0402 | Confirma a M2 sin reajustar |
| Auditoría de Nexa v4 frente a **popularidad** | +0,0659 | +0,0625 a +0,0690 | Mejor que la popularidad |

**Lectura.** Con el protocolo de la Etapa 3, el peso adaptativo es lo que hace funcionar al contenido: el contenido puro (M1) empeora al baseline, pero mezclarlo según el respaldo de cada producto (`w = t / (n + t)`) lo mejora en todos los escenarios. Cabe aclarar:

- La Etapa 3 usa un protocolo distinto al de las Etapas 1 y 2: sus valores absolutos **no se pueden comparar** con los de la DEMO 1 ni con los de A2.
- El baseline de la Etapa 3 es la valoración media ajustada por categoría y presupuesto; la popularidad se mide por separado en la auditoría.
- El intervalo agrupa por usuario, pero incluye consultas repetidas entre presupuestos, y el EDA previo vio todo el dataset; por eso el test no es completamente independiente del diseño.
- La auditoría reproduce a M2 con una diferencia mínima (+0,03772 frente a +0,03781), pero no es idéntica: ver [Pendientes](#pendientes-por-confirmar).

Los resultados completos están en `reports/Etapa_03/` (`metricas_test.csv`, `validacion_parametros.csv`, `resumen_evaluacion.json` y `cobertura_evaluacion.csv` de cada modelo). Para regenerarlos:

```bash
python src/E3_preparacion_datos.py --project-root .
python src/E3_diagnostico_inicial.py --project-root .
python src/E3_M1_hibrido.py --project-root .
python src/E3_M2_adaptativo.py --project-root .
```

**Limitaciones declaradas por los propios scripts:**

- La categoría del producto objetivo simula la consulta; no es una intención observada.
- Precio, categoría y catálogo corresponden a una captura posterior, no al catálogo histórico.
- Los usuarios nuevos quedan fuera de la evaluación y requieren una evaluación de cold start aparte.
- Los productos sin reseñas se puntúan con la media global; no se afirma personalización para ellos.
- La relevancia es solo la observada en reseñas, no ventas; un producto no reseñado no es un negativo confirmado.
- El EDA previo vio todo el dataset, por lo que el test no es completamente independiente del diseño del enfoque.
- La similitud de ingredientes no demuestra compatibilidad clínica.

---

### Solución final — Nexa v4: recomendador por categoría y presupuesto

Es la **solución final del proyecto**: el modelo resultante de `E3_M2_adaptativo.py`, entregado ya construido en `models/nexa_v4/` y empaquetado en la clase `NexaRecommender` (`src/nexa_recommender.py`).

**Cómo funciona.**

1. **Ranking general.** Para la categoría y el presupuesto, ordena los productos por su valoración media suavizada con el número de reseñas (`m = 100`), de modo que un producto con pocas reseñas no sube solo por tener una nota alta. El puntaje es un percentil entre los candidatos, no una probabilidad.
2. **Personalización con historial.** Si el usuario existe y tiene productos valorados positivamente, se construye su perfil como el promedio de los vectores TF-IDF de esos productos (highlights, ingredientes y categoría terciaria; 8.000 términos, unigramas y bigramas) y se calcula su similitud coseno con cada candidato.
3. **Mezcla adaptativa.** `score = (1 − w) · ranking_general + w · similitud`, con `w = 100 / (n_reseñas + 100)`: los productos con pocas reseñas dependen más del contenido y los muy valorados, de su ranking general.
4. **Usuarios nuevos o desconocidos** reciben el ranking general (`fallback_reason: unknown_or_new_user`). Los productos ya valorados por el usuario se excluyen.
5. **Explicabilidad:** cada producto trae una explicación en texto y su `content_weight`.

**Modos** (`--mode`): `adaptativo` (por defecto), `baseline` (ranking general) y `popularidad` (por número de usuarios con reseñas).

**Artefactos** (`models/nexa_v4/`): `catalog.csv` (2.420 productos Skincare), `features.npz`, `positive.npz`, `seen.npz`, `users.json` (439.645 usuarios), `config.json` y `checksums.json` (SHA-256 verificado al cargar). La configuración declara `m_general = 100`, `transition_users = 100` y `"evaluation_status": "exploratory; test previously inspected"`: el test ya se había revisado, por lo que sus métricas no son una validación limpia.

**Estado de la evaluación.** Nexa v4 comparte la lógica y los parámetros principales con el experimento M2, y la auditoría con parámetros congelados (`reports/Etapa_03/auditoria_nexa/`) confirma su resultado en la [Etapa 3](#etapa-3--nuevo-planteamiento-categoría--presupuesto--historial): ΔNDCG@10 ≈ +0,0377 frente al baseline y ≈ +0,0659 frente a la popularidad. La auditoría no es idéntica a M2 y **queda pendiente una prueba automatizada que verifique que los datos y los rankings del modelo guardado son exactamente equivalentes a los de M2**. El `config.json` del modelo conserva la marca `"exploratory; test previously inspected"`.

**Ejecución** (desde la raíz, con `src/`, `app/`, `models/` y `requirements.txt` en su lugar):

```bash
pip install -r requirements.txt
python app/nexa_deploy.py --self-test
python app/nexa_deploy.py --category Moisturizers --budget 50
python app/nexa_deploy.py --category Moisturizers --budget 50 --user-id 10000015049
```

La interfaz web se describe en [API, interfaz y monitoreo](#api-interfaz-y-monitoreo-de-nexa-v4). No se necesitan los datos crudos ni los procesados para usar el modelo guardado. Los resultados se guardan en `artifacts/nexa_deploy/` (JSON, `deploy.log` y `metricas_ejecucion.jsonl`). Reconstruir el modelo (`python src/nexa_recommender.py`) sí requiere `data/processed/recomendador_v2/`, que no forma parte de los archivos entregados.

**Ejemplo de recomendación** (`Moisturizers`, USD 50, sin usuario; 233 de 551 productos cumplen el presupuesto):

| # | Producto | Marca | Precio (USD) | Reseñas en train | Rating medio |
|---|---|---|---|---|---|
| 1 | Rose & Hyaluronic Acid Deep Hydration Moisturizer | fresh | 46 | 1.628 | 4,69 |
| 2 | Plum Plump Hyaluronic Acid Moisturizer | Glow Recipe | 39 | 714 | 4,70 |
| 3 | Nourishing Moisturizer with Prebiotics | SEPHORA COLLECTION | 18 | 236 | 4,80 |
| 4 | Barrier+ Triple Lipid-Peptide Lotion Moisturizer | Skinfix | 42 | 535 | 4,71 |
| 5 | Mini Indigo Overnight Repair Serum in Cream Treatment | Tatcha | 22 | 942 | 4,64 |

---

### API, interfaz y monitoreo de Nexa v4

**API (FastAPI)** — `api/main.py`, rama `main`. Carga el modelo una sola vez al iniciar y expone:

| Endpoint | Qué hace |
|---|---|
| `GET /` | Mensaje de bienvenida y enlace a la documentación (`/docs`) |
| `GET /health` | Estado de la API y si el modelo está cargado |
| `GET /categories` | Categorías válidas |
| `POST /recommend` | Recibe `category`, `budget_usd`, `user_id` (opcional) y `top_n` (1–100) y devuelve el ranking |

Responde 503 si el modelo no está disponible y 400 si la solicitud es inválida.

**Interfaz web (Streamlit)** — `app/nexa_streamlit.py`, rama `Fernanda`. Con el logo y el ícono de NEXA; consume la API (`NEXA_API_URL`, por defecto `http://127.0.0.1:8000`). Se inicia en dos terminales, desde la raíz:

```bash
python -m uvicorn api.main:app --reload
python -m streamlit run app/nexa_streamlit.py
```

La rama `main` incluye otra versión, `nexa_streamlit.py`, que carga el modelo directamente y registra cada inferencia.

**Registro de inferencias.** La API y esa versión de la interfaz agregan una fila a `reports/inference_history.csv` por cada recomendación exitosa, con `timestamp_utc`, `category`, `budget_usd` y `mode_used`. **No se guarda el `user_id`.** Abrir o recargar la interfaz no cuenta como inferencia; las solicitudes inválidas o fallidas tampoco se registran.

**Monitoreo** — `Sephora_Monitoring.py` y `README_Monitoring.md`, rama `main`. Lee el historial y revisa tres cosas:

1. **Salud del sistema:** integridad del modelo, productos del catálogo (2.420), usuarios y dimensiones de la matriz de características.
2. **Data drift:** PSI (Population Stability Index) de los presupuestos registrados frente a los precios del catálogo. Estable por debajo de 0,1, advertencia entre 0,1 y 0,25 y alerta roja por encima de 0,25. No compara dos períodos de solicitudes.
3. **Modos de inferencia:** distribución porcentual de `mode_used` (`personalized`, `general`, etc.).

Genera dos gráficos en `reports/monitoring_charts/`: modos de inferencia (anillo) y PSI (barra con umbrales). Si todavía no hay solicitudes, avisa y termina sin generar gráficos.

> **Nota sobre las rutas.** La API y el monitoreo de `main` asumen una carpeta `Nexa.v4/` con `src/` y `models/` dentro de la raíz del proyecto (así está en el equipo de quien los desarrolló). En la rama `Fernanda`, `src/` y `models/` están en la raíz; hay que alinear ambas estructuras al unificar las ramas.

### Línea alternativa previa — Despliegue y paquete de A2

**`Sephora_Deploy.py`.** CLI (sin API) que genera recomendaciones con A2, la línea alternativa previa, (K=500, alpha=0,5):

```bash
python Sephora_Deploy.py                                   # perfil de ejemplo
python Sephora_Deploy.py --profile-file perfiles.json      # uno o varios perfiles
python Sephora_Deploy.py --profile-json '{"skin_type":"oily","skin_tone":"dark"}' --top-n 5
python Sephora_Deploy.py --self-test                       # pruebas de perfiles y validación de entradas
```

Valida el perfil (rechaza campos desconocidos o de tipo incorrecto; los valores no vistos en entrenamiento se tratan como `Unknown`, o dan error con `--strict-profile`) y guarda `artifacts/sephora_deploy/recomendaciones.json`, `deploy.log` y `metricas_ejecucion.jsonl`. Parámetros opcionales: `--top-n` (1–100), `--k`, `--alpha`, `--output`.

**Paquete `deploy_provisional_A2`.** Versión mínima para ejecutar A2 por separado: el notebook de A2, `sephora_common.py` y `data/processed/` con `products_clean.parquet` (2,2 MB) y `reviews_clean.parquet` (218 MB). No requiere `Sephora_EDA.py` ni `Sephora_Features.py`.

**Cambios en `sephora_common.py`.** La lógica del protocolo no cambió; solo `find_project_root()`, que ahora busca `data/processed` desde la ubicación del archivo y también dentro de `deploy_provisional_A2/deploy_provisional`. A2 también se entrega como script (`model_A2-perfil_popularidad.py`).

**Orden de ejecución local acordado:** `Sephora_EDA.py → Sephora_Features.py → sephora_common.py → model_A2 → Sephora_Deploy.py`, con los scripts en la raíz del proyecto, junto a `perfil.json`, `perfiles.json` y el tablero `Sephora_PBI.pbix` (ver [Antecedente](#antecedente--exploración-del-dataset-en-power-bi)).

## Changelog DEMO 1 → DEMO 2

| Componente | DEMO 1 | DEMO 2 | Cambio |
|---|---|---|---|
| Mejor modelo de perfil | A: vecinos por perfil (k=250), HR@10 5,8 % | **A2** (línea alternativa previa): vecinos ponderados + popularidad (K=500, alpha=0,5), HR@10 7,2 % | +1,4 puntos; a un acierto del baseline |
| Puntaje de vecinos | Voto simple | Ponderado por similitud (A1) | Nuevo |
| Señal de popularidad | Solo en el baseline | Combinada con el perfil mediante `alpha` (A2) | Nuevo |
| Variantes exploradas | – | A3 (popularidad segmentada), A4 (contenido TF-IDF), A5 (posiciones protegidas) | Evaluadas; ninguna supera a A2 |
| Diagnóstico | – | Concentración de popularidad y solapamiento de aciertos entre A2 y baseline | Nuevo |
| Planteamiento | Cold start: solo perfil | Categoría + presupuesto + historial opcional (Etapa 3) | Cambio de problema |
| Protocolo de evaluación | Split por usuario, 500/500, HR@10 y NDCG@10 | Split cronológico 80/10/10, escenarios de presupuesto, Recall@10 y NDCG@10 por cohorte, bootstrap por usuario | Nuevo |
| Baseline | Popularidad | Valoración media ajustada por categoría y presupuesto (Etapa 3) | Cambia; métricas no comparables |
| Modelos de la Etapa 3 | – | Nexa v3 (contenido, peso fijo; ΔNDCG@10 = −0,0128 frente a su baseline) y **Nexa v4, solución final** (peso adaptativo por producto; ΔNDCG@10 = +0,0378 frente a su baseline y +0,0659 frente a popularidad) | Nuevo |
| API | – | FastAPI (`/health`, `/categories`, `/recommend`) con registro de inferencias | Nuevo |
| Monitoreo | – | `Sephora_Monitoring.py`: salud del modelo, PSI de presupuestos y distribución de modos, con gráficos | Nuevo |
| Validación automática | – | GitHub Actions: verifica la sintaxis de `src/` y `app/` en cada push | Nuevo |
| Señal de contenido | `Sephora_Features.py` (TF-IDF de 500 términos) sin conectar al modelado | TF-IDF de 8.000 términos (unigramas y bigramas), integrado en Nexa | Conectado al modelo |
| `sephora_common.py` | Busca la raíz desde el directorio actual | Busca desde su ubicación y en `deploy_provisional_A2/` | Solo rutas |
| Uso del modelo | Solo notebooks | CLI de A2; CLI, API y app Streamlit de Nexa v4 | Nuevo |
| Integridad del modelo | – | `checksums.json` con SHA-256 verificado al cargar | Nuevo |
| Estructura | Carpeta local con notebooks 01–05 | Repositorio con `notebooks/Etapa_01`, `Etapa_02`, `src/`, `models/` | Reorganización |

---

## Pendientes por confirmar

- **Unificar las ramas:** `main` (API, monitoreo, CLI de A2) y `Fernanda` (proyecto completo, modelo, app y resultados) no están integradas. La interfaz de `Fernanda` requiere la API de `main`.
- **Rama `martinpa-spec-upload-1`:** reemplaza el `README.md` de la raíz por el texto de la carpeta `app/` y elimina `nexa_streamlit.py`; conviene no integrarla tal cual.
- **Rutas de la API y del monitoreo:** asumen `Nexa.v4/src` y `Nexa.v4/models`; en el repositorio están en `src/` y `models/`.
- **Dependencias:** `requirements.txt` no incluye `requests` (que usa la interfaz de `Fernanda`), ni `pydantic`, ni `seaborn` (que usa el monitoreo).
- **Archivos compilados en `main`:** `api/__pycache__/*.pyc` está versionado; conviene agregarlo al `.gitignore` de `main`.
- **Equivalencia Nexa v4 ↔ M2:** prueba automatizada de datos y rankings pendiente; el script de la auditoría (`reports/Etapa_03/auditoria_nexa/`) no está en el repositorio.
- **Terminología del baseline:** el código ordena por valoración media ajustada por número de reseñas; el resumen del equipo lo llama "popularidad histórica". Unificar el término en la presentación.
- **`Sephora_EDA_complementario.py`:** sus conclusiones (`reports/EDA/complementario/`) no están en el repositorio.
- **`reports/registro.md` y `tests/`:** siguen siendo plantillas sin contenido.
- **`Sephora_PBI.pbix`** y las dos imágenes de `reports/figures/`: falta subirlos.
- **Usuarios únicos:** el tablero de Power BI muestra 503.196 y el EDA de la DEMO 1 reporta 503.216 (20 de diferencia); confirmar el origen.
- **`perfil.json` y `perfiles.json`:** contenido y uso exacto.
- **Roles de cada integrante** (el historial de commits muestra aportes de tres cuentas de GitHub) y **enlace de descarga del dataset.**

---

## Equipo

**NEXA Data Solutions**

- Alexander Ruiz
- Martin Pavon
- Fernanda Villafañe
- Andrés Pinto

**TODO:** roles y enlaces (GitHub / LinkedIn) de cada integrante.
