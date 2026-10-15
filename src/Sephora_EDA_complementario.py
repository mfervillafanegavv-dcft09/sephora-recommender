"""Nexa: EDA complementario; no modifica el EDA ni sus datos originales.

Guardar en src/ y ejecutar desde la raíz:
    python src/Sephora_EDA_complementario.py --project-root .
Necesita pandas y numpy, ya utilizados en el EDA original.
Entradas: data/processed/products_clean.csv y reviews_clean.csv (o parquet).
Salidas nuevas: reports/EDA/complementario/ (CSV y conclusiones.md).
No entrena modelos. Las asociaciones descriptivas no prueban causalidad,
compatibilidad clínica ni mejora frente al baseline.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

PROFILE = ['skin_type', 'skin_tone', 'eye_color', 'hair_color']

def load(folder, stem):
    csv = folder / (stem + '.csv')
    parquet = folder / (stem + '.parquet')
    # CSV conserva IDs como texto, incluso con ceros iniciales.
    if csv.exists():
        return pd.read_csv(csv, dtype={'author_id': 'string', 'product_id': 'string'}, low_memory=False)
    if parquet.exists():
        return pd.read_parquet(parquet)
    raise FileNotFoundError(f'Falta {csv} o {parquet}. Buscar en data/processed del proyecto de tu compañero; reports no contiene las reseñas individuales necesarias.')

def run(root, min_users=30):
    folder = root / 'data' / 'processed'
    p, r = load(folder, 'products_clean'), load(folder, 'reviews_clean')
    required = {'product_id', 'author_id', 'rating'}
    if not required.issubset(r.columns) or 'product_id' not in p:
        raise ValueError(f'Las reseñas necesitan {required}; el catálogo necesita product_id.')
    for df in (p, r):
        df['product_id'] = df['product_id'].astype('string').str.strip().replace('', pd.NA)
    r['author_id'] = r['author_id'].astype('string').str.strip().replace('', pd.NA)
    for col in PROFILE:
        if col not in r:
            r[col] = pd.NA
        r[col] = r[col].astype('string').str.strip().str.lower().replace('', pd.NA)
    r['eye_color'] = r['eye_color'].replace('grey', 'gray')
    r['rating'] = pd.to_numeric(r['rating'], errors='coerce')
    valid = r['rating'].isin([1, 2, 3, 4, 5]) & r['author_id'].notna() & r['product_id'].notna()
    invalid_rows = int((~valid).sum())
    r = r.loc[valid].copy()
    if r.empty:
        raise ValueError('No hay reseñas con IDs y puntuaciones válidos.')
    if p['product_id'].isna().any() or p['product_id'].duplicated().any():
        raise ValueError('El catálogo tiene IDs nulos o repetidos; revisar antes de unir.')
    metadata = [c for c in ['product_id', 'product_name', 'primary_category', 'secondary_category', 'price_usd'] if c in p]
    r = r.drop(columns=[c for c in metadata if c != 'product_id' and c in r], errors='ignore').merge(p[metadata], on='product_id', how='left', validate='many_to_one')
    if 'primary_category' not in r:
        r['primary_category'] = pd.NA
    out = root / 'reports' / 'EDA' / 'complementario'
    out.mkdir(parents=True, exist_ok=True)
    def save(df, name):
        df.to_csv(out / name, index=False, encoding='utf-8-sig')
    notes = ['# Nexa — resultados del EDA complementario',
             f'Reseñas válidas: {len(r):,}. Excluidas por IDs/rating inválidos: {invalid_rows:,}.',
             'Resultados exploratorios sobre reseñas observadas; no representan compras ni todo el mercado.']
    # Una puntuación promedio por usuario-producto evita contar varias veces el par.
    pairs = r.groupby(['author_id', 'product_id'], as_index=False)['rating'].mean()
    # Perfil modal por usuario; auditoría de contradicciones antes de usarlo.
    users = pd.DataFrame({'author_id': sorted(r['author_id'].unique())})
    for col in PROFILE:
        f = r.dropna(subset=[col]).groupby(['author_id', col]).size().rename('n').reset_index()
        f = f.sort_values(['author_id', 'n', col], ascending=[True, False, True])
        chosen = f.drop_duplicates('author_id')[['author_id', col]]
        counts = f.groupby('author_id').size().rename(col + '_distinct_values')
        users = users.merge(chosen, on='author_id', how='left').merge(counts, on='author_id', how='left')
        users[col + '_distinct_values'] = users[col + '_distinct_values'].fillna(0).astype(int)
    save(users, 'auditoria_perfiles_usuario.csv')
    d = pairs.merge(users[['author_id'] + PROFILE], on='author_id', validate='many_to_one').merge(p[metadata], on='product_id', how='left', validate='many_to_one')
    if 'primary_category' not in d:
        d['primary_category'] = pd.NA
    # P1: comparar dentro del mismo producto, seca vs otros tipos conocidos.
    skin = d.loc[d['skin_type'].notna()].copy()
    skin['grupo'] = np.where(skin['skin_type'].eq('dry'), 'dry', 'other_known')
    stat = skin.groupby(['product_id', 'grupo']).agg(n=('rating', 'size'), media=('rating', 'mean'), sd=('rating', 'std')).unstack('grupo')
    if {'dry', 'other_known'}.issubset(set(stat.columns.get_level_values(1))):
        result = pd.DataFrame(index=stat.index)
        for group in ['dry', 'other_known']:
            result['usuarios_' + group] = stat[('n', group)]
            result['media_' + group] = stat[('media', group)]
        result['diferencia_seca_menos_otros'] = result['media_dry'] - result['media_other_known']
        result['soporte_minimo'] = (result['usuarios_dry'] >= min_users) & (result['usuarios_other_known'] >= min_users)
        save(result.reset_index().merge(p[metadata], on='product_id'), 'p1_piel_seca_por_producto.csv')
        notes.append(f'## 1. Piel seca\nComparación dentro de cada producto: {int(result.soporte_minimo.sum())} productos con al menos {min_users} usuarios por grupo. Ver diferencias; requieren validación en datos separados antes de personalizar. No son pruebas de significancia.')
    else:
        notes.append('## 1. Piel seca\nNo hay ambos grupos conocidos suficientes para compararlos.')
    # P2: nunca extrapolar skincare a maquillaje.
    makeup = d.loc[d['primary_category'].astype('string').str.lower().eq('makeup').fillna(False)]
    tone = makeup.groupby(['product_id', 'skin_tone']).agg(usuarios=('rating', 'size'), media=('rating', 'mean')).reset_index()
    save(tone, 'p2_maquillaje_por_tono.csv')
    notes.append(f'## 2. Tono y maquillaje\n{len(makeup):,} pares con reseñas de maquillaje. ' + ('No evaluable con esta base.' if makeup.empty else 'Comparar tonos dentro del mismo producto; no permite recomendar tonalidades específicas sin IDs de variantes.'))
    # P3: descripción; texto y helpfulness no se usan como inputs predictivos.
    r['rating_group'] = np.where(r['rating'] >= 4, '4-5', '1-3')
    for col in PROFILE + ['primary_category', 'secondary_category']:
        if col in r:
            temp = r.assign(atributo=r[col].fillna('(sin dato)')).groupby('atributo').agg(resenas=('rating', 'size'), usuarios=('author_id', 'nunique'), media=('rating', 'mean'), positivas=('rating', lambda s: int((s >= 4).sum()))).reset_index()
            temp['pct_positivas'] = 100 * temp['positivas'] / temp['resenas']
            save(temp, 'p3_positividad_' + col + '.csv')
    r['text_length'] = r.get('review_text', pd.Series(pd.NA, index=r.index, dtype='string')).astype('string').str.len()
    save(r.groupby('rating_group').agg(resenas=('rating', 'size'), longitud_media_texto=('text_length', 'mean')).reset_index(), 'p3_texto_por_calificacion.csv')
    notes.append('## 3. Reseñas positivas\nTablas por atributo y categoría, con cantidad de reseñas y usuarios. Asociaciones descriptivas; el producto y la repetición de usuarios pueden explicar diferencias. Longitud del texto es una descripción posterior a la reseña, no un input disponible antes de recomendar.')
    # P4: contar usuarios (no filas de reseñas) y perfiles coherentes.
    for cols, name in [(['skin_type'], 'tipo_piel'), (PROFILE, 'perfil_completo')]:
        complete = users.dropna(subset=cols)
        coherent = complete.loc[(complete[[c + '_distinct_values' for c in cols]] <= 1).all(axis=1)]
        support = complete.groupby(cols).size().rename('usuarios').reset_index()
        consistent = coherent.groupby(cols).size().rename('usuarios_sin_conflictos').reset_index()
        save(support.merge(consistent, on=cols, how='left').fillna({'usuarios_sin_conflictos': 0}), 'p4_soporte_' + name + '.csv')
    notes.append('## 4. Perfiles similares\nVer soporte por tipo de piel y combinación de los cuatro atributos. El perfil modal resume valores observados; auditar contradicciones. Tener atributos completos no significa tener muchos vecinos. En evaluación, reconstruir perfiles usando solo entrenamiento.')
    # P5: incluir productos sin reseñas; separar catálogo de skincare.
    counts = r.groupby('product_id').size().rename('resenas')
    coverage = p[metadata].merge(counts, on='product_id', how='left')
    coverage['resenas'] = coverage['resenas'].fillna(0).astype(int)
    coverage['grupo_resenas'] = pd.cut(coverage['resenas'], [-1, 0, 4, 19, 49, float('inf')], labels=['0', '1-4', '5-19', '20-49', '50+'])
    save(coverage.sort_values('resenas'), 'p5_soporte_por_producto.csv')
    grouping = ['grupo_resenas'] + (['primary_category'] if 'primary_category' in coverage else [])
    save(coverage.groupby(grouping, observed=True, dropna=False).size().rename('productos').reset_index(), 'p5_resumen_soporte.csv')
    notes.append(f'## 5. Pocas reseñas\n{int((coverage.resenas == 0).sum()):,} productos sin reseñas. Revisar cobertura por categoría: ausencia por alcance del dataset no equivale a falta de ventas o satisfacción.')
    # P6: escala y sesgo positivo.
    dist = r['rating'].value_counts().sort_index().rename_axis('rating').reset_index(name='resenas')
    dist['porcentaje'] = 100 * dist['resenas'] / len(r)
    save(dist, 'p6_distribucion_rating.csv')
    notes.append(f'## 6. Calificaciones positivas\n{100 * r.rating.ge(4).mean():.2f}% de reseñas con 4-5 estrellas. Esto no demuestra por sí solo por qué ganó el baseline.')
    # P7: productos distintos, relevantes y tamaño de población elegible.
    history = pairs.groupby('author_id').agg(productos_distintos=('product_id', 'size'), positivos=('rating', lambda s: int((s >= 4).sum()))).reset_index()
    save(history, 'p7_historial_por_usuario.csv')
    eligibility = []
    for k in [2, 3, 5, 10]:
        mask = history.productos_distintos.ge(k)
        eligibility.append({'min_productos_distintos': k, 'usuarios': int(mask.sum()), 'pct_usuarios': 100 * mask.mean(), 'usuarios_con_positivo': int((mask & history.positivos.ge(1)).sum())})
    save(pd.DataFrame(eligibility), 'p7_elegibilidad_evaluacion.csv')
    notes.append('## 7. Historial suficiente\nDos productos distintos permiten reservar uno y conservar uno en entrenamiento; no garantizan evaluación robusta. Los umbrales 3/5/10 describen escenarios, no reglas universales. Revisar existencia de positivos en test, fechas, cobertura y candidatos comunes. No interpretar productos no reseñados como negativos confirmados.')
    # Sparsity counts distinct pairs, rather than repeated reviews.
    possible = pairs.author_id.nunique() * pairs.product_id.nunique()
    save(pd.DataFrame([{'pares_unicos': len(pairs), 'pares_posibles': possible, 'sparsity_pct': 100 * (1-len(pairs)/possible)}]), 'dispersion_pares_unicos.csv')
    notes.append('## Siguiente decisión\nSi solo hay reseñas de skincare, enfocar allí el experimento. Comparar un baseline general y uno por subcategoría contra el mismo ranking con ajuste por tipo de piel. Misma población, separación de datos, candidatos y métricas; construir estadísticas solo con entrenamiento. Si el ajuste no mejora, conservar el ranking de referencia y documentar el resultado. Utilidad comercial potencial no equivale a ventas o conversión demostradas.')
    (out / 'conclusiones.md').write_text('\n\n'.join(notes), encoding='utf-8')
    print(f'Resultados guardados en: {out.resolve()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--min-users', type=int, default=30, help='Umbral exploratorio por grupo para P1 (no prueba estadística).')
    args = parser.parse_args()
    if args.min_users < 1:
        parser.error('--min-users debe ser positivo')
    run(args.project_root.resolve(), args.min_users)
