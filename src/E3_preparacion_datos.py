"""Nexa: prepara datos y estadísticas para recomendación de skincare.
Guardar en src/. Desde raíz: python src/E3_preparacion_datos.py --project-root .
Dependencias: pandas, numpy; pyarrow si se usan entradas Parquet.
No modifica entradas. Escribe data/processed/Etapa_03 y reports/Etapa_03.
Precios y categorías corresponden a la captura del catálogo, no a su estado histórico.
No usa rating/reviews del catálogo, votos de utilidad ni favoritos para ordenar.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def load(folder, stem):
    candidates = [folder / (stem + '.parquet'), folder / (stem + '(1).parquet'), folder / (stem + '.csv'), folder / (stem + '(1).csv')]
    for path in candidates:
        if path.exists():
            if path.suffix == '.parquet':
                try:
                    return pd.read_parquet(path), str(path)
                except ImportError:
                    continue
            return pd.read_csv(path, dtype={'author_id':'string', 'product_id':'string'}, low_memory=False), str(path)
    raise FileNotFoundError(f'No se pudo leer {stem} en {folder}. Si solo hay Parquet: python -m pip install pyarrow')

def stats(df, groups):
    return df.groupby(groups, as_index=False).agg(n_users=('author_id','nunique'), mean_rating=('rating','mean'), positive_count=('positive','sum'), positive_rate=('positive','mean'))

def prepare(root, train_end=None, validation_end=None):
    source = root / 'data' / 'processed'
    products, product_path = load(source, 'products_clean')
    reviews, review_path = load(source, 'reviews_clean')
    required = {'author_id','product_id','rating','submission_time','skin_type'}
    if not required.issubset(reviews):
        raise ValueError(f'Faltan columnas de reseñas: {required-set(reviews)}')
    for df in [products,reviews]:
        df['product_id'] = df['product_id'].astype('string').str.strip().replace('',pd.NA)
    if products.product_id.isna().any() or products.product_id.duplicated().any():
        raise ValueError('El catálogo necesita IDs únicos y no nulos.')
    if not {'primary_category','secondary_category','price_usd'}.issubset(products):
        raise ValueError('Faltan categoría o precio en productos.')
    products['price_usd'] = pd.to_numeric(products.price_usd,errors='coerce')
    # Auditoría antes de seleccionar skincare.
    audit = {'input_reviews':len(reviews), 'input_products':len(products)}
    if 'price_usd' in reviews:
        price = reviews[['product_id','price_usd']].copy()
        price['price_usd'] = pd.to_numeric(price.price_usd,errors='coerce')
        price = price.merge(products[['product_id','price_usd']],on='product_id',how='left',suffixes=('_review','_catalog'),validate='many_to_one')
        comparable = price.price_usd_review.notna() & price.price_usd_catalog.notna()
        audit['price_comparable_review_rows'] = int(comparable.sum())
        audit['price_disagreement_review_rows'] = int(((price.price_usd_review-price.price_usd_catalog).abs().gt(.01)&comparable).sum())
    catalog = products.loc[products.primary_category.astype('string').str.lower().eq('skincare').fillna(False)].copy()
    audit['skincare_catalog_products'] = len(catalog)
    audit['skincare_products_invalid_price'] = int((catalog.price_usd.isna() | ~np.isfinite(catalog.price_usd) | catalog.price_usd.le(0)).sum())
    # Preserve products with invalid price for audit, exclude from budget candidates later.
    reviews['author_id'] = reviews.author_id.astype('string').str.strip().replace('',pd.NA)
    reviews['rating'] = pd.to_numeric(reviews.rating,errors='coerce')
    reviews['submission_time'] = pd.to_datetime(reviews.submission_time,errors='coerce',utc=True)
    reviews['skin_type'] = reviews.skin_type.astype('string').str.lower().str.strip().replace('',pd.NA)
    valid = reviews.author_id.notna() & reviews.product_id.notna() & reviews.rating.isin([1,2,3,4,5]) & reviews.submission_time.notna()
    audit['invalid_review_rows'] = int((~valid).sum())
    columns = ['author_id','product_id','rating','submission_time','skin_type']
    r = reviews.loc[valid & reviews.product_id.isin(catalog.product_id),columns].copy()
    if r.empty:
        raise ValueError('No hay reseñas válidas de skincare.')
    # Global chronological split. Identical timestamps always stay together.
    t1 = pd.to_datetime(train_end,utc=True) if train_end else r.submission_time.quantile(.8)
    t2 = pd.to_datetime(validation_end,utc=True) if validation_end else r.submission_time.quantile(.9)
    if not r.submission_time.min() <= t1 < t2 < r.submission_time.max():
        raise ValueError('Los cortes deben definir tres períodos no vacíos y ordenados.')
    raw_parts = {'train':r[r.submission_time.le(t1)], 'validation':r[r.submission_time.gt(t1)&r.submission_time.le(t2)], 'test':r[r.submission_time.gt(t2)]}
    parts = {}
    for name,df in raw_parts.items():
        # Last available review per pair inside each period, no future information.
        parts[name] = df.sort_values(['submission_time','author_id','product_id'],kind='stable').drop_duplicates(['author_id','product_id'],keep='last').copy()
        parts[name]['positive'] = parts[name].rating.ge(4).astype(int)
        audit[name+'_raw_rows'] = len(df)
        audit[name+'_unique_pairs'] = len(parts[name])
    train = parts['train']
    # Use the most recent known skin type, strictly from training.
    profiles = train.dropna(subset=['skin_type']).sort_values('submission_time',kind='stable').drop_duplicates('author_id',keep='last')[['author_id','skin_type']]
    train = train.drop(columns='skin_type').merge(profiles,on='author_id',how='left',validate='many_to_one')
    parts['train'] = train
    out = source / 'Etapa_03'; out.mkdir(parents=True,exist_ok=True)
    report = root / 'reports' / 'Etapa_03'; report.mkdir(parents=True,exist_ok=True)
    def save(df,name,folder=out):
        df.to_csv(folder / name,index=False,encoding='utf-8-sig')
    keep = [c for c in ['product_id','product_name','brand_name','primary_category','secondary_category','tertiary_category','price_usd','size','out_of_stock','loves_count'] if c in catalog]
    save(catalog[keep],'catalogo_skincare.csv')
    save(profiles,'perfiles_train.csv')
    save(stats(train,['product_id']),'estadisticas_generales_train.csv')
    save(stats(train.dropna(subset=['skin_type']),['product_id','skin_type']),'estadisticas_piel_train.csv')
    save(catalog.groupby('secondary_category',dropna=False).price_usd.agg(['count','min','median','max']).reset_index(),'precios_por_categoria.csv',report)
    history = set(zip(train.author_id,train.product_id))
    known_users = set(train.author_id); known_products = set(train.product_id)
    for name,df in parts.items():
        save(df, name+'.csv')
        if name == 'train':
            continue
        unseen = np.array([(u,p) not in history for u,p in zip(df.author_id,df.product_id)])
        targets = df.loc[unseen & df.author_id.isin(known_users)].drop(columns='skin_type').merge(profiles,on='author_id',how='left',validate='many_to_one')
        targets = targets.merge(catalog[['product_id','secondary_category','price_usd']],on='product_id',how='left',validate='many_to_one')
        targets['product_seen_in_train'] = targets.product_id.isin(known_products)
        save(targets,'objetivos_'+name+'.csv')
        audit[name+'_new_pairs_known_users'] = len(targets)
        audit[name+'_positive_targets'] = int(targets.positive.sum())
        audit[name+'_positive_targets_unknown_product'] = int((targets.positive.eq(1)&~targets.product_seen_in_train).sum())
        audit[name+'_new_users_pairs'] = int((~df.author_id.isin(known_users)).sum())
    audit['train_end'] = t1.isoformat(); audit['validation_end'] = t2.isoformat()
    audit['train_global_mean'] = float(train.rating.mean())
    audit['train_positive_rate'] = float(train.positive.mean())
    audit['sources'] = [product_path,review_path]
    audit['catalog_snapshot_limitation'] = 'Precio, categoría y existencia del producto corresponden a la captura, no al pasado; no hay stock/precios históricos.'
    (report/'resumen_preparacion.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    guide = '''# Preparación Nexa v2
Separación global por fecha: primeros ~80% train, siguientes ~10% validación, últimos ~10% test. Cortes iguales permanecen en un período. Por período se conserva la última reseña por usuario-producto.
Perfiles y estadísticas calculados únicamente en train. Validación permite elegir parámetros; test queda reservado. No se utilizan helpfulness, rating/reviews del catálogo ni loves_count en las estadísticas.
Los objetivos de evaluación son productos nuevos para usuarios conocidos en train. Se conservan productos sin historial para contabilizar fallos de cobertura. Los usuarios nuevos se contabilizan por separado y necesitan evaluación de cold-start.
Propuesta de ranking: media ajustada (n*media + m*media_global)/(n+m). Versión por piel: combinar media del grupo con el ranking general, peso n_grupo/(n_grupo+m_piel). Elegir m y m_piel en validación.
Filtrar candidatos y objetivos con la misma categoría y presupuesto; excluir productos ya reseñados en train. Precio del catálogo en USD; presupuestos son escenarios simulados. No condicionar el presupuesto al precio del producto objetivo. Reportar cuántos objetivos quedan fuera por presupuesto, categoría o falta de soporte.
Para la prueba con presupuesto, solo objetivos positivos (4-5) que cumplen el filtro forman el denominador de Recall@10. Medir también NDCG@10, cobertura y tamaño de cohortes; productos no reseñados tienen relevancia desconocida, no negativa confirmada.
No usar stock de la captura para simular stock histórico. Una comparación con catálogo actual es retrospectiva y condicionada a esa captura, no un backtest fiel del catálogo histórico.
Entrenamiento y evaluación no deben reutilizar conclusiones del EDA como parámetros optimizados en test. El EDA previo vio el conjunto completo: el test no es evidencia completamente independiente de la selección del enfoque.
Estos archivos preparan la comparación; todavía no demuestran mejora del modelo ni ventas.
'''
    (report/'LEEME.md').write_text(guide,encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(f'Datos guardados en: {out.resolve()}\nInforme guardado en: {report.resolve()}')
    return audit

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--train-end',help='Corte ISO opcional, inclusivo para train')
    parser.add_argument('--validation-end',help='Corte ISO opcional, inclusivo para validación')
    args=parser.parse_args()
    prepare(args.project_root.resolve(),args.train_end,args.validation_end)
