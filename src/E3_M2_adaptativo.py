"""Nexa v4: ranking híbrido de atributos de productos e historial positivo.
Guardar en src/. Ejecutar: python src/E3_M2_adaptativo.py --project-root .
Requiere pandas, numpy, scipy y scikit-learn. Entradas generadas por E3_preparacion_datos.py.
Salida: reports/Etapa_03/diagnostico/evaluacion. No modifica modelos anteriores.
Presupuestos USD 30/60/100 y sin límite: escenarios, no presupuestos observados.
La categoría consultada se toma del producto objetivo: evaluación condicionada
por categoría, no predicción de intención ni de compras.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

BUDGETS = [30.,60.,100.,float('inf')]

def run(root,k=10):
    folder=root/'data'/'processed'/'Etapa_03'
    out=root/'reports'/'Etapa_03'/'M2_adaptativo'/'evaluacion';out.mkdir(parents=True,exist_ok=True)
    def read(name):
        path=folder/name
        if not path.exists():
            raise FileNotFoundError(f'Falta {path}. Ejecutar primero E3_preparacion_datos.py.')
        return pd.read_csv(path,dtype={'author_id':'string','product_id':'string'},low_memory=False)
    catalog=read('catalogo_skincare.csv')
    general=read('estadisticas_generales_train.csv')
    skin=read('estadisticas_piel_train.csv')
    history=read('train.csv')
    prior=float(history.rating.mean())
    known=set(general.product_id)
    excluded=history.groupby('author_id').product_id.agg(set).to_dict()
    catalog['price_usd']=pd.to_numeric(catalog.price_usd,errors='coerce')
    eligible=catalog.price_usd.gt(0)&np.isfinite(catalog.price_usd)&catalog.secondary_category.notna()
    catalog=catalog.loc[eligible].copy()
    catalog=catalog.merge(general,on='product_id',how='left',validate='one_to_one')
    catalog['n_users']=catalog.n_users.fillna(0)
    catalog['mean_rating']=catalog.mean_rating.fillna(prior)
    idset=set(catalog.product_id)
    discounts=1/np.log2(np.arange(2,k+2))
    cohorts={};audit=[]
    # Cohort is identical for all models/parameters. No test used for tuning.
    for split in ['validation','test']:
        targets=read('objetivos_'+split+'.csv')
        targets=targets.loc[targets.positive.eq(1)].copy()
        targets['skin_type']=targets.skin_type.fillna('(sin dato)')
        for budget in BUDGETS:
            mask=targets.product_id.isin(idset)&targets.price_usd.le(budget)&targets.secondary_category.notna()
            filtered=targets.loc[mask]
            groups=[]
            for (user,category,profile),df in filtered.groupby(['author_id','secondary_category','skin_type'],sort=False):
                relevant=set(df.product_id)
                groups.append((user,category,profile,relevant,relevant&known,relevant-known))
            cohorts[(split,budget)]=groups
            audit.append({'split':split,'budget_usd':budget,'positive_targets_total':len(targets),'positive_targets_eligible':len(filtered),'positive_targets_excluded':len(targets)-len(filtered),'queries_user_category':len(groups),'distinct_users':filtered.author_id.nunique(),'positive_targets_new_product':int((~filtered.product_id.isin(known)).sum())})
    pd.DataFrame(audit).to_csv(out/'cobertura_evaluacion.csv',index=False,encoding='utf-8-sig')
    # Fit the vocabulary only on products observed in training.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from scipy.sparse import csr_matrix
    raw=root/'data'/'processed'
    product_file=next((raw/n for n in ['products_clean.csv','products_clean(1).csv','products_clean.parquet','products_clean(1).parquet'] if (raw/n).exists()),None)
    if product_file is None:raise FileNotFoundError('Falta products_clean en data/processed.')
    metadata=pd.read_csv(product_file,low_memory=False) if product_file.suffix=='.csv' else pd.read_parquet(product_file)
    metadata['product_id']=metadata.product_id.astype('string')
    cols=['product_id']+[c for c in ['ingredients','highlights','tertiary_category'] if c in metadata]
    data=catalog[['product_id']].merge(metadata[cols],on='product_id',how='left',validate='one_to_one')
    def text(column):
        return data[column].fillna('').astype(str) if column in data else pd.Series('',index=data.index)
    documents=(text('highlights')+' '+text('highlights')+' '+text('tertiary_category')+' '+text('ingredients'))
    train_items=catalog.product_id.isin(known).to_numpy()
    vectorizer=TfidfVectorizer(min_df=2,max_features=8000,ngram_range=(1,2),dtype=np.float32,sublinear_tf=True)
    vectorizer.fit(documents[train_items])
    features=vectorizer.transform(documents)
    # Positive training ratings only. No validation/test ratings enter user profiles.
    query_users=sorted({q[0] for groups in cohorts.values() for q in groups})
    user_index={u:i for i,u in enumerate(query_users)}
    item_index={p:i for i,p in enumerate(catalog.product_id)}
    positive=history[history.rating.ge(4)&history.author_id.isin(query_users)&history.product_id.isin(item_index)]
    ui=positive.author_id.map(user_index).to_numpy();pi=positive.product_id.map(item_index).to_numpy()
    preference=csr_matrix((np.ones(len(positive),dtype=np.float32),(ui,pi)),shape=(len(query_users),len(catalog)))
    user_features=normalize(preference@features)
    # Dense item similarity scores only for users actually evaluated.
    similarity=(user_features@features.T).toarray().astype(np.float32)
    has_profile=np.asarray(preference.sum(axis=1)).ravel()>0
    support={'catalog_products':len(catalog),'ingredients_non_null':int(text('ingredients').ne('').sum()),'highlights_non_null':int(text('highlights').ne('').sum()),'evaluation_users':len(query_users),'users_with_positive_content_profile':int(has_profile.sum()),'tfidf_features':len(vectorizer.vocabulary_),'fit_on_training_products_only':True}
    (out/'auditoria_contenido.json').write_text(json.dumps(support,indent=2),encoding='utf-8')
    similarity=np.round(similarity,6)
    all_ids=catalog.product_id.to_numpy().astype(str)
    item_support=catalog.n_users.to_numpy()
    from scipy.stats import rankdata
    similarity_ranks={}
    # Weighted combination of percentile ranks within identical candidates.
    def rankings(m,mp=None):
        transition=0 if mp is None else mp
        pure_content=transition==-1
        base=(catalog.n_users.to_numpy()*catalog.mean_rating.to_numpy()+m*prior)/(catalog.n_users.to_numpy()+m)
        subsets={}
        for category,frame in catalog.groupby('secondary_category'):
            for budget in BUDGETS:
                idx=frame.index[frame.price_usd.le(budget)].to_numpy()
                subsets[(category,budget)]=(idx,rankdata(base[idx],method='average')/len(idx) if len(idx) else np.array([]))
        baseline_cache={}
        class UserRanking:
            def get(self,key,default=None):
                user,profile,category,budget=key
                if (category,budget) not in subsets:return []
                idx,brank=subsets[(category,budget)]
                if len(idx)==0:return []
                u=user_index[user]
                if transition==0 or not has_profile[u]:
                    if (category,budget) not in baseline_cache:
                        ids=all_ids[idx]
                        baseline_cache[(category,budget)]=ids[np.lexsort((ids,-brank))].tolist()
                    return baseline_cache[(category,budget)]
                if transition!=0 and has_profile[u]:
                    skey=(user,category,budget)
                    if skey not in similarity_ranks:
                        similarity_ranks[skey]=(rankdata(similarity[u,idx],method='average')/len(idx)).astype(np.float32)
                    srank=similarity_ranks[skey]
                    weight=np.ones(len(idx)) if pure_content else transition/(item_support[idx]+transition)
                    scores=(1-weight)*brank+weight*srank
                else:scores=brank
                ids=all_ids[idx]
                order=np.lexsort((ids,-scores))
                return ids[order].tolist()
        return UserRanking()
    def evaluate(cache,split,model,m,mp=None,details=False):
        rows=[];detail=[]
        for budget in BUDGETS:
            sums={name:np.zeros(3) for name in ['all','known_product','new_product']}
            count={name:0 for name in sums};recommended=set();slots=[];profiled=0
            for user,category,profile,all_rel,warm,cold in cohorts[(split,budget)]:
                rank=cache.get((user,profile,category,budget),[])
                seen=excluded.get(user,set());top=[]
                for pid in rank:
                    if pid not in seen:
                        top.append(pid)
                        if len(top)==k:break
                recommended.update(top);slots.append(len(top));profiled+=profile!='(sin dato)'
                for cohort,rel in [('all',all_rel),('known_product',warm),('new_product',cold)]:
                    if not rel:continue
                    hits=np.array([pid in rel for pid in top],dtype=float)
                    recall=hits.sum()/len(rel)
                    dcg=float((hits*discounts[:len(hits)]).sum());ideal=float(discounts[:min(k,len(rel))].sum())
                    ndcg=dcg/ideal;hit=float(hits.sum()>0)
                    sums[cohort]+=np.array([recall,ndcg,hit]);count[cohort]+=1
                    if details and cohort=='all':
                        detail.append({'author_id':user,'category':category,'skin_type':profile,'budget_usd':budget,'model':model,'recall':recall,'ndcg':ndcg,'hit':hit})
            nqueries=len(cohorts[(split,budget)])
            for cohort in sums:
                n=count[cohort];means=sums[cohort]/n if n else np.full(3,np.nan)
                rows.append({'split':split,'model':model,'m_general':m,'transition_users':mp,'budget_usd':budget,'cohort':cohort,'queries':n,'recall_at_k':means[0],'ndcg_at_k':means[1],'hit_rate_at_k':means[2],'distinct_recommended_products':len(recommended),'mean_recommendations':np.mean(slots) if slots else 0,'queries_with_skin_type':profiled,'queries_all':nqueries})
        return rows,detail
    grid=[]
    for m in [10,50,100]:
        rows,_=evaluate(rankings(m),'validation','baseline',m);grid+=rows
        print(f'Validación baseline: m={m}',flush=True)
    g=pd.DataFrame(grid)
    scores=g[g.cohort.eq('all')].groupby('m_general').ndcg_at_k.mean()
    if scores.notna().sum()==0:raise ValueError('No hay objetivos de validación elegibles.')
    best_m=int(scores.idxmax())
    # Same baseline prior strength for both versions isolates skin adjustment.
    for mp in [0,10,50,100]:
        rows,_=evaluate(rankings(best_m,mp),'validation','hibrido',best_m,mp);grid+=rows
        print(f'Validación híbrido: transicion={mp}',flush=True)
    ref,_=evaluate(rankings(best_m,-1),'validation','contenido_puro',best_m,-1);grid+=ref
    g=pd.DataFrame(grid);g.to_csv(out/'validacion_parametros.csv',index=False,encoding='utf-8-sig')
    best_mp=float(g[g.model.eq('hibrido')&g.cohort.eq('all')].groupby('transition_users').ndcg_at_k.mean().idxmax())
    results=[];details=[]
    for model,mp in [('baseline',None),('contenido_puro',-1),('hibrido',best_mp)]:
        rows,detail=evaluate(rankings(best_m,mp),'test',model,best_m,mp,True);results+=rows;details+=detail
    result=pd.DataFrame(results);result.to_csv(out/'metricas_test.csv',index=False,encoding='utf-8-sig')
    # Paired bootstrap across users (keeps correlated categories and budgets together).
    d=pd.DataFrame(details)
    differences=d.pivot(index=['author_id','category','skin_type','budget_usd'],columns='model',values='ndcg').reset_index()
    differences['delta']=differences.hibrido-differences.baseline
    differences.to_csv(out/'diferencias_pareadas_test.csv',index=False,encoding='utf-8-sig')
    per_user=differences.groupby('author_id').delta.agg(['sum','count'])
    rng=np.random.default_rng(42);boot=[]
    vals=per_user['sum'].to_numpy();counts=per_user['count'].to_numpy()
    for _ in range(500):
        indices=rng.integers(0,len(vals),len(vals));boot.append(vals[indices].sum()/counts[indices].sum())
    interval=np.quantile(boot,[.025,.975])
    summary={'k':k,'m_general':best_m,'transition_users':best_mp,'selection_metric':'Macro NDCG@k promediado entre escenarios de presupuesto en validación','test_paired_ndcg_delta':float(differences.delta.mean()),'bootstrap_user_cluster_95_interval':[float(x) for x in interval],'test_users':len(per_user),'budget_scenarios_usd':[30,60,100,'sin limite'],'rule':'peso_contenido=t/(n_usuarios_producto+t); t=0 es baseline; sin historial positivo del usuario se usa baseline', 'similarity_rounding_decimals':6,'limitations':['Categoría del producto objetivo simula consulta; no es una intención observada.','Precio y catálogo corresponden a una captura, no al catálogo histórico.','Usuarios nuevos no incluidos: requieren evaluación adicional.','Atributos del catálogo son de una captura posterior; no conocemos su disponibilidad histórica. Similitud de ingredientes no demuestra compatibilidad clínica.','Relevancia solo observada en reseñas, no ventas; productos no reseñados no son negativos confirmados.','EDA previo consultó todo el dataset; la prueba no es completamente independiente del diseño.','Intervalo combinado incluye consultas repetidas entre presupuestos; remuestreo agrupado por usuario.']}
    (out/'resumen_evaluacion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result[result.cohort.eq('all')][['model','budget_usd','queries','recall_at_k','ndcg_at_k']].to_string(index=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(f'Resultados guardados en: {out.resolve()}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--k',type=int,default=10)
    args=parser.parse_args()
    if args.k<1:parser.error('--k debe ser positivo')
    run(args.project_root.resolve(),args.k)
