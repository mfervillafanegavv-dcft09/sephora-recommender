"""Nexa: compara ranking general y ranking por tipo de piel.
Guardar en src/. Ejecutar: python src/E3_diagnostico_inicial.py --project-root .
Requiere pandas y numpy. Entradas generadas por E3_preparacion_datos.py.
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
    out=root/'reports'/'Etapa_03'/'evaluacion';out.mkdir(parents=True,exist_ok=True)
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
    profiles=sorted(set(skin.skin_type.dropna())|{'(sin dato)'})
    def rankings(m,mp=None):
        c=catalog.copy()
        c['score']=(c.n_users*c.mean_rating+m*prior)/(c.n_users+m)
        cache={}
        for profile in profiles:
            ranked=c.copy()
            if mp is not None and profile!='(sin dato)':
                s=skin.loc[skin.skin_type.eq(profile),['product_id','n_users','mean_rating']].rename(columns={'n_users':'n_skin','mean_rating':'mean_skin'})
                ranked=ranked.merge(s,on='product_id',how='left',validate='one_to_one')
                ranked['n_skin']=ranked.n_skin.fillna(0)
                ranked['mean_skin']=ranked.mean_skin.fillna(ranked.score)
                ranked['score']=(ranked.n_skin*ranked.mean_skin+mp*ranked.score)/(ranked.n_skin+mp)
            for category,frame in ranked.groupby('secondary_category'):
                for budget in BUDGETS:
                    subset=frame.loc[frame.price_usd.le(budget)].sort_values(['score','product_id'],ascending=[False,True],kind='stable')
                    cache[(profile,category,budget)]=subset.product_id.tolist()
        return cache
    def evaluate(cache,split,model,m,mp=None,details=False):
        rows=[];detail=[]
        for budget in BUDGETS:
            sums={name:np.zeros(3) for name in ['all','known_product','new_product']}
            count={name:0 for name in sums};recommended=set();slots=[];profiled=0
            for user,category,profile,all_rel,warm,cold in cohorts[(split,budget)]:
                rank=cache.get((profile,category,budget),cache.get(('(sin dato)',category,budget),[]))
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
                    if details:
                        detail.append({'author_id':user,'category':category,'skin_type':profile,'budget_usd':budget,'model':model,'cohort':cohort,'recall':recall,'ndcg':ndcg,'hit':hit})
            nqueries=len(cohorts[(split,budget)])
            for cohort in sums:
                n=count[cohort];means=sums[cohort]/n if n else np.full(3,np.nan)
                rows.append({'split':split,'model':model,'m_general':m,'m_skin':mp,'budget_usd':budget,'cohort':cohort,'queries':n,'recall_at_k':means[0],'ndcg_at_k':means[1],'hit_rate_at_k':means[2],'distinct_recommended_products':len(recommended),'mean_recommendations':np.mean(slots) if slots else 0,'queries_with_skin_type':profiled,'queries_all':nqueries})
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
    for mp in [10,50,100,500]:
        rows,_=evaluate(rankings(best_m,mp),'validation','personalizado',best_m,mp);grid+=rows
        print(f'Validación personalizado: m_piel={mp}',flush=True)
    g=pd.DataFrame(grid);g.to_csv(out/'validacion_parametros.csv',index=False,encoding='utf-8-sig')
    best_mp=int(g[g.model.eq('personalizado')&g.cohort.eq('all')].groupby('m_skin').ndcg_at_k.mean().idxmax())
    results=[];details=[];diagnostic=[]
    for split in ['validation','test']:
        for model,mp in [('baseline',None),('personalizado',best_mp)]:
            _,detail=evaluate(rankings(best_m,mp),split,model,best_m,mp,True)
            diagnostic.extend(dict(row,split=split) for row in detail)
    diag=pd.DataFrame(diagnostic)
    category=diag.groupby(['split','category','budget_usd','cohort','model']).agg(queries=('ndcg','size'),users=('author_id','nunique'),recall_at_k=('recall','mean'),ndcg_at_k=('ndcg','mean'),hit_rate_at_k=('hit','mean')).reset_index()
    category.to_csv(out/'metricas_por_categoria.csv',index=False,encoding='utf-8-sig')
    comparison=category.pivot(index=['split','category','budget_usd','cohort'],columns='model',values=['queries','users','recall_at_k','ndcg_at_k','hit_rate_at_k'])
    comparison.columns=['_'.join(c) for c in comparison.columns]
    comparison=comparison.reset_index()
    comparison['delta_ndcg']=comparison.ndcg_at_k_personalizado-comparison.ndcg_at_k_baseline
    comparison['delta_recall']=comparison.recall_at_k_personalizado-comparison.recall_at_k_baseline
    comparison.to_csv(out/'comparacion_por_categoria.csv',index=False,encoding='utf-8-sig')
    for model,mp in [('baseline',None),('personalizado',best_mp)]:
        rows,detail=evaluate(rankings(best_m,mp),'test',model,best_m,mp,True);results+=rows;details+=detail
    result=pd.DataFrame(results);result.to_csv(out/'metricas_test.csv',index=False,encoding='utf-8-sig')
    # Paired bootstrap across users (keeps correlated categories and budgets together).
    d=pd.DataFrame(details)
    d=d[d.cohort.eq('all')]
    differences=d.pivot(index=['author_id','category','skin_type','budget_usd'],columns='model',values='ndcg').reset_index()
    differences['delta']=differences.personalizado-differences.baseline
    per_user=differences.groupby('author_id').delta.agg(['sum','count'])
    rng=np.random.default_rng(42);boot=[]
    vals=per_user['sum'].to_numpy();counts=per_user['count'].to_numpy()
    for _ in range(500):
        indices=rng.integers(0,len(vals),len(vals));boot.append(vals[indices].sum()/counts[indices].sum())
    interval=np.quantile(boot,[.025,.975])
    summary={'k':k,'m_general':best_m,'m_skin':best_mp,'selection_metric':'Macro NDCG@k promediado entre escenarios de presupuesto en validación','test_paired_ndcg_delta':float(differences.delta.mean()),'bootstrap_user_cluster_95_interval':[float(x) for x in interval],'test_users':len(per_user),'budget_scenarios_usd':[30,60,100,'sin limite'],'limitations':['Categoría del producto objetivo simula consulta; no es una intención observada.','Precio y catálogo corresponden a una captura, no al catálogo histórico.','Usuarios nuevos no incluidos: requieren evaluación adicional.','Productos sin reseñas se puntúan con la media global; no se afirma personalización para ellos.','Relevancia solo observada en reseñas, no ventas; productos no reseñados no son negativos confirmados.','EDA previo consultó todo el dataset; la prueba no es completamente independiente del diseño.','Intervalo combinado incluye consultas repetidas entre presupuestos; remuestreo agrupado por usuario.']}
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
