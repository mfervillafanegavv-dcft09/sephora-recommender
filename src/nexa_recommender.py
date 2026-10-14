"""Nexa v4: inference for the evaluated history-based recommender.
No skin/ingredient preference bonuses; no product-reference mode.
"""
from pathlib import Path
import json,hashlib,platform
from functools import lru_cache
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import rankdata
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class NexaRecommender:
    def __init__(self,catalog,features,positive,seen,users,config):
        self.catalog=catalog.reset_index(drop=True)
        self.features=features.tocsr();self.positive=positive.tocsr();self.seen=seen.tocsr()
        self.users=list(users);self.user_index={u:i for i,u in enumerate(users)}
        self.config=config;self.ids=self.catalog.product_id.astype(str).to_numpy()
        self.n=self.catalog.n_users.fillna(0).to_numpy()
        self.categories=set(self.catalog.secondary_category)
        self.category_indices={c:np.flatnonzero(self.catalog.secondary_category.eq(c).to_numpy()) for c in self.categories}
        self.prices=self.catalog.price_usd.to_numpy()
        self.base=(self.n*self.catalog.mean_rating.fillna(config['global_mean']).to_numpy()+config['m_general']*config['global_mean'])/(self.n+config['m_general'])
        if features.shape[0]!=len(catalog) or positive.shape!=seen.shape or positive.shape[1]!=len(catalog) or positive.shape[0]!=len(users):raise ValueError('Inconsistent model dimensions.')
    @lru_cache(maxsize=256)
    def _profile_scores(self,user):
        row=self.positive.getrow(self.user_index[user])
        if not row.nnz:return None
        profile=normalize(row@self.features)
        return np.round((profile@self.features.T).toarray().ravel(),6)
    @lru_cache(maxsize=256)
    def _candidates(self,category,budget):
        idx=self.category_indices[category]
        idx=idx[self.prices[idx]<=budget]
        return idx,rankdata(self.base[idx],method='average')/len(idx) if len(idx) else np.array([])
    def recommend(self,category,budget_usd,user_id=None,top_n=10,mode='adaptativo',_ids_only=False):
        """Returns dict(status, mode_used, items, counts). Unknown users fall back.
        Percentile ranks are calculated BEFORE seen-item exclusion, matching evaluation.
        """
        if category not in self.categories:raise ValueError('Categoría desconocida.')
        if not np.isfinite(budget_usd) or budget_usd<=0:raise ValueError('Presupuesto debe ser finito y positivo.')
        if not isinstance(top_n,int) or not 1<=top_n<=100:raise ValueError('top_n debe estar entre 1 y 100.')
        if mode not in ['adaptativo','baseline','popularidad']:raise ValueError('Modo desconocido.')
        user=None if user_id is None else str(user_id)
        known=user in self.user_index
        similarity=self._profile_scores(user) if known and mode=='adaptativo' else None
        personalized=similarity is not None
        idx,general=self._candidates(category,float(budget_usd))
        if len(idx)==0 and _ids_only:return []
        if len(idx)==0:return {'status':'no_candidates','mode_used':'general','items':[],'candidates_before_exclusion':0,'eligible_after_exclusion':0}
        weights=np.zeros(len(idx))
        if mode=='popularidad':score=rankdata(self.n[idx],method='average')/len(idx)
        elif personalized:
            content=rankdata(similarity[idx],method='average')/len(idx)
            weights=self.config['transition_users']/(self.n[idx]+self.config['transition_users'])
            score=(1-weights)*general+weights*content
        else:score=general
        excluded=set(self.seen.getrow(self.user_index[user]).indices) if known else set()
        order=np.lexsort((self.ids[idx],-score))
        selected=[j for j in order if idx[j] not in excluded][:top_n]
        if _ids_only:return self.ids[idx[selected]].tolist()
        items=[]
        for j in selected:
            i=idx[j];p=self.catalog.iloc[i];n=int(self.n[i])
            reason='Ranking general de valoraciones con respaldo de opiniones.'
            if mode=='popularidad':reason='Orden por número de usuarios con reseñas en entrenamiento.'
            elif personalized:reason='Combina parecido textual con tus productos valorados positivamente y respaldo de opiniones.' if n else 'Sin reseñas de entrenamiento: usa parecido textual con tus productos valorados positivamente.'
            items.append({'product_id':str(p.product_id),'product_name':str(p.product_name),'brand_name':str(p.brand_name) if 'brand_name' in p and pd.notna(p.brand_name) else None,'category':str(p.secondary_category),'price_usd':float(p.price_usd),'training_reviewers':n,'mean_rating_train':float(p.mean_rating) if n and pd.notna(p.mean_rating) else None,'content_weight':float(weights[j]),'ranking_score':float(score[j]),'explanation':reason})
        return {'status':'ok' if items else 'no_unseen_candidates','mode_used':'popularidad' if mode=='popularidad' else ('personalized' if personalized else 'general'),'fallback_reason':None if personalized or mode!='adaptativo' else ('unknown_or_new_user' if not known else 'no_positive_history'),'items':items,'candidates_before_exclusion':len(idx),'eligible_after_exclusion':sum(i not in excluded for i in idx)}
    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        self.catalog.to_csv(directory/'catalog.csv',index=False,encoding='utf-8-sig')
        for name,matrix in [('features',self.features),('positive',self.positive),('seen',self.seen)]:sparse.save_npz(directory/(name+'.npz'),matrix)
        (directory/'users.json').write_text(json.dumps(self.users),encoding='utf-8')
        (directory/'config.json').write_text(json.dumps(self.config,ensure_ascii=False,indent=2),encoding='utf-8')
        files=[p for p in directory.iterdir() if p.name!='checksums.json' and p.is_file()]
        checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (directory/'checksums.json').write_text(json.dumps(checksums,indent=2))
    @classmethod
    def load(cls,directory):
        directory=Path(directory)
        checks=json.loads((directory/'checksums.json').read_text())
        for name,expected in checks.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise ValueError('Model file checksum mismatch: '+name)
        catalog=pd.read_csv(directory/'catalog.csv',dtype={'product_id':'string'},float_precision='round_trip')
        return cls(catalog,sparse.load_npz(directory/'features.npz'),sparse.load_npz(directory/'positive.npz'),sparse.load_npz(directory/'seen.npz'),json.loads((directory/'users.json').read_text()),json.loads((directory/'config.json').read_text()))

def build(project_root):
    root=Path(project_root);folder=root/'data'/'processed'/'recomendador_v2'
    cat=pd.read_csv(folder/'catalogo_skincare.csv',dtype={'product_id':'string'})
    stats=pd.read_csv(folder/'estadisticas_generales_train.csv',dtype={'product_id':'string'})
    history=pd.read_csv(folder/'train.csv',dtype={'author_id':'string','product_id':'string'},low_memory=False)
    if history.duplicated(['author_id','product_id']).any():raise ValueError('Train contiene pares repetidos.')
    if int(stats.n_users.sum())!=len(history):raise ValueError('Train incompleto o estadísticas de otra versión; n_users no coincide con filas de train.')
    summary_path=root/'reports'/'recomendador_v2'/'resumen_preparacion.json'
    if summary_path.exists():
        expected=json.loads(summary_path.read_text())['train_unique_pairs']
        if expected!=len(history):raise ValueError('Train no coincide con resumen_preparacion.')
    cat=cat[cat.price_usd.gt(0)&np.isfinite(cat.price_usd)&cat.secondary_category.notna()].copy()
    cat=cat.merge(stats,on='product_id',how='left',validate='one_to_one')
    cat['n_users']=cat.n_users.fillna(0).astype(int)
    raw=root/'data'/'processed';path=next((raw/n for n in ['products_clean.csv','products_clean(1).csv','products_clean.parquet','products_clean(1).parquet'] if (raw/n).exists()),None)
    if path is None:raise FileNotFoundError('Falta products_clean.')
    meta=pd.read_csv(path,dtype={'product_id':'string'},low_memory=False) if path.suffix=='.csv' else pd.read_parquet(path)
    data=cat[['product_id']].merge(meta[['product_id','highlights','ingredients','tertiary_category']],on='product_id',how='left',validate='one_to_one')
    docs=data.highlights.fillna('')+' '+data.highlights.fillna('')+' '+data.tertiary_category.fillna('')+' '+data.ingredients.fillna('')
    tfidf=TfidfVectorizer(min_df=2,max_features=8000,ngram_range=(1,2),dtype=np.float32,sublinear_tf=True)
    tfidf.fit(docs[cat.product_id.isin(stats.product_id)]);features=tfidf.transform(docs)
    users=sorted(history.author_id.dropna().unique().tolist());uid={u:i for i,u in enumerate(users)};pid={p:i for i,p in enumerate(cat.product_id)}
    h=history[history.product_id.isin(pid)].copy()
    def matrix(df):return sparse.csr_matrix((np.ones(len(df),dtype=np.float32),(df.author_id.map(uid).to_numpy(),df.product_id.map(pid).to_numpy())),shape=(len(users),len(cat)))
    config={'model':'Nexa v4 history-based','m_general':100,'transition_users':100,'global_mean':float(history.rating.mean()),'similarity_rounding_decimals':6,'score_is_probability':False,'scope':'Skincare discovery from positive training history; broad category and budget','evaluation_status':'exploratory; test previously inspected','versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__},'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'train.csv',folder/'estadisticas_generales_train.csv',folder/'catalogo_skincare.csv',path]}}
    return NexaRecommender(cat,features,matrix(h[h.rating.ge(4)]),matrix(h),users,config)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--output',type=Path);a=p.parse_args()
    model=build(a.project_root.resolve());dest=a.output or a.project_root/'models'/'nexa_v4';model.save(dest);print('Modelo guardado en:',dest.resolve())
