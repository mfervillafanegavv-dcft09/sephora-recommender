import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def main():
    # Define file path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    FEATURES_DIR = PROJECT_ROOT / "data" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("--- STARTING FEATURE ENGINEERING ---")

    
    # ==========================================
    # PRODUCT TRANSFORMATION (TF-IDF)
    # ==========================================


    print("1. Loading products and processing text...")
    products = pd.read_parquet(PROCESSED_DIR / "products_clean.parquet")

    # Filter the relevant text columns
    products['text_feature'] = (
        products['primary_category'].fillna('') + ' ' +
        products['highlights'].fillna('') + ' ' +
        products['ingredients'].fillna('')
    )
   
    # Convert the text into a numerical matrix of 500 keywords.
    '''Este fragmento de código realiza el proceso de vectorización de texto mediante la técnica TF-IDF (Term Frequency - Inverse Document Frequency). 
    En términos sencillos, sirve para transformar las descripciones textuales de los productos en números que los algoritmos de Machine Learning (como K-Means) puedan entender y procesar.
    Sugerencia de IA'''

    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X_products_tfidf = tfidf.fit_transform(products['text_feature'])
    
    #Guarda matriz TF-IDF
    sparse.save_npz(
        FEATURES_DIR / "products_tfidf.npz",
        X_products_tfidf
    )

    #Guarda el vector entrenado
    joblib.dump(
        tfidf,
        FEATURES_DIR / "tfidf_vectorizer.joblib"
    )

    #Guarda IDs respetando el mismo orden de la matriz
    products[['product_id']].to_parquet(
        FEATURES_DIR / "products_id_index.parquet", 
        index=False
    )

    print(f"   -> Vectorized products: {X_products_tfidf.shape[0]} products x {X_products_tfidf.shape[1]} keywords.")

   
    # ==========================================
    # USER CODIFICATION (ONE-HOT ENCODING)
    # ==========================================


    print("2. Loading users and encoding profiles...")
    user_profiles = pd.read_parquet(PROCESSED_DIR / "user_profiles.parquet")
    
    user_cols = ['skin_type', 'skin_tone', 'eye_color', 'hair_color']
    
    # Fill nulls with 'Unknown' to maintain the explicit category.
    user_profiles_filled = user_profiles[user_cols].fillna('Unknown')
    
    # Apply One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_users = ohe.fit_transform(user_profiles_filled)

    #Guardar encoder entrenado
    joblib.dump(
        ohe,
        FEATURES_DIR / "user_ohe_encoder.joblib"
    )

    df_users_encoded = pd.DataFrame(
        X_users, 
        columns=ohe.get_feature_names_out(user_cols)
    )

    df_users_encoded.insert(
        0, 
        'author_id', 
        user_profiles['author_id'].values
    )

    df_users_encoded.to_parquet(
        FEATURES_DIR / "users_features.parquet", 
        index=False
    )
    
    print(f"   -> Usuarios codificados: {df_users_encoded.shape[0]} usuarios x {df_users_encoded.shape[1] - 1} atributos binarios.")

    # ==========================================
    # FILTERING POSITIVE INTERACTIONS
    # ==========================================

    print("3. Filtering positive interactions...")
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    
    # Keep only high-rating reviews (rating >= 4 or recommended)
    pos_interactions = interactions[
        (interactions['rating'] >= 4) | (interactions['is_recommended'] == True)
    ][['author_id', 'product_id']].drop_duplicates()
    
    pos_interactions.to_parquet(FEATURES_DIR / "positive_interactions.parquet", index=False)
    print(f"   -> Positive interactions saved: {len(pos_interactions)} records.")

    print("\nDone! All feature files were saved to 'data/features/'.")

if __name__ == "__main__":
    main()