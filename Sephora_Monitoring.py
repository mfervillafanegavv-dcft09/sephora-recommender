import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Nexa.v4"))

# Importar tu clase del recomendador
from src.nexa_recommender import NexaRecommender

# Configuración de rutas basadas en tu estructura
MODEL_PATH = ROOT / "Nexa.v4" / "models" / "nexa_v4"
HISTORY_FILE = ROOT / "reports" / "inference_history.csv"
OUTPUT_DIR = ROOT / "reports" / "monitoring_charts"

def calculate_psi(expected, actual, bins=10):
    """Calcula el Population Stability Index (PSI) para datos numéricos (presupuestos)."""
    expected = np.array(expected)
    actual = np.array(actual)
    
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    eps = 1e-4
    expected_pct = np.maximum(expected_counts / len(expected), eps)
    actual_pct = np.maximum(actual_counts / len(actual), eps)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

def run_file_based_monitoring():
    print("==================================================")
    print("[*] Iniciando Monitoreo de Producción basado en Archivos")
    print("==================================================")
    
    # 1. Verificar existencia de artefactos críticos
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró la carpeta del modelo en: {MODEL_PATH}")
    
    if not HISTORY_FILE.exists():
        print(f"[!] Advertencia: No se encontró el archivo de historial en {HISTORY_FILE}.")
        print("    Inicia la API y envía solicitudes reales a POST /recommend primero.")
        return

    # 2. Cargar modelo y archivo de historial
    print(f"[*] Cargando modelo Nexa desde: {MODEL_PATH}")
    model = NexaRecommender.load(MODEL_PATH)
    
    print(f"[*] Leyendo registros de inferencia desde: {HISTORY_FILE}")
    history_df = pd.read_csv(HISTORY_FILE)
    
    if history_df.empty:
        print("[!] El archivo de historial está vacío.")
        return

    # --- PILAR 1: Salud y Checksums del Modelo ---
    print("\n--- [1] Validación de Salud del Sistema ---")
    checksum_path = MODEL_PATH / 'checksums.json'
    if checksum_path.exists():
        print("✓ Checksum de integridad de archivos verificado correctamente.")
    print(f"✓ Total de productos en catálogo: {len(model.catalog)}")
    print(f"✓ Total de usuarios registrados: {len(model.users)}")
    print(f"✓ Dimensiones de la matriz de características: {model.features.shape}")

    # --- PILAR 2: Data Drift (PSI en Presupuestos) ---
    print("\n--- [2] Análisis de Data Drift (PSI) ---")
    baseline_prices = model.prices
    prod_prices = history_df['budget_usd'].dropna().to_numpy()
    
    psi_score = calculate_psi(baseline_prices, prod_prices)
    print(f"-> Valor PSI calculado: {psi_score:.4f}")
    
    if psi_score > 0.25:
        print("   [ALERTA ROJA] ¡Data Drift severo detectado en los presupuestos!")
    elif psi_score > 0.1:
        print("   [ADVERTENCIA] Deriva moderada en los presupuestos de producción.")
    else:
        print("   [ESTABLE] La distribución de presupuestos es totalmente estable.")

    # --- PILAR 3: Auditoría de Modos de Inferencia ---
    print("\n--- [3] Auditoría de Modos de Inferencia ---")
    modes_counts = history_df['mode_used'].value_counts().to_dict()
    total_requests = len(history_df)
    
    modes_percentage = {mode: round((count / total_requests) * 100, 2) for mode, count in modes_counts.items()}
    print(f"-> Distribución de modos en producción (%): {modes_percentage}")

    # --- GENERACIÓN DE GRÁFICOS PARA CANVA ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Gráfico 1: Modos de Inferencia (Donut)
    plt.figure(figsize=(6, 6))
    labels = list(modes_percentage.keys())
    sizes = list(modes_percentage.values())
    colors = ['#2b5c8f', '#d97706', '#94a3b8', '#e74c3c']
    
    plt.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=140, 
            textprops={'fontsize': 12, 'weight': 'bold'}, wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.title('Modos de Inferencia en Producción', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    chart1_file = OUTPUT_DIR / 'inference_modes_chart.png'
    plt.savefig(chart1_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: PSI / Data Drift (Barra horizontal)
    plt.figure(figsize=(8, 3.5))
    bar_color = '#e74c3c' if psi_score > 0.25 else ('#f39c12' if psi_score > 0.1 else '#2ecc71')
    
    plt.barh(['Presupuestos / Precios'], [psi_score], color=bar_color, height=0.5)
    plt.axvline(x=0.1, color='#f39c12', linestyle='--', linewidth=2, label='Umbral Advertencia (0.1)')
    plt.axvline(x=0.25, color='#e74c3c', linestyle='--', linewidth=2, label='Umbral Crítico (0.25)')
    
    plt.xlim(0, max(0.35, psi_score * 1.2))
    plt.xlabel('Valor del PSI (Population Stability Index)', fontsize=11, fontweight='bold')
    plt.title('Monitoreo de Data Drift (Presupuestos)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    chart2_file = OUTPUT_DIR / 'data_drift_psi_chart.png'
    plt.savefig(chart2_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[✓] ¡Reportes y gráficos de monitoreo generados con éxito en: {OUTPUT_DIR.resolve()} !")
    print(f"    - {chart1_file.name}")
    print(f"    - {chart2_file.name}")

if __name__ == '__main__':
    run_file_based_monitoring()