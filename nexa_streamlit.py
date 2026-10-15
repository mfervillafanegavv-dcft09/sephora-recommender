"""Interfaz local de NEXA v4. Ejecutar desde la raíz: streamlit run app/nexa_streamlit.py"""
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from threading import Lock

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.nexa_recommender import NexaRecommender  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "nexa_v4"
HISTORY_PATH = PROJECT_ROOT.parent / "reports" / "inference_history.csv"
HISTORY_LOCK = Lock()
HISTORY_FIELDS = ("timestamp_utc", "category", "budget_usd", "mode_used")


def registrar_inferencia(category, budget_usd, mode_used):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_LOCK:
        write_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
        with HISTORY_PATH.open("a", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "category": category,
                    "budget_usd": budget_usd,
                    "mode_used": mode_used,
                }
            )

st.set_page_config(page_title="NEXA | Sephora", page_icon="✨", layout="wide")
st.title("✨ NEXA — Recomendador Sephora")
st.write("Explorá productos según tu categoría y presupuesto. Si tenés un usuario con historial, NEXA puede personalizar el ranking.")

@st.cache_resource(show_spinner="Cargando NEXA v4...")
def load_model():
    return NexaRecommender.load(MODEL_DIR)

try:
    model = load_model()
except Exception as exc:
    st.error("No se pudo cargar el modelo guardado en models/nexa_v4.")
    st.exception(exc)
    st.stop()

categories = sorted(model.categories)
if not categories:
    st.error("El modelo no contiene categorías disponibles.")
    st.stop()

with st.form("recommendation_form"):
    category = st.selectbox("Categoría", categories, index=categories.index("Moisturizers") if "Moisturizers" in categories else 0)
    budget = st.number_input("Presupuesto máximo (USD)", min_value=1.0, value=50.0, step=5.0)
    user_id = st.text_input("ID de usuario (opcional)", placeholder="Dejalo vacío para recomendaciones generales")
    top_n = st.slider("Cantidad de productos", min_value=1, max_value=20, value=10)
    submitted = st.form_submit_button("Ver recomendaciones", type="primary")

if submitted:
    try:
        result = model.recommend(
            category=category,
            budget_usd=float(budget),
            user_id=user_id.strip() or None,
            top_n=top_n,
        )
        registrar_inferencia(category, float(budget), result["mode_used"])
    except Exception as exc:
        st.error("No se pudieron generar o registrar las recomendaciones.")
        st.exception(exc)
    else:
        if result.get("status") != "ok":
            st.warning(f"No se generaron recomendaciones. Estado: {result.get('status', 'desconocido')}")
        else:
            mode = result.get("mode_used", "desconocido")
            st.success(f"{len(result['items'])} recomendaciones generadas · Modo: {mode}")
            if user_id.strip() and mode != "personalized":
                st.info("No se pudo utilizar un historial personalizado; NEXA aplicó el modo alternativo.")
            for i, item in enumerate(result["items"], start=1):
                with st.container(border=True):
                    st.subheader(f"{i}. {item.get('product_name', 'Producto')}")
                    st.write(f"**Marca:** {item.get('brand_name', 'Sin dato')}  ·  **Precio:** USD {float(item.get('price_usd', 0)):,.2f}")
                    if item.get("explanation"):
                        st.caption(item["explanation"])
            st.download_button(
                "Descargar recomendaciones (JSON)",
                data=__import__("json").dumps(result, ensure_ascii=False, indent=2),
                file_name="nexa_recomendaciones.json",
                mime="application/json",
            )

st.divider()
st.caption("NEXA v4 · Demostración local. Los rankings no equivalen a probabilidades de compra.")
