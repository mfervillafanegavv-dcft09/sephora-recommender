"""Interfaz Streamlit de NEXA v4 conectada a FastAPI.

Desde la raíz del proyecto, iniciar en dos terminales:
  python -m uvicorn api.main:app --reload
  python -m streamlit run app/nexa_streamlit.py
"""

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("NEXA_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 30

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "nexa_logo.png"
ICON_PATH = ASSETS_DIR / "nexa_icon.png"

st.set_page_config(
    page_title="NEXA | Recomendador Sephora",
    page_icon=str(ICON_PATH),
    layout="wide",
)
# st.image(str(LOGO_PATH), width=230)
st.title("NEXA — Data Solutions")
st.subheader("Recomendador de productos Sephora")
st.write(
    "Explorá productos según tu categoría y presupuesto. "
    "Si tenés un usuario con historial, NEXA puede personalizar el ranking."
)


import base64

logo_path = Path(__file__).resolve().parent / "assets" / "nexa_logo.png"

logo_base64 = base64.b64encode(logo_path.read_bytes()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{logo_base64}");
        background-position: center center;
        background-repeat: no-repeat;
        background-size: 450px;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(255, 255, 255, 0.88);
        pointer-events: none;
        z-index: 0;
    }}

    .stApp > header,
    .stApp > div {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def api_get(path):
    response = requests.get(f"{API_URL}{path}", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


try:
    health = api_get("/health")
    if not health.get("model_loaded"):
        st.error("La API está encendida, pero todavía no cargó el modelo NEXA v4.")
        st.stop()
    categories = api_get("/categories").get("categories", [])
except requests.exceptions.RequestException:
    st.error("No se pudo conectar con la API de NEXA.")
    st.info(
        "Abrí otra terminal en la carpeta principal del proyecto y ejecutá: "
        "`python -m uvicorn api.main:app --reload`. "
        f"Dirección configurada: `{API_URL}`"
    )
    st.stop()
except ValueError:
    st.error("La API devolvió una respuesta que no se pudo interpretar.")
    st.stop()

if not categories:
    st.error("La API no devolvió categorías disponibles.")
    st.stop()


# Selección del usuario fuera del formulario
st.markdown("#### ¿Cómo querés recibir tus recomendaciones?")

tipo_usuario = st.radio(
    "Tipo de usuario",
    options=["Soy un usuario nuevo", "Soy un usuario con historial"],
    help="El historial proviene de las reseñas utilizadas para entrenar NEXA.",
)

if tipo_usuario == "Soy un usuario con historial":
    st.caption(
        "Ingresá el ID (author_id) registrado en la base de reseñas de Sephora "
        "utilizada por NEXA. No es un ID generado por esta página ni una contraseña."
    )
    user_id = st.text_input(
        "ID de usuario en la base de datos",
        value="10000015049",
        help="Usuario de ejemplo con historial para probar la personalización.",
    )
    st.caption("Para la demostración podés usar el ID: 10000015049.")
else:
    user_id = ""
    st.caption(
        "Sin ID: NEXA mostrará recomendaciones generales según los filtros elegidos."
    )

# Filtros y botón de recomendaciones
with st.form("recommendation_form"):
    category = st.selectbox(
        "Categoría",
        categories,
        index=categories.index("Moisturizers") if "Moisturizers" in categories else 0,
    )
    budget = st.number_input(
        "Presupuesto máximo (USD)",
        min_value=1.0,
        value=50.0,
        step=5.0,
    )
    top_n = st.slider(
        "Cantidad de productos",
        min_value=1,
        max_value=20,
        value=10,
    )
    submitted = st.form_submit_button(
        "Ver recomendaciones",
        type="primary",
    )

if submitted:
    payload = {
        "category": category,
        "budget_usd": float(budget),
        "user_id": user_id.strip() or None,
        "top_n": top_n,
    }
    try:
        response = requests.post(
            f"{API_URL}/recommend", json=payload, timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get("detail", "Error de la API")
        except ValueError:
            detail = "Error de la API"
        st.error(f"No se pudieron generar recomendaciones: {detail}")
    except requests.exceptions.RequestException:
        st.error("No se pudo completar la solicitud. Comprobá que FastAPI siga funcionando.")
    except ValueError:
        st.error("La API devolvió una respuesta que no se pudo interpretar.")
    else:
        if result.get("status") != "ok":
            st.warning(
                "No se generaron recomendaciones. "
                f"Estado: {result.get('status', 'desconocido')}"
            )
        else:
            mode = result.get("mode_used", "desconocido")
            items = result.get("items", [])
            modo_visible = {
                "personalized": "Personalizado (con historial)",
                "general": "General",
            }.get(mode, mode)
            st.success(f"{len(items)} recomendaciones generadas · Modo: {modo_visible}")
            if user_id.strip() and mode != "personalized":
                st.info(
                    "No se pudo utilizar un historial personalizado; "
                    "NEXA aplicó el ranking general. El ID debe existir en los datos "
                    "y contar con historial útil para personalizar."
                )
            for i, item in enumerate(items, start=1):
                with st.container(border=True):
                    st.subheader(f"{i}. {item.get('product_name', 'Producto')}")
                    st.write(
                        f"**Marca:** {item.get('brand_name', 'Sin dato')}  ·  "
                        f"**Precio:** USD {float(item.get('price_usd', 0)):,.2f}"
                    )
                    if item.get("explanation"):
                        st.caption(item["explanation"])
            st.download_button(
                "Descargar recomendaciones (JSON)",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name="nexa_recomendaciones.json",
                mime="application/json",
            )

st.divider()
st.caption(
    "NEXA v4 · Interfaz conectada a FastAPI. "
    "Los rankings no equivalen a probabilidades de compra."
)
