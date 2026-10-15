import csv
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ubicación principal del proyecto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Nexa.v4"))

from src.nexa_recommender import NexaRecommender

MODEL_PATH = ROOT / "Nexa.v4" / "models" / "nexa_v4"
HISTORY_PATH = ROOT / "reports" / "inference_history.csv"
HISTORY_LOCK = Lock()
HISTORY_FIELDS = ("timestamp_utc", "category", "budget_usd", "mode_used")

# Cargamos el modelo una sola vez al iniciar la API
modelo = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo
    modelo = NexaRecommender.load(MODEL_PATH)
    yield
    modelo = None


app = FastAPI(
    title="NEXA API",
    description="API del sistema recomendador de productos Sephora",
    version="1.0.0",
    lifespan=lifespan,
)


class RecommendationRequest(BaseModel):
    category: str = Field(..., min_length=1)
    budget_usd: float = Field(..., gt=0)
    user_id: str | None = None
    top_n: int = Field(default=10, ge=1, le=100)


def registrar_inferencia(datos: RecommendationRequest, resultado: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_LOCK:
        write_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
        with HISTORY_PATH.open("a", newline="", encoding="utf-8") as archivo:
            writer = csv.DictWriter(archivo, fieldnames=HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "category": datos.category,
                    "budget_usd": datos.budget_usd,
                    "mode_used": resultado["mode_used"],
                }
            )


@app.get("/")
def inicio():
    return {
        "mensaje": "Bienvenido a NEXA API",
        "modelo": "NEXA v4",
        "documentacion": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if modelo is not None else "unavailable",
        "model_loaded": modelo is not None,
    }


@app.get("/categories")
def categorias():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    return {
        "categories": sorted(modelo.categories)
    }


@app.post("/recommend")
def recomendar(datos: RecommendationRequest):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        resultado = modelo.recommend(
            category=datos.category,
            budget_usd=datos.budget_usd,
            user_id=datos.user_id,
            top_n=datos.top_n,
        )
        try:
            registrar_inferencia(datos, resultado)
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail="No se pudo guardar el historial de inferencias.",
            ) from error
        return resultado

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))