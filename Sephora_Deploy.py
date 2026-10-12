from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import logging
import math
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sephora_common as sc
from sephora_common import PROFILE_COLS

K_RECOMENDACION = 500
ALPHA = 0.5
TOP_N = 10
SEED = 42
N_VALIDACION = 500
N_TEST = 500

PERFIL_EJEMPLO = {
    "skin_type": "oily",
    "skin_tone": "dark",
    "eye_color": "blue",
    "hair_color": "brown",
}

NOTA_MODELO = (
    "La puntuación indica el orden relativo del recomendador; no demuestra "
    "eficacia ni compatibilidad dermatológica."
)


def valor_texto(valor):
    if valor is None or pd.isna(valor):
        return None
    texto = str(valor).strip()
    return texto if texto else None


def valor_numerico(valor):
    if valor is None or pd.isna(valor):
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) else None


def json_seguro(objeto):
    return json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def resolver_raiz():
    archivo = Path(__file__).resolve()
    for candidato in (archivo.parent, *archivo.parents):
        if (candidato / "data" / "processed").is_dir():
            return candidato
    return sc.find_project_root()


def configurar_registro(directorio):
    directorio.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sephora_deploy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    manejador = logging.FileHandler(directorio / "deploy.log", encoding="utf-8")
    manejador.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(manejador)
    return logger


class RecomendadorPerfilPopularidad:
    def __init__(self, contexto):
        self.contexto = contexto
        self.perfiles = contexto.profiles.loc[sorted(contexto.train.author_id.unique())]
        self.codificador = DictVectorizer(sparse=True)
        registros = self.perfiles.to_dict("records")
        self.codificador.fit(registros + [{col: "Unknown" for col in PROFILE_COLS}])
        self.valores_permitidos = {
            col: set(self.perfiles[col].dropna().astype(str)) | {"Unknown"}
            for col in PROFILE_COLS
        }
        self.grupos = {}
        for uid, fila in self.perfiles.iterrows():
            firma = tuple(str(fila[col]) for col in PROFILE_COLS)
            self.grupos.setdefault(firma, []).append(uid)
        self.firmas = list(self.grupos)
        self.matriz_perfiles = self.codificador.transform([
            dict(zip(PROFILE_COLS, firma)) for firma in self.firmas
        ])
        positivas = contexto.train[contexto.train.positive]
        self.interacciones_positivas = (
            positivas.groupby("author_id").product_id.apply(list).to_dict()
        )
        popularidad = positivas.groupby("product_id").size()
        maximo = float(popularidad.max()) if not popularidad.empty else 1.0
        self.popularidad = (popularidad / maximo).to_dict()
        self.numero_usuarios = len(self.perfiles)
        self.catalogo = contexto.catalog.copy()
        self.catalogo["product_id"] = self.catalogo.product_id.astype(str).str.strip()
        self.catalogo = self.catalogo.drop_duplicates("product_id").set_index(
            "product_id", drop=False
        )
        if self.numero_usuarios < K_RECOMENDACION:
            raise ValueError(
                f"El modelo requiere al menos {K_RECOMENDACION} usuarios de entrenamiento; "
                f"solo hay {self.numero_usuarios}."
            )

    def validar_perfil(self, perfil, estricto=False):
        if not isinstance(perfil, dict):
            raise TypeError("El perfil debe ser un objeto JSON/diccionario.")
        campos_extra = sorted(set(perfil) - set(PROFILE_COLS))
        if campos_extra:
            raise ValueError(f"Campos no reconocidos en el perfil: {campos_extra}")

        resultado = {}
        advertencias = []
        for columna in PROFILE_COLS:
            valor = perfil.get(columna, "Unknown")
            if valor is None:
                valor = "Unknown"
            if not isinstance(valor, str):
                raise TypeError(f"El campo '{columna}' debe ser texto o null.")
            valor = valor.strip()
            if not valor or valor.casefold() in {"unknown", "none", "nan", "<na>"}:
                resultado[columna] = "Unknown"
                continue
            equivalencias = {
                opcion.casefold(): opcion
                for opcion in self.valores_permitidos[columna]
            }
            coincidencia = equivalencias.get(valor.casefold())
            if coincidencia is None:
                if estricto:
                    ejemplos = sorted(self.valores_permitidos[columna])[:12]
                    raise ValueError(
                        f"Valor no reconocido para '{columna}': '{valor}'. "
                        f"Algunos valores conocidos: {ejemplos}"
                    )
                resultado[columna] = "Unknown"
                advertencias.append(
                    f"'{columna}={valor}' no aparece en entrenamiento; se trató como Unknown."
                )
            else:
                resultado[columna] = coincidencia
        return resultado, advertencias

    def vecinos(self, perfil, k):
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= self.numero_usuarios:
            raise ValueError(f"k debe ser un entero entre 1 y {self.numero_usuarios}.")
        firma = tuple(perfil[col] for col in PROFILE_COLS)
        consulta = self.codificador.transform([dict(zip(PROFILE_COLS, firma))])
        similitudes = cosine_similarity(consulta, self.matriz_perfiles).ravel()
        orden = np.argsort(-similitudes, kind="stable")
        vecinos = []
        for indice in orden:
            similitud = float(similitudes[indice])
            for uid in self.grupos[self.firmas[indice]]:
                vecinos.append((uid, similitud))
                if len(vecinos) >= k:
                    return vecinos
        return vecinos

    def recomendar(self, perfil, top_n=TOP_N, k=K_RECOMENDACION, alpha=ALPHA, estricto=False):
        if not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= 100:
            raise ValueError("top_n debe ser un entero entre 1 y 100.")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 <= alpha <= 1:
            raise ValueError("alpha debe ser un número entre 0 y 1.")
        perfil, advertencias = self.validar_perfil(perfil, estricto=estricto)
        votos_perfil = Counter()
        for uid, similitud in self.vecinos(perfil, k):
            votos_perfil.update({
                pid: similitud * cantidad
                for pid, cantidad in Counter(self.interacciones_positivas.get(uid, [])).items()
            })
        maximo_perfil = max(votos_perfil.values()) if votos_perfil else 1.0
        productos = set(votos_perfil) | set(self.popularidad)
        puntuaciones = []
        for pid in productos:
            puntuacion_perfil = votos_perfil.get(pid, 0.0) / maximo_perfil
            puntuacion_popularidad = float(self.popularidad.get(pid, 0.0))
            puntuacion_final = alpha * puntuacion_perfil + (1 - alpha) * puntuacion_popularidad
            puntuaciones.append({"product_id": str(pid), "score": float(puntuacion_final)})
        puntuaciones.sort(key=lambda item: (-item["score"], item["product_id"]))
        filas = []
        for item in puntuaciones:
            pid = item["product_id"]
            if pid not in self.catalogo.index:
                continue
            producto = self.catalogo.loc[pid]
            filas.append({
                "position": len(filas) + 1,
                "product_id": pid,
                "product_name": valor_texto(producto.get("product_name")),
                "brand_name": valor_texto(producto.get("brand_name")),
                "primary_category": valor_texto(producto.get("primary_category")),
                "secondary_category": valor_texto(producto.get("secondary_category")),
                "price_usd": valor_numerico(producto.get("price_usd")),
                "score": round(item["score"], 6),
            })
            if len(filas) == top_n:
                break
        return {
            "modelo": "ProfilePopularity",
            "configuracion": {"k": k, "alpha": float(alpha)},
            "perfil": perfil,
            "cantidad": len(filas),
            "recomendaciones": filas,
            "advertencias": advertencias,
            "nota": NOTA_MODELO,
        }


def cargar_recomendador():
    raiz = resolver_raiz()
    directorio = raiz / "data" / "processed"
    faltantes = [
        nombre for nombre in ("reviews_clean", "products_clean")
        if not (directorio / f"{nombre}.parquet").exists()
        and not (directorio / f"{nombre}.csv").exists()
    ]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan archivos en {directorio}: {', '.join(faltantes)}"
        )
    contexto = sc.load_context(n_validation=N_VALIDACION, n_test=N_TEST, seed=SEED)
    recomendador = RecomendadorPerfilPopularidad(contexto)
    return raiz, contexto, recomendador


def cargar_perfiles(args):
    if args.profile_file:
        ruta = Path(args.profile_file).expanduser().resolve()
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo de perfiles: {ruta}")
        contenido = json.loads(ruta.read_text(encoding="utf-8-sig"))
    elif args.profile_json:
        contenido = json.loads(args.profile_json)
    else:
        contenido = PERFIL_EJEMPLO
    perfiles = contenido if isinstance(contenido, list) else [contenido]
    if not perfiles:
        raise ValueError("La lista de perfiles está vacía.")
    if not all(isinstance(perfil, dict) for perfil in perfiles):
        raise TypeError("La entrada debe ser un perfil JSON o una lista de perfiles JSON.")
    return perfiles


def guardar_json(ruta, contenido):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json_seguro(contenido), encoding="utf-8")
    temporal.replace(ruta)


def registrar_ejecucion(ruta, evento):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(evento, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def ejecutar_pruebas(recomendador):
    resultados = []
    perfiles_base = [PERFIL_EJEMPLO]
    for columna in PROFILE_COLS:
        conocidos = sorted(recomendador.valores_permitidos[columna] - {"Unknown"})
        if conocidos:
            alternativo = dict(PERFIL_EJEMPLO)
            alternativo[columna] = conocidos[0]
            perfiles_base.append(alternativo)
    for indice, perfil in enumerate(perfiles_base, start=1):
        inicio = time.perf_counter()
        resultado = recomendador.recomendar(perfil, top_n=TOP_N)
        assert resultado["cantidad"] <= TOP_N
        assert len(resultado["recomendaciones"]) == resultado["cantidad"]
        ids = [fila["product_id"] for fila in resultado["recomendaciones"]]
        assert len(ids) == len(set(ids)), "Se encontraron productos repetidos."
        assert all(fila["score"] >= 0 for fila in resultado["recomendaciones"])
        resultados.append({
            "prueba": f"perfil_{indice}",
            "estado": "correcto",
            "cantidad": resultado["cantidad"],
            "duracion_ms": round((time.perf_counter() - inicio) * 1000, 2),
        })
    casos_invalidos = [
        (None, TypeError),
        ({"skin_type": 123}, TypeError),
        ({"campo_inexistente": "valor"}, ValueError),
    ]
    for perfil, error_esperado in casos_invalidos:
        try:
            recomendador.recomendar(perfil)
        except error_esperado:
            resultados.append({"prueba": f"entrada_invalida_{len(resultados) + 1}", "estado": "correcto"})
        else:
            raise AssertionError(f"No se rechazó una entrada inválida: {perfil}")
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "estado": "correcto",
        "pruebas_aprobadas": len(resultados),
        "resultados": resultados,
    }


def crear_argumentos():
    parser = argparse.ArgumentParser(
        description="Genera recomendaciones Sephora con ProfilePopularity, sin API."
    )
    parser.add_argument("--profile-json", help="Perfil o lista de perfiles en formato JSON.")
    parser.add_argument("--profile-file", help="Ruta a un archivo JSON con un perfil o una lista de perfiles.")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="Cantidad de productos por perfil (1-100).")
    parser.add_argument("--k", type=int, default=K_RECOMENDACION, help="Número de usuarios vecinos; por defecto 500.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Peso del perfil entre 0 y 1; por defecto 0.5.")
    parser.add_argument("--strict-profile", action="store_true", help="Rechaza valores de perfil que no aparezcan en entrenamiento.")
    parser.add_argument("--self-test", action="store_true", help="Ejecuta pruebas de perfiles y validación de entradas.")
    parser.add_argument("--output", help="Ruta de salida JSON; por defecto artifacts/sephora_deploy/recomendaciones.json.")
    return parser


def main():
    args = crear_argumentos().parse_args()
    inicio_total = time.perf_counter()
    raiz = resolver_raiz()
    directorio_salida = raiz / "artifacts" / "sephora_deploy"
    logger = configurar_registro(directorio_salida)
    ruta_metricas = directorio_salida / "metricas_ejecucion.jsonl"
    evento = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": "ProfilePopularity",
        "estado": "error",
    }
    try:
        raiz, contexto, recomendador = cargar_recomendador()
        evento["protocol_id"] = contexto.protocol_id
        evento["usuarios_entrenamiento"] = recomendador.numero_usuarios
        evento["productos_catalogo"] = len(recomendador.catalogo)
        if args.self_test:
            pruebas = ejecutar_pruebas(recomendador)
            pruebas["protocol_id"] = contexto.protocol_id
            guardar_json(directorio_salida / "resultado_pruebas.json", pruebas)
            print(json_seguro(pruebas))
            evento["estado"] = "correcto"
            evento["modo"] = "self_test"
            evento["pruebas_aprobadas"] = pruebas["pruebas_aprobadas"]
            return 0

        perfiles = cargar_perfiles(args)
        resultados = []
        duraciones = []
        for perfil in perfiles:
            inicio = time.perf_counter()
            resultado = recomendador.recomendar(
                perfil,
                top_n=args.top_n,
                k=args.k,
                alpha=args.alpha,
                estricto=args.strict_profile,
            )
            duraciones.append((time.perf_counter() - inicio) * 1000)
            resultados.append(resultado)
        salida = resultados[0] if len(resultados) == 1 else {
            "modelo": "ProfilePopularity",
            "cantidad_perfiles": len(resultados),
            "resultados": resultados,
        }
        ruta_json = Path(args.output).expanduser() if args.output else (
            directorio_salida / "recomendaciones.json"
        )
        if not ruta_json.is_absolute():
            ruta_json = raiz / ruta_json
        guardar_json(ruta_json, salida)
        print(json_seguro(salida))
        print(f"\nRecomendaciones guardadas en: {ruta_json.resolve()}")
        evento.update({
            "estado": "correcto",
            "modo": "recomendacion",
            "cantidad_perfiles": len(resultados),
            "top_n_solicitado": args.top_n,
            "k": args.k,
            "alpha": args.alpha,
            "duracion_recomendacion_ms_total": round(sum(duraciones), 2),
            "duracion_recomendacion_ms_promedio": round(sum(duraciones) / len(duraciones), 2),
            "resultados_vacios": sum(r["cantidad"] == 0 for r in resultados),
            "salida": str(ruta_json.resolve()),
        })
        logger.info("Ejecución completada: %s", json.dumps(evento, ensure_ascii=False))
        return 0
    except Exception as error:
        evento["mensaje_error"] = str(error)
        logger.exception("Falló la ejecución del recomendador")
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        evento["duracion_total_ms"] = round((time.perf_counter() - inicio_total) * 1000, 2)
        try:
            registrar_ejecucion(ruta_metricas, evento)
        except Exception:
            logger.exception("No fue posible guardar las métricas de ejecución")


if __name__ == "__main__":
    raise SystemExit(main())
