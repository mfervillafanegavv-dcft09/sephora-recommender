"""CLI de NEXA v4: carga artefactos existentes y genera recomendaciones JSON.

Ejemplos (desde la raiz del proyecto):
    python app/nexa_deploy.py --category Moisturizers --budget 50
    python app/nexa_deploy.py --category Moisturizers --budget 50 --user-id 10000015049
    python app/nexa_deploy.py --self-test
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.nexa_recommender import NexaRecommender  # noqa: E402

MODEL_DIR = PROJECT_ROOT / 'models' / 'nexa_v4'
OUTPUT_DIR = PROJECT_ROOT / 'artifacts' / 'nexa_deploy'


def json_text(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json_text(data), encoding='utf-8')
    temp.replace(path)


def make_logger():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('nexa_deploy')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(OUTPUT_DIR / 'deploy.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(handler)
    return logger


def self_test(model):
    if not model.categories:
        raise AssertionError('El catálogo no contiene categorías.')
    cat = 'Moisturizers' if 'Moisturizers' in model.categories else sorted(model.categories)[0]
    idx = model.category_indices[cat]
    if not len(idx):
        raise AssertionError('La categoría de prueba no contiene productos.')
    budget = float(max(model.prices[idx]))
    if not math.isfinite(budget) or budget <= 0:
        raise AssertionError('No hay presupuesto válido para la prueba.')
    general = model.recommend(cat, budget, top_n=10)
    assert general['status'] == 'ok' and general['mode_used'] == 'general'
    assert len(general['items']) == len({p['product_id'] for p in general['items']})
    assert all(p['price_usd'] <= budget for p in general['items'])
    candidate_users = model.positive.getnnz(axis=1).nonzero()[0]
    if not len(candidate_users):
        raise AssertionError('No hay usuarios con historial positivo.')
    user = model.users[int(candidate_users[0])]
    personalized = model.recommend(cat, budget, user_id=user, top_n=10)
    assert personalized['status'] == 'ok' and personalized['mode_used'] == 'personalized'
    assert len(personalized['items']) == len({p['product_id'] for p in personalized['items']})
    return {'estado': 'correcto', 'categoria': cat, 'presupuesto_usd': budget,
            'usuario_prueba': user, 'general_items': len(general['items']),
            'personalized_items': len(personalized['items'])}


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy CLI de NEXA v4 (sin API).')
    parser.add_argument('--category', help='Categoría exacta, por ejemplo Moisturizers.')
    parser.add_argument('--budget', type=float, help='Presupuesto máximo en USD.')
    parser.add_argument('--user-id', help='ID opcional de un usuario con historial.')
    parser.add_argument('--top-n', type=int, default=10, help='Número de recomendaciones (1 a 100).')
    parser.add_argument('--mode', choices=['adaptativo', 'baseline', 'popularidad'], default='adaptativo')
    parser.add_argument('--output', type=Path, help='Ruta del archivo JSON de salida.')
    parser.add_argument('--self-test', action='store_true', help='Prueba los modos general y personalizado.')
    args = parser.parse_args()
    if not args.self_test and (args.category is None or args.budget is None):
        parser.error('Para recomendar, indicá --category y --budget.')
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    logger = make_logger()
    event = {'fecha_utc': datetime.now(timezone.utc).isoformat(),
             'modelo': 'Nexa v4 history-based', 'estado': 'error'}
    try:
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(f'No se encontró el modelo guardado en {MODEL_DIR}.')
        model = NexaRecommender.load(MODEL_DIR)
        if args.self_test:
            result = self_test(model)
            output = OUTPUT_DIR / 'resultado_pruebas.json'
            event['modo'] = 'self_test'
        else:
            result = model.recommend(category=args.category, budget_usd=args.budget,
                                     user_id=args.user_id, top_n=args.top_n, mode=args.mode)
            output = args.output or OUTPUT_DIR / 'recomendaciones.json'
            if not output.is_absolute():
                output = PROJECT_ROOT / output
            event.update({'modo': result['mode_used'], 'categoria': args.category,
                          'presupuesto_usd': args.budget, 'usuario': args.user_id,
                          'cantidad': len(result['items']), 'estado_recomendacion': result['status']})
        save_json(output, result)
        print(json_text(result))
        print(f'\nArchivo guardado en: {output.resolve()}')
        event['estado'] = 'correcto'
        logger.info('Ejecución correcta: %s', json.dumps(event, ensure_ascii=False))
        return 0
    except Exception as exc:
        event['mensaje_error'] = str(exc)
        logger.exception('Error durante la ejecución')
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        event['duracion_total_ms'] = round((time.perf_counter() - started) * 1000, 2)
        try:
            with (OUTPUT_DIR / 'metricas_ejecucion.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        except OSError:
            logger.exception('No se pudo guardar el registro de ejecución')


if __name__ == '__main__':
    raise SystemExit(main())
