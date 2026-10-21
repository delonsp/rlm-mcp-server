"""Local, synthetic retrieval benchmark. No network or persisted corpus access.

Run: PYTHONPATH=src .venv/bin/python scripts/benchmark_retrieval.py
"""
import json
import random
import statistics
import time

from rlm_mcp.indexer import create_index, fingerprint_source


def median_ms(function, repeats=31):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(timings), 4)


def main():
    rng = random.Random(20261020)
    text = '\n\n'.join('commonterm ' * rng.randint(1, 12) + f'document{i} '
                        + 'contextword ' * rng.randint(1, 30) for i in range(20000))
    index = create_index(text, 'benchmark')
    start = time.perf_counter()
    index.search_bm25(['commonterm'], text, limit=10)
    cold_ms = (time.perf_counter() - start) * 1000
    fingerprint_source(text)
    print(json.dumps({
        'chars': len(text), 'segments': index.bm25_n, 'repeats': 31,
        'cold_bm25_ms': round(cold_ms, 3),
        'warm_top10_median_ms': median_ms(lambda: index.search_bm25(['commonterm'], text, limit=10)),
        'version_check_median_ms': median_ms(lambda: fingerprint_source(text)),
        'top10_lines': [h['line'] for h in index.search_bm25(['commonterm'], text, limit=10)],
    }, indent=2))


if __name__ == '__main__':
    main()
