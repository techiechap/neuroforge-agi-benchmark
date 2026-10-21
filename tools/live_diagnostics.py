"""Bounded, local-only Neuroforge Ollama inference diagnostics. No Kaggle dependency."""
import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from src.benchmark_quality import make_case, parse_grid


def classify(raw, expected, input_grid):
    parsed = parse_grid(raw)
    if parsed is None:
        return 'FORMAT_FAILURE', None
    if parsed == expected:
        return 'CORRECT', parsed
    if parsed == input_grid:
        return 'INPUT_ECHO', parsed
    return 'WRONG_GRID', parsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['gemma3:1b', 'llama3.2:1b'])
    p.add_argument('--cases', type=int, default=5)
    p.add_argument('--output', type=Path, default=Path.home()/'Downloads/tmp/neuroforge-live-diagnostics.jsonl')
    args = p.parse_args()
    if not 1 <= args.cases <= 20:
        p.error('--cases must be 1..20 (bounded local test)')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tally = {}
    with args.output.open('w', encoding='utf-8') as out:
        for model in args.models:
            tally[model] = {}
            for seed in range(1, args.cases + 1):
                grid, expected = make_case(seed)
                prompt = (f'Input Grid: {grid}\nTask: Find the value 1 and move it one column to the right. '
                          'Leave all other values exactly where they are. Output only the grid.')
                rec = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'model': model,
                       'seed': seed, 'input': grid, 'expected': expected, 'prompt': prompt}
                t0 = time.monotonic()
                try:
                    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                                       'stream': False, 'options': {'temperature': 0.1}}).encode()
                    request = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=body,
                                                     headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=120) as resp:
                        data = json.load(resp)
                    raw = data['message']['content']
                    rec['raw_response'] = raw
                    rec['classification'], rec['parsed_grid'] = classify(raw, expected, grid)
                except Exception as exc:
                    rec['classification'] = 'API_ERROR'
                    rec['error_type'] = type(exc).__name__
                    rec['error_message'] = str(exc)[:500]
                rec['elapsed_seconds'] = round(time.monotonic() - t0, 3)
                tally[model][rec['classification']] = tally[model].get(rec['classification'], 0) + 1
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                print(f"{model} seed={seed:02d} {rec['classification']} {rec['elapsed_seconds']}s", flush=True)
    print('SUMMARY', json.dumps(tally, indent=2))
    print('SAVED', args.output)

if __name__ == '__main__':
    main()
