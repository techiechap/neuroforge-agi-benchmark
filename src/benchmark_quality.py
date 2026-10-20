"""Dependency-free evaluation helpers for Neuroforge's grid benchmark."""
import ast
import random


def make_case(seed: int):
    rng = random.Random(seed)
    size = rng.randint(3, 5)
    grid = [[0] * size for _ in range(size)]
    row, col = rng.randrange(size), rng.randrange(size - 1)
    grid[row][col] = 1
    available = [(r, c) for r in range(size) for c in range(size)
                 if (r, c) not in {(row, col), (row, col + 1)}]
    drow, dcol = rng.choice(available)
    grid[drow][dcol] = 2
    expected = [r[:] for r in grid]
    expected[row][col] = 0
    expected[row][col + 1] = 1
    return grid, expected


def _is_grid(value):
    return (isinstance(value, list) and 3 <= len(value) <= 5
            and all(isinstance(row, list) and len(row) == len(value)
                    and all(type(cell) is int and cell in (0, 1, 2) for cell in row)
                    for row in value))


def parse_grid(text: str):
    """Extract first well-formed 3–5 square grid, without evaluating Python code."""
    if not isinstance(text, str):
        return None
    # Try every opening bracket with a bounded bracket-count scan so prose/fences work.
    for start, char in enumerate(text[:12000]):
        if char != '[':
            continue
        depth = 0
        for end in range(start, min(len(text), start + 2400)):
            if text[end] == '[':
                depth += 1
            elif text[end] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        value = ast.literal_eval(text[start:end + 1])
                    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                        break
                    if _is_grid(value):
                        return value
                    break
    return None


def extract_pass(run):
    """Fail closed for unknown Kaggle result structures; never bool(run.result)."""
    if isinstance(run, bool):
        return run
    for item in getattr(run, 'results', []) or []:
        result = item if isinstance(item, dict) else vars(item) if hasattr(item, '__dict__') else {}
        if result.get('type') == 'AGGREGATED' and type(result.get('booleanResult')) is bool:
            return result['booleanResult']
    value = getattr(run, 'result', None)
    if type(value) is bool:
        return value
    # Unrecognized result is not a demonstrated pass.
    raise ValueError(f'Unsupported Kaggle run result: {type(run).__name__}')
