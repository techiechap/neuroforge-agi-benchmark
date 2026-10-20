import kaggle_benchmarks as kbench
import random
import re
import ast
import solver # Your logic engine

def parse_grid_robust(text: str):
    """
    Extracts a grid [[...]] from LLM response even if it contains 
    conversational 'filler' or markdown.
    """
    # Look for the grid pattern specifically
    match = re.search(r"(\[[\s]*\[.*\][\s]*\])", text, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except:
            return None
    return None

def classify_failure(expected, got):
    """
    Categorizes the specific type of spatial reasoning failure.
    Useful for the 'Technical Insights' section of the Kaggle Writeup.
    """
    if got is None: return "Parsing Error / No Grid Found"
    if len(expected) != len(got): return "Dimensional Collapse (Size Change)"
    
    # Check if the distractor (2) moved
    distractor_moved = False
    for r in range(len(expected)):
        for c in range(len(expected[0])):
            if expected[r][c] == 2 and (r >= len(got) or c >= len(got[0]) or got[r][c] != 2):
                distractor_moved = True
                break
    
    if distractor_moved: return "Inhibitory Control Failure (Distractor Interference)"
    return "Target Displacement Error (Wrong Move)"

@kbench.task(
    name="spatial_inhibition_v1",
    description="Tests if the model can move objects while ignoring distractor values.",
    version=1
)
def spatial_inhibition_task(llm):
    import os
    import json
    from src.benchmark_quality import make_case, parse_grid

    seed = int(os.environ.get("NEUROFORGE_CASE_SEED", "0"))
    grid, expected = make_case(seed)
    prompt = (f"Input Grid: {grid}\n"
              "Task: Find the value 1 and move it one column to the right. "
              "Leave all other values exactly where they are. Output only the grid.")
    response = llm.prompt(prompt)
    got = parse_grid(response)
    passed = got == expected
    # Tiny non-sensitive reproducibility trace. No raw prompts, tokens, or generated code.
    print("CASE_JSON=" + json.dumps({"seed": seed, "input": grid,
                                    "expected": expected, "got": got,
                                    "passed": passed}, separators=(",", ":")))
    return passed