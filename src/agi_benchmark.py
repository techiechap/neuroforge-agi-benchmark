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
    # 1. Generate dynamic grid
    size = random.randint(3, 5)
    grid = [[0 for _ in range(size)] for _ in range(size)]

    # Place target '1' so it can always move right
    start_row = random.randint(0, size - 1)
    start_col = random.randint(0, size - 2)
    grid[start_row][start_col] = 1

    # Place distractor '2' safely
    while True:
        d_row, d_col = random.randint(0, size-1), random.randint(0, size-1)
        if (d_row, d_col) not in [(start_row, start_col), (start_row, start_col + 1)]:
            grid[d_row][d_col] = 2
            break

    # 2. Calculate Truth
    expected_grid = solver.move_right(grid, target_value=1)
    expected_str = str(expected_grid)

    # 3. Prompt
    prompt = (
        f"Input Grid: {grid}\n"
        "Task: Find the value 1 and move it one column to the right. "
        "Leave all other values exactly where they are. Output only the grid."
    )

    # 4. Execute & Parse
    response = llm.prompt(prompt)
    got_grid = parse_grid_robust(response)
    got_str = str(got_grid) if got_grid else "None"

    # 5. Diagnostic Logging
    is_correct = (expected_str == got_str)
    if not is_correct:
        reason = classify_failure(expected_grid, got_grid)
        print(f"❌ FAILURE [{reason}]")
        print(f"   Input:    {grid}")
        print(f"   Expected: {expected_str}")
        print(f"   Got:      {got_str}")
    else:
        print(f"✅ SUCCESS")

    return is_correct