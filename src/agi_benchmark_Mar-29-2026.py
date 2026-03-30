import sys
from pathlib import Path
# Adds the root to path so src/ can see dsl.py
sys.path.append(str(Path(__file__).parent.parent)) 
import kaggle_benchmarks as kbench 
import dsl 
import augmentation

@kbench.task(
    name="spatial_inhibition_v1",
    description="Tests if the model can move objects while ignoring distractor colors.",
    version=1
)
def spatial_inhibition_task(llm):
    grid = [[0, 1, 0], [0, 0, 2], [0, 0, 0]]
    prompt = f"Grid: {grid}\nInstruction: Move the value 1 one step to the right. Do not change any other values."
    expected_grid = "[[0, 0, 1], [0, 0, 2], [0, 0, 0]]"

    response = llm.prompt(prompt)
    
    # Logic: If the stringified expected grid is in the response, score 1.0
    if expected_grid in response:
        return 1.0  # The SDK will wrap this float into a Score automatically
    else:
        return 0.0