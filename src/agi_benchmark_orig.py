import kaggle_benchmarks as kbench
import numpy as np

@kbench.task(
    name="spatial_inhibition_v1",
    description="Tests if the model can move objects while ignoring distractor colors.",
    version=1
)
def spatial_inhibition_task(llm):
    # 1. Define the Puzzle Logic
    grid = [
        [0, 1, 0],
        [0, 0, 2],
        [0, 0, 0]
    ]
    # Rule: Move '1' (Blue) one step right. Ignore '2' (Red).
    prompt = f"Grid: {grid}\nInstruction: Move the value 1 one step to the right. Do not change any other values."
    expected_grid = [
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 0]
    ]

    # 2. Get the Model's "Thinking"
    response = llm.prompt(prompt)
    
    # 3. Validation Logic
    # The SDK helps parse the response and compare it to our expected output
    if str(expected_grid) in response:
        return kbench.Score(value=1.0, message="Perfect spatial reasoning.")
    else:
        return kbench.Score(value=0.0, message="Failed to inhibit distractor or move object.")

if __name__ == "__main__":
    # This lets you run the task locally to test your L4
    print("Running local benchmark test...")
    # kbench.run(spatial_inhibition_task)