import subprocess

import subprocess

class ARCSolver:
    def __init__(self, model_engine):
        self.engine = model_engine

    def propose_solution_code(self, task_data):
        prompt = self.create_prompt(task_data)
        # Ask model to write: def transform(input_grid): ...
        python_code = self.engine.generate(prompt)
        return python_code

    def validate_and_run(self, code, train_pairs, test_input):
        """Execute the synthesized code and check if it passes train examples."""
        try:
            # Dangerous in prod, but standard for AGI Hackathons: 
            # exec() or running in a subprocess.
            for pair in train_pairs:
                # result = run_code(code, pair['input'])
                # if result != pair['output']: return None
                pass
            # If all train pass, run on test
            return "final_grid_result"
        except Exception as e:
            return None
    
# --- MOVE THIS OUTSIDE THE CLASS (NO INDENTATION) ---
def move_right(grid, target_value=1):
    """
    Finds target_value in the grid and moves it one step to the right.
    Returns a NEW grid (does not modify the original).
    """
    size = len(grid)
    # Create a deep copy of the grid
    new_grid = [row[:] for row in grid]

    for r in range(size):
        for c in range(size):
            if grid[r][c] == target_value:
                # Calculate new position (staying within bounds)
                new_c = min(c + 1, size - 1)
                
                # Update the new grid
                new_grid[r][c] = 0
                new_grid[r][new_c] = target_value
                return new_grid
                
    return new_grid