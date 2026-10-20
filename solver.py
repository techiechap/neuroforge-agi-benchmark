import subprocess
import copy

class ARCSolver:
    """
    Main solver class for higher-level AGI tasks (ARC-style).
    Used when the model needs to synthesize Python code to solve a task.
    """
    def __init__(self, model_engine):
        self.engine = model_engine # e.g., your OllamaLLM instance

    def propose_solution_code(self, task_data):
        """Asks the model to write a 'transform(grid)' function."""
        prompt = f"Write a Python function 'transform(input_grid)' for this task: {task_data}"
        python_code = self.engine.prompt(prompt)
        return python_code

    def validate_and_run(self, code, train_pairs, test_input):
        """
        Executes synthesized code against training examples to verify logic 
        before applying it to the final test input.
        """
        raise RuntimeError("Untrusted model-generated Python execution disabled pending OS-level sandbox")
        try:
            # Note: In a production benchmark, use a sandboxed subprocess.
            # For local validation, we assume the code is a string containing 'def transform'
            local_vars = {}
            exec(code, {}, local_vars)
            transform = local_vars.get('transform')
            
            if not transform:
                return None
                
            # Verify against training pairs
            for pair in train_pairs:
                if transform(pair['input']) != pair['output']:
                    return None
            
            # Run on actual test input
            return transform(test_input)
        except Exception as e:
            print(f"Solver execution error: {e}")
            return None

# --- TOP LEVEL UTILITIES (The 'Ground Truth' for agi_benchmark.py) ---

def move_right(grid, target_value=1):
    """
    Core Logic for Spatial Inhibition Task:
    Finds the 'target_value' and shifts it one cell to the right.
    Crucial: It ignores all other values (like distractors/2s).
    """
    size = len(grid)
    # Deep copy using list comprehension for speed/stability
    new_grid = [row[:] for row in grid]

    for r in range(size):
        for c in range(size):
            if grid[r][c] == target_value:
                # Calculate new position
                # If at edge, it stays at the edge (or you could wrap-around)
                new_c = min(c + 1, size - 1)
                
                # Update: Clear old position, set new position
                new_grid[r][c] = 0
                new_grid[r][new_c] = target_value
                
                # Return immediately after the single move is complete
                return new_grid
                
    return new_grid