import numpy as np

class ARGAugmentor:
    """Apply Dihedral Group D4 symmetries to ARC grids."""
    @staticmethod
    def get_all_symmetries(grid):
        grid = np.array(grid)
        symmetries = []
        for k in range(4):
            # Rotations
            rot = np.rot90(grid, k)
            symmetries.append(rot.tolist())
            # Flips
            symmetries.append(np.fliplr(rot).tolist())
        return symmetries

    @staticmethod
    def color_permute(grid, permutation_map):
        """Randomly swap colors (0-9) to see if the rule is color-invariant."""
        return [[permutation_map.get(c, c) for c in row] for row in grid]