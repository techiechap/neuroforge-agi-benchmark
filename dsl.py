import json
import torch
from torch.utils.data import Dataset
import numpy as np

class ARCDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.tasks = json.load(f)
        self.task_ids = list(self.tasks.keys())

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, idx):
        tid = self.task_ids[idx]
        task = self.tasks[tid]
        
        # 'train' contains the demonstration pairs
        train_pairs = task['train']
        # 'test' contains the challenge input
        test_inputs = task['test']
        
        return {
            "id": tid,
            "train": train_pairs,
            "test": test_inputs
        }

def grid_to_tensor(grid):
    # Convert list of lists to torch tensor [0-9]
    return torch.tensor(grid, dtype=torch.long)