from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'vendor')]
import torch
torch.set_num_threads(1)
