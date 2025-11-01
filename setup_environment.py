import torch, random, numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def check_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"