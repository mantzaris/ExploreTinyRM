import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GameDataset(Dataset):
    """Base class for game datasets. Subclass and implement _generate_sample."""
    def __init__(self, n_samples: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.samples = [self._generate_sample() for _ in range(n_samples)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def _generate_sample(self): raise NotImplementedError()

class CommunityDetectionDataset(GameDataset):
    """Synthetic SBM community detection puzzles."""
    def __init__(self, n_samples: int, n_nodes: int = 30, n_communities: int = 3, p_in: float = 0.6, p_out: float = 0.05, seed: int = 0):
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.p_in = p_in
        self.p_out = p_out
        self.rng = np.random.default_rng(seed)
        self.samples = [self._generate_sample() for _ in range(n_samples)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def _generate_sample(self):
        sizes = [self.n_nodes // self.n_communities] * self.n_communities
        for i in range(self.n_nodes % self.n_communities):
            sizes[i] += 1
        probs = np.full((self.n_communities, self.n_communities), self.p_out)
        np.fill_diagonal(probs, self.p_in)
        G = nx.stochastic_block_model(sizes, probs, seed=int(self.rng.integers(1e9)))
        labels = []
        for idx, size in enumerate(sizes):
            labels.extend([idx] * size)
        labels = np.array(labels)
        adj = nx.to_numpy_array(G)
        x_tokens = adj.flatten()
        x_tokens = torch.from_numpy(x_tokens.astype(np.int64))  # [n_nodes, n_nodes]
        y_tokens = torch.from_numpy(labels.astype(np.int64))  # [n_nodes]
        return x_tokens, y_tokens

def get_gc_loaders(n_train=512, n_val=128, batch_size=16, n_nodes=30, n_communities=3, p_in=0.6, p_out=0.05, seed=42):
    ds_tr = CommunityDetectionDataset(n_samples=n_train, n_nodes=n_nodes, n_communities=n_communities, p_in=p_in, p_out=p_out, seed=seed)
    ds_va = CommunityDetectionDataset(n_samples=n_val, n_nodes=n_nodes, n_communities=n_communities, p_in=p_in, p_out=p_out, seed=seed+1)
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=True)
    )

train_loader, val_loader = get_gc_loaders(
    n_train=2048,
    n_val=512,
    batch_size=16,
    n_nodes=5, #N_NODES
    n_communities=3,
    p_in=0.6,
    p_out=0.05,
    seed=123
)

N_NODES = 5
N_COMMUNITIES = 3
INPUT_TOKENS = 2  # adjacency values: 0 or 1 (float)
OUTPUT_TOKENS = N_COMMUNITIES
SEQ_LEN = N_NODES * N_NODES

D_MODEL = 128
N_SUP = 16
N = 6
T = 3
USE_ATT = False

