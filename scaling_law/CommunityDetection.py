import os, sys
sys.path.append(os.path.join("..", "src"))
from exploretinyrm.trm import TRM, TRMConfig
from base import BaseTrainer, EMA, make_grad_scaler, ExperimentLogger
import torch
import networkx as nx
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import csv
import json
from datetime import datetime

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

class CommunityDetectionTrainer(BaseTrainer):
    def __init__(self, config, train_loader, val_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = TRMConfig(
            input_vocab_size=config['INPUT_TOKENS'],
            output_vocab_size=config['OUTPUT_TOKENS'],
            seq_len=config['SEQ_LEN'],
            d_model=config['D_MODEL'],
            n_layers=config.get('N_LAYERS', 2),
            use_attention=config['USE_ATT'],
            n_heads=8,
            dropout=0.0,
            mlp_ratio=4.0,
            token_mlp_ratio=2.0,
            n=config['N'],
            T=config['T'],
            k_last_ops=None,
            stabilize_input_sums=True
        )
        model = TRM(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0, betas=(0.9, 0.95))
        scaler = make_grad_scaler(device.type == "cuda")
        ema = EMA(model, decay=0.999)
        super().__init__("CommunityDetection", model, optimizer, scaler, ema, device, train_loader, val_loader, config)

if __name__ == "__main__":
    grid = [
        # (n_nodes, n_communities, d_model, n_layers)
        (5, 2, 64, 2),
        (10, 2, 128, 2),
        (20, 3, 256, 4),
        (30, 4, 512, 6),
    ]
    for n_nodes, n_communities, d_model, n_layers in grid:
        config = {
            'N_NODES': n_nodes,
            'N_COMMUNITIES': n_communities,
            'INPUT_TOKENS': 2,
            'OUTPUT_TOKENS': n_communities,
            'SEQ_LEN': n_nodes * n_nodes,
            'D_MODEL': d_model,
            'N_SUP': 16,
            'N': 6,
            'T': 3,
            'USE_ATT': False,
            'N_LAYERS': n_layers
        }
        train_loader, val_loader = get_gc_loaders(
            n_train=2048,
            n_val=512,
            batch_size=16,
            n_nodes=n_nodes,
            n_communities=n_communities,
            p_in=0.6,
            p_out=0.05,
            seed=123
        )
        trainer = CommunityDetectionTrainer(config, train_loader, val_loader)
        acc_raw_baseline = trainer.evaluate()
        print(f"Grid: n_nodes={n_nodes}, n_communities={n_communities}, d_model={d_model}, n_layers={n_layers}")
        print("Baseline accuracy (random guessing): {:.3f}".format(acc_raw_baseline))
        trainer.logger.log_metrics(0, None, acc_raw_baseline)
        EPOCHS = 5
        for epoch in range(1, EPOCHS+1):
            acc_train = trainer.train_one_epoch(epoch)
            acc_raw = trainer.evaluate()
            print(f"Validation (raw) | Node accuracy {acc_raw:.3f}")
            trainer.logger.log_metrics(epoch, acc_train, acc_raw)

