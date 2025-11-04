import os, sys
sys.path.append(os.path.join("..", "src"))
from exploretinyrm.trm import TRM, TRMConfig
from base import BaseTrainer, EMA, make_grad_scaler, ExperimentLogger
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

class GameDataset(Dataset):
    """Base class for game datasets. Subclass and implement _generate_sample."""
    def __init__(self, n_samples: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.samples = [self._generate_sample() for _ in range(n_samples)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def _generate_sample(self): raise NotImplementedError()

class GraphColoringDataset(GameDataset):
    """Synthetic undirected graph coloring puzzles."""
    def __init__(self, n_samples: int, n_nodes: int = 6, n_colors: int = 3, edge_prob: float = 0.4, seed: int = 0):
        self.n_nodes = n_nodes
        self.n_colors = n_colors
        self.edge_prob = edge_prob
        super().__init__(n_samples, seed)
    def _generate_sample(self):
        while True:
            adj = np.triu((self.rng.random((self.n_nodes, self.n_nodes)) < self.edge_prob).astype(np.int64), 1)
            adj = adj + adj.T
            colors = np.full(self.n_nodes, -1, dtype=np.int64)
            for node in range(self.n_nodes):
                forbidden = set(colors[adj[node] == 1])
                for c in range(self.n_colors):
                    if c not in forbidden:
                        colors[node] = c
                        break
            if np.all((colors >= 0) & (colors < self.n_colors)):
                x_tokens = adj.flatten()
                y_tokens = colors
                return torch.from_numpy(x_tokens), torch.from_numpy(y_tokens)

def get_gc_loaders(n_train=512, n_val=128, batch_size=16, n_nodes=6, n_colors=3, edge_prob=0.4, seed=123):
    ds_tr = GraphColoringDataset(n_train, n_nodes=n_nodes, n_colors=n_colors, edge_prob=edge_prob, seed=seed)
    ds_va = GraphColoringDataset(n_val, n_nodes=n_nodes, n_colors=n_colors, edge_prob=edge_prob, seed=seed+1)
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=True)
    )

class GraphColoringTrainer(BaseTrainer):
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
        super().__init__("GraphColoring", model, optimizer, scaler, ema, device, train_loader, val_loader, config)

if __name__ == "__main__":
    grid = [
        # (n_nodes, n_colors, edge_prob, d_model, n_layers)
        (4, 3, 0.2, 64, 2),
        (6, 3, 0.3, 128, 2),
        (8, 4, 0.4, 256, 4),
        (10, 4, 0.5, 512, 6),
    ]
    for n_nodes, n_colors, edge_prob, d_model, n_layers in grid:
        config = {
            'N_NODES': n_nodes,
            'N_COLORS': n_colors,
            'INPUT_TOKENS': 2,
            'OUTPUT_TOKENS': n_colors,
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
            n_colors=n_colors,
            edge_prob=edge_prob,
            seed=123
        )
        trainer = GraphColoringTrainer(config, train_loader, val_loader)
        acc_raw_baseline = trainer.evaluate()
        print(f"Grid: n_nodes={n_nodes}, n_colors={n_colors}, edge_prob={edge_prob}, d_model={d_model}, n_layers={n_layers}")
        print("Baseline accuracy (random guessing): {:.3f}".format(acc_raw_baseline))
        trainer.logger.log_metrics(0, None, acc_raw_baseline)
        EPOCHS = 5
        for epoch in range(1, EPOCHS+1):
            trainer.train_one_epoch(epoch)
            acc_raw = trainer.evaluate()
            print(f"Validation (raw) | Node accuracy {acc_raw:.3f}")
            trainer.logger.log_metrics(epoch, None, acc_raw)
