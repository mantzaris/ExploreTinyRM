
# tsp_data.py
# Minimal, reproducible TSP data generator for TRM-style training.
# Produces (x_tokens, y_tokens) pairs where:
#   - x_tokens[j] encodes the 2D integer coordinates of city j as a single token in [0, grid_size*grid_size - 1].
#   - y_tokens[j] is the index of the "next city" after city j on the canonical optimal tour (0-based).
#
# Canonicalization: tours are solved with start city fixed to 0. If multiple optimal tours exist
# (e.g., reverse orientation), tie-breaking is deterministic to keep labels single-valued.
#
# The shapes match TRM's expectation: input and output are both length L = n_cities.
#
# Usage in notebook:
#   from tsp_data import TSPConfig, TSPDataset, get_tsp_loaders, decode_coords, distance_matrix, tour_length_from_successors
#
#   cfg = TSPConfig(n_cities=10, grid_size=16, solver="auto", seed=123)
#   train_loader, val_loader = get_tsp_loaders(2048, 512, batch_size=32, cfg=cfg)
#   INPUT_TOKENS = cfg.grid_size * cfg.grid_size
#   OUTPUT_TOKENS = cfg.n_cities
#   SEQ_LEN = cfg.n_cities
#
#   bx, by = next(iter(train_loader))
#   assert bx.min().item() >= 0 and bx.max().item() < INPUT_TOKENS
#   assert by.min().item() >= 0 and by.max().item() < OUTPUT_TOKENS
#
#   # Inspect one validation example
#   x_tokens, y_tokens = next(iter(val_loader))
#   coords = decode_coords(x_tokens[0].numpy(), cfg.grid_size)
#   dist = distance_matrix(coords)
#   print("Tour length:", tour_length_from_successors(dist, y_tokens[0].numpy()))
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from typing import List  # (you already have this)
import matplotlib.pyplot as plt  # add this if not present

# Make exports explicit so collaborators know what is public
__all__ = [
    "TSPConfig", "TSPDataset", "get_tsp_loaders",
    "decode_coords", "distance_matrix", "tour_length_from_successors",
    "path_from_successors", "plot_tsp_instance_from_tokens", "plot_tsp_compare_from_tokens"
]

# ---------- Visualization helpers (TOP-LEVEL; not under __main__) ----------
from typing import Sequence, Optional, Union, Tuple
import numpy as np
import torch

ArrayLike1D = Union[np.ndarray, torch.Tensor, Sequence[int]]





# -------------------------------------
# Configuration
# -------------------------------------

@dataclass
class TSPConfig:
    n_cities: int = 10
    grid_size: int = 16
    # solver: "auto" uses exact Held-Karp up to max_exact_cities, else 2-opt
    # "held_karp" forces exact, "two_opt" forces approximate
    solver: str = "auto"
    max_exact_cities: int = 12
    seed: int = 0


# -------------------------------------
# Tokenization helpers
# -------------------------------------

def coord_to_token(x: int, y: int, grid_size: int) -> int:
    """Encode (x, y) on a grid_size x grid_size integer grid into a single token id."""
    return int(y) * grid_size + int(x)


def token_to_coord(token: int, grid_size: int) -> Tuple[int, int]:
    """Decode token id back to (x, y)."""
    x = int(token) % grid_size
    y = int(token) // grid_size
    return x, y


def decode_coords(x_tokens: np.ndarray, grid_size: int) -> np.ndarray:
    """Decode a length-L vector of tokens into integer coordinates array of shape [L, 2]."""
    coords = np.zeros((len(x_tokens), 2), dtype=np.int64)
    for i, t in enumerate(x_tokens):
        coords[i] = token_to_coord(int(t), grid_size)
    return coords


# -------------------------------------
# Geometry and distance
# -------------------------------------

def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for coords shape [N, 2]. Returns [N, N] float64."""
    coords = coords.astype(np.float64)
    diff_x = coords[:, None, 0] - coords[None, :, 0]
    diff_y = coords[:, None, 1] - coords[None, :, 1]
    dist = np.sqrt(diff_x * diff_x + diff_y * diff_y)
    # Zero on diagonal, symmetric
    np.fill_diagonal(dist, 0.0)
    return dist


# -------------------------------------
# Exact Held-Karp solver (O(n^2 2^n)). Works well up to about 12 cities in Python.
# -------------------------------------

def held_karp(dist: np.ndarray, start: int = 0) -> List[int]:
    """
    Solve TSP exactly with Held-Karp DP.
    Returns a canonical tour path [start, v1, v2, ..., vn-1, start].
    Ties are broken deterministically by preferring smaller city indices.
    """
    n = dist.shape[0]
    assert start == 0, "This implementation assumes start city 0 for canonicalization."
    if n <= 2:
        return [0, 1, 0] if n == 2 else [0, 0]

    size = 1 << n
    inf = np.inf
    # dp[mask, j] = min cost to start at 0, visit mask, and end at j (mask includes j and 0)
    dp = np.full((size, n), inf, dtype=np.float64)
    parent = np.full((size, n), -1, dtype=np.int16)

    # base: only city 0 in mask
    dp[1 << 0, 0] = 0.0

    # iterate over masks that include 0
    for mask in range(size):
        if (mask & 1) == 0:
            continue  # must include start
        # for each j in mask
        for j in range(n):
            if (mask & (1 << j)) == 0:
                continue
            if j == 0 and mask != (1 << 0):
                # we never end at 0 unless mask is only {0}
                continue
            prev_mask = mask ^ (1 << j)
            if prev_mask == 0 and j == 0:
                continue
            if prev_mask == 0:
                # cannot reach j without visiting at least 0 and j
                continue
            # transition i -> j
            best_cost = dp[mask, j]
            best_i = parent[mask, j]
            for i in range(n):
                if (prev_mask & (1 << i)) == 0:
                    continue
                cand = dp[prev_mask, i] + dist[i, j]
                if cand < best_cost - 1e-12:
                    best_cost = cand
                    best_i = i
                elif np.isfinite(cand) and np.isfinite(best_cost) and abs(cand - best_cost) <= 1e-12 and (best_i == -1 or i < best_i):
                    # tie-break deterministically on city index
                    best_cost = cand
                    best_i = i
            dp[mask, j] = best_cost
            parent[mask, j] = best_i

    # close the cycle back to 0
    full_mask = size - 1
    best_final = inf
    last = -1
    for j in range(1, n):
        cand = dp[full_mask, j] + dist[j, 0]
        if cand < best_final - 1e-12:
            best_final = cand
            last = j
        elif np.isfinite(cand) and np.isfinite(best_final) and abs(cand - best_final) <= 1e-12 and (last == -1 or j < last):
            best_final = cand
            last = j


    # reconstruct path
    path = [0] * (n + 1)
    path[-1] = 0
    mask = full_mask
    j = last
    for k in range(n - 1, 0, -1):
        path[k] = j
        pmask = mask ^ (1 << j)
        i = int(parent[mask, j])
        mask = pmask
        j = i
    path[0] = 0

    # make sure the representation is canonical with respect to reverse duplicates:
    # Compare the second city in forward vs backward direction and pick the lexicographically smaller.
    if n >= 3:
        f_second = path[1]
        b_second = path[-2]
        if b_second < f_second:
            path = list(reversed(path))

    return path


# -------------------------------------
# Approximate 2-opt solver with nearest-neighbor initialization
# -------------------------------------

def nearest_neighbor_path(dist: np.ndarray, start: int = 0) -> List[int]:
    n = dist.shape[0]
    visited = [False] * n
    order = [start]
    visited[start] = True
    for _ in range(n - 1):
        last = order[-1]
        best = None
        best_d = np.inf
        for j in range(n):
            if not visited[j] and dist[last, j] < best_d - 1e-12:
                best_d = dist[last, j]
                best = j
        order.append(best)
        visited[best] = True
    order.append(start)
    return order


def two_opt_tour(dist: np.ndarray, start: int = 0, max_iter: int = 10000) -> List[int]:
    """2-opt local search on a cycle. Returns [start, ..., start]."""
    n = dist.shape[0]
    path = nearest_neighbor_path(dist, start=start)

    def segment_length(i, j):
        return dist[path[i], path[i + 1]] + dist[path[j], path[j + 1]]

    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(0, n - 1):
            for j in range(i + 1, n - 1):
                delta = (dist[path[i], path[j]] + dist[path[i + 1], path[j + 1]]) - segment_length(i, j)
                if delta < -1e-12:
                    # reverse segment (i+1 to j)
                    path[i + 1 : j + 1] = reversed(path[i + 1 : j + 1])
                    improved = True
        # loop again if improved
    # enforce canonical start and orientation tie-break
    if path[0] != start:
        # rotate so that path starts at 'start'
        idx = path.index(start)
        path = path[idx:] + path[1:idx + 1]

    if n >= 3:
        f_second = path[1]
        b_second = path[-2]
        if b_second < f_second:
            path = list(reversed(path))

    return path


# -------------------------------------
# Conversions between path and successor labels
# -------------------------------------

def successors_from_path(path: List[int], n: int) -> np.ndarray:
    """From path [0, v1, ..., vn-1, 0], build successor array s where s[i] is the next city after i."""
    succ = np.empty(n, dtype=np.int64)
    for a, b in zip(path[:-1], path[1:]):
        succ[a] = b
    return succ


def tour_length_from_successors(dist: np.ndarray, succ: np.ndarray) -> float:
    """Compute length of cycle defined by successor mapping succ."""
    n = len(succ)
    total = 0.0
    cur = 0
    for _ in range(n):
        nxt = int(succ[cur])
        total += dist[cur, nxt]
        cur = nxt
    total += dist[cur, 0] if cur != 0 else 0.0
    return float(total)


# -------------------------------------
# Sampling utilities
# -------------------------------------

def sample_coords_unique(n_cities: int, grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n_cities unique integer coordinates on a grid without replacement."""
    total = grid_size * grid_size
    if n_cities > total:
        raise ValueError("n_cities must be <= grid_size * grid_size")
    tokens = rng.choice(total, size=n_cities, replace=False)
    coords = decode_coords(tokens, grid_size)
    return coords


# -------------------------------------
# Dataset
# -------------------------------------

class TSPDataset(Dataset):
    """
    Returns tuples (x_tokens, y_tokens) of length L = n_cities.
    x_tokens[j] encodes the coordinate of city j.
    y_tokens[j] is the index of the successor city in the tour.
    """
    def __init__(self, n_samples: int, cfg: TSPConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_samples):
            coords = sample_coords_unique(cfg.n_cities, cfg.grid_size, self.rng)
            dist = distance_matrix(coords)

            if cfg.solver == "held_karp" or (cfg.solver == "auto" and cfg.n_cities <= cfg.max_exact_cities):
                path = held_karp(dist, start=0)
            elif cfg.solver == "two_opt" or (cfg.solver == "auto" and cfg.n_cities > cfg.max_exact_cities):
                path = two_opt_tour(dist, start=0)
            else:
                raise ValueError(f"Unknown solver mode: {cfg.solver}")

            succ = successors_from_path(path, cfg.n_cities)
            x_tokens = np.array([coord_to_token(coords[i, 0], coords[i, 1], cfg.grid_size) for i in range(cfg.n_cities)], dtype=np.int64)
            y_tokens = succ.astype(np.int64)

            # sanity check: successor mapping forms a single cycle of length n_cities
            if not _is_single_cycle(y_tokens):
                raise RuntimeError("Generated successor mapping is not a single cycle. This should not happen.")

            self.samples.append((x_tokens, y_tokens))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def _is_single_cycle(succ: np.ndarray) -> bool:
    n = len(succ)
    seen = np.zeros(n, dtype=bool)
    cur = 0
    for _ in range(n):
        if seen[cur]:
            return False
        seen[cur] = True
        cur = int(succ[cur])
    return cur == 0 and seen.all()


# -------------------------------------
# Dataloaders and simple validations
# -------------------------------------

def get_tsp_loaders(n_train: int, n_val: int, batch_size: int, cfg: TSPConfig, drop_last: bool = True):
    ds_tr = TSPDataset(n_train, cfg)
    ds_va = TSPDataset(n_val, TSPConfig(**{**cfg.__dict__, "seed": cfg.seed + 1}))
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=drop_last, pin_memory=True),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=True),
    )


# -------------------------------------
# Module self-test
# -------------------------------------

if __name__ == "__main__":
    cfg = TSPConfig(n_cities=8, grid_size=12, solver="auto", seed=123)
    ds = TSPDataset(4, cfg)
    x, y = ds[0]
    coords = decode_coords(x.numpy(), cfg.grid_size)
    dist = distance_matrix(coords)
    length = tour_length_from_successors(dist, y.numpy())
    print("Example tour length:", length)
    print("x tokens:", x)
    print("y tokens (successors):", y)









# visualizations
# --- TSP visualization helpers ---
# Paste this cell into your notebook (after importing from tsp_data).

from typing import List, Tuple, Optional, Sequence, Union
import numpy as np
import torch
import matplotlib.pyplot as plt


ArrayLike1D = Union[np.ndarray, torch.Tensor, Sequence[int]]

def path_from_successors(successor: ArrayLike1D, start: int = 0) -> List[int]:
    """
    Convert a successor mapping s where s[i] is the next city after i
    into an explicit tour path [start, v1, ..., v_{n-1}, start].
    Assumes successor is a single Hamiltonian cycle.
    """
    if isinstance(successor, torch.Tensor):
        successor = successor.detach().cpu().numpy()
    successor = np.asarray(successor, dtype=np.int64)

    n = successor.shape[0]
    path = [start]
    current = start
    visited = set([start])
    for _ in range(n - 1):
        current = int(successor[current])
        if current in visited:
            # Defensive programming: if the mapping closes early, we stop.
            break
        visited.add(current)
        path.append(current)
    path.append(start)
    return path

def plot_tsp_instance_from_tokens(
    x_tokens: ArrayLike1D,
    y_successor: ArrayLike1D,
    grid_size: int,
    annotate: bool = True,
    title: Optional[str] = None,
    axis_padding: float = 0.5,
) -> Tuple[plt.Figure, plt.Axes, List[int], float]:
    """
    Plot a ground-truth TSP instance from tokens and its canonical tour.

    Parameters
    ----------
    x_tokens : length-L array-like of ints
        One coordinate token per city (0..grid_size*grid_size-1).
    y_successor : length-L array-like of ints
        Successor city index for each city (0..L-1).
    grid_size : int
        Size of the integer grid along one axis.
    annotate : bool
        If True, annotate each city with its index.
    title : str or None
        Title to place on the plot. If None, a default with tour length is used.
    axis_padding : float
        Margin around [0, grid_size] axes.

    Returns
    -------
    fig, ax, path, tour_len
    """
    # Convert to numpy
    if isinstance(x_tokens, torch.Tensor):
        x_tokens = x_tokens.detach().cpu().numpy()
    if isinstance(y_successor, torch.Tensor):
        y_successor = y_successor.detach().cpu().numpy()

    x_tokens = np.asarray(x_tokens, dtype=np.int64)
    y_successor = np.asarray(y_successor, dtype=np.int64)

    # Decode coordinates and compute distances and tour length
    coords = decode_coords(x_tokens, grid_size)       # shape [L, 2]
    dist = distance_matrix(coords)                    # shape [L, L]
    path = path_from_successors(y_successor, start=0) # [0, ..., 0]
    tour_len = tour_length_from_successors(dist, y_successor)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    # Draw tour edges in order
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ax.plot([coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                linewidth=1.5)

    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1], s=40, zorder=3)
    # Highlight start city 0
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=80, marker='s', zorder=4)

    if annotate:
        for i, (x, y) in enumerate(coords):
            ax.text(x + 0.1, y + 0.1, str(i), fontsize=8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-axis_padding, grid_size + axis_padding)
    ax.set_ylim(-axis_padding, grid_size + axis_padding)
    ax.set_xticks(range(0, grid_size + 1, max(1, grid_size // 4)))
    ax.set_yticks(range(0, grid_size + 1, max(1, grid_size // 4)))
    ax.grid(True, linewidth=0.5, alpha=0.3)

    if title is None:
        ax.set_title(f"TSP instance, L = {coords.shape[0]}, length = {tour_len:.3f}")
    else:
        ax.set_title(title + f"  (len = {tour_len:.3f})")

    return fig, ax, path, float(tour_len)

def plot_tsp_compare_from_tokens(
    x_tokens: ArrayLike1D,
    successor_true: ArrayLike1D,
    successor_pred: ArrayLike1D,
    grid_size: int,
    annotate: bool = False,
    title: Optional[str] = None,
    axis_padding: float = 0.5,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overlay ground-truth tour (solid) and predicted tour (dashed) on the same coordinates.
    Useful to visually confirm learned successor mappings.

    Returns
    -------
    fig, ax
    """
    # Convert to numpy
    if isinstance(x_tokens, torch.Tensor):
        x_tokens = x_tokens.detach().cpu().numpy()
    successor_true = np.asarray(
        successor_true.detach().cpu().numpy() if isinstance(successor_true, torch.Tensor) else successor_true,
        dtype=np.int64,
    )
    successor_pred = np.asarray(
        successor_pred.detach().cpu().numpy() if isinstance(successor_pred, torch.Tensor) else successor_pred,
        dtype=np.int64,
    )

    coords = decode_coords(x_tokens, grid_size)
    dist = distance_matrix(coords)
    path_true = path_from_successors(successor_true, start=0)
    path_pred = path_from_successors(successor_pred, start=0)

    len_true = tour_length_from_successors(dist, successor_true)
    len_pred = tour_length_from_successors(dist, successor_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # True tour (solid)
    for i in range(len(path_true) - 1):
        a, b = path_true[i], path_true[i + 1]
        ax.plot([coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                linewidth=2.0, alpha=0.9, label='GT' if i == 0 else None)

    # Pred tour (dashed)
    for i in range(len(path_pred) - 1):
        a, b = path_pred[i], path_pred[i + 1]
        ax.plot([coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                linewidth=1.5, linestyle='--', alpha=0.9, label='Pred' if i == 0 else None)

    # Nodes
    ax.scatter(coords[:, 0], coords[:, 1], s=40, zorder=3)
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=80, marker='s', zorder=4)

    if annotate:
        for i, (x, y) in enumerate(coords):
            ax.text(x + 0.1, y + 0.1, str(i), fontsize=8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-axis_padding, grid_size + axis_padding)
    ax.set_ylim(-axis_padding, grid_size + axis_padding)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    if title is None:
        ax.set_title(f"GT vs Pred tours  |  len(GT)={len_true:.3f}, len(Pred)={len_pred:.3f}")
    else:
        ax.set_title(title + f"  |  len(GT)={len_true:.3f}, len(Pred)={len_pred:.3f}")
    ax.legend(loc='best')
    return fig, ax
