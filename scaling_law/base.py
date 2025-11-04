import os, math, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import csv
import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir, config):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.config = config
        self.metrics_file = os.path.join(log_dir, 'metrics.csv')
        self.config_file = os.path.join(log_dir, 'config.json')
        # Save config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        # Prepare metrics file
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_acc', 'val_acc', 'timestamp'])
        # Set up logging
        logging.basicConfig(filename=os.path.join(log_dir, 'experiment.log'), level=logging.INFO)
        logging.info(f"Experiment started at {datetime.now().isoformat()}")
        logging.info(f"Config: {json.dumps(config)}")

    def log_metrics(self, epoch, train_acc, val_acc):
        timestamp = datetime.now().isoformat()
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_acc, val_acc, timestamp])
        logging.info(f"Epoch {epoch}: train_acc={train_acc}, val_acc={val_acc}")


class BaseTrainer:
    def __init__(self, application, model, optimizer, scaler, ema, device, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.ema = ema
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        log_dir = os.path.join('results', application, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.logger = ExperimentLogger(log_dir, config)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_ce, total_halt, total_em, total_steps = 0.0, 0.0, 0.0, 0
        for x_tokens, y_true in self.train_loader:
            x_tokens = x_tokens.to(self.device, non_blocking=True)
            y_true   = y_true.to(self.device,   non_blocking=True)
            y_state, z_state = self.model.init_state(batch_size=x_tokens.size(0), device=self.device)
            for _ in range(self.config['N_SUP']):
                self.optimizer.zero_grad(set_to_none=True)
                y_state, z_state, logits, halt_logit = self.model.forward_step(
                    x_tokens, y=y_state, z=z_state, n=self.config['N'], T=self.config['T'], k_last_ops=None
                )
                logits_nodes = logits.float().view(logits.size(0), self.config['N_NODES'], self.config['N_NODES'], logits.size(2)).mean(dim=2)
                loss_ce = F.cross_entropy(logits_nodes.reshape(-1, self.config['OUTPUT_TOKENS']), y_true.reshape(-1))
                with torch.no_grad():
                    em = (logits_nodes.argmax(dim=-1) == y_true).float().mean(dim=1)
                loss_halt = F.binary_cross_entropy_with_logits(halt_logit.float(), em)
                loss = loss_ce + loss_halt
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.ema is not None:
                    self.ema.update(self.model)
                total_ce   += loss_ce.detach().item()
                total_halt += loss_halt.detach().item()
                total_em   += em.mean().item()
                total_steps += 1
        
        training_acc = total_em/max(1,total_steps)
        print(f"Epoch {epoch:02d} | CE {total_ce/max(1,total_steps):.4f} | HaltBCE {total_halt/max(1,total_steps):.4f} | Node match {training_acc:.3f}")
        return training_acc

    @torch.no_grad()
    def evaluate(self, show_examples=3):
        self.model.eval()
        acc_list = []
        example_count = 0
        for x_tokens, y_true in self.val_loader:
            x_tokens = x_tokens.to(self.device)
            y_true   = y_true.to(self.device)
            y_state, z_state = self.model.init_state(batch_size=x_tokens.size(0), device=self.device)
            for _ in range(self.config['N_SUP']):
                y_state, z_state, logits, halt_logit = self.model.forward_step(
                    x_tokens, y=y_state, z=z_state, n=self.config['N'], T=self.config['T'], k_last_ops=None
                )
            logits_nodes = logits.float().view(logits.size(0), self.config['N_NODES'], self.config['N_NODES'], logits.size(2)).mean(dim=2)
            em = (logits_nodes.argmax(dim=-1) == y_true).float()
            acc = em.mean()
            acc_list.append(acc)
            if example_count < show_examples:
                preds = logits_nodes.argmax(dim=-1)
                xs = x_tokens.cpu().numpy()
                ys = y_true.cpu().numpy()
                preds_np = preds.cpu().numpy()
                for i in range(min(show_examples - example_count, xs.shape[0])):
                    print(f"\nExample {example_count + 1}:")
                    print("Adjacency matrix:")
                    print(xs[i].reshape(self.config['N_NODES'], self.config['N_NODES']))
                    print("Predicted communities:", preds_np[i])
                    print("True communities:", ys[i])
                    example_count += 1
        acc = torch.stack(acc_list).mean().item()
        print(f"Validation | Node match {acc:.3f}")
        return acc

#%
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _USE_TORCH_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _USE_TORCH_AMP = False

def make_grad_scaler(is_cuda: bool):
    if _USE_TORCH_AMP:
        try:
            return _GradScaler("cuda", enabled=is_cuda)
        except TypeError:
            return _GradScaler(enabled=is_cuda)
    else:
        return _GradScaler(enabled=is_cuda)

def amp_autocast(is_cuda: bool, use_amp: bool):
    if _USE_TORCH_AMP:
        try:
            return _autocast(device_type="cuda", enabled=(is_cuda and use_amp))
        except TypeError:
            return _autocast(enabled=(is_cuda and use_amp))
    else:
        return _autocast(enabled=(is_cuda and use_amp))

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self.shadow[name].mul_(d).add_(param.detach(), alpha=1.0 - d)

    def copy_to(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name])

from contextlib import contextmanager

@contextmanager
def use_ema_weights(model: torch.nn.Module, ema: EMA):
    backup = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    ema.copy_to(model)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])
