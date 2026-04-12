"""
model.py
PyTorch CNN for V2X spectrum sensing (5-class modulation recognition + binary).
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Fixed seeds for reproducibility across numpy, random, and PyTorch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Automatically select GPU if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class V2XSpectrumCNN(nn.Module):
    """
    CNN for 5-class modulation recognition from 64x64 spectrograms.
    Conv(1,32,3)->BN->ReLU->MP -> Conv(32,64,3)->BN->ReLU->MP ->
    Conv(64,128,3)->BN->ReLU->MP -> FC(128*8*8,128)->ReLU->Drop->FC(128,5)
    
    Architecture design choices:
      - 3 conv blocks with doubling channels (32->64->128): standard deep learning
        pattern to progressively increase feature abstraction capacity.
      - BatchNorm after each conv: stabilizes training by normalizing activations,
        reduces internal covariate shift, and acts as a mild regularizer.
      - MaxPool2d(2) after each block: halves spatial dimensions (64->32->16->8),
        reducing computation and providing translation invariance.
      - padding=1 with 3x3 kernels: preserves spatial dimensions before pooling.
      - Dropout(0.5): strong regularization to prevent overfitting on the small
        spectrogram dataset.
      - Single-channel input (grayscale): spectrograms are single-channel magnitude
        representations.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        # Feature extractor: 3 conv blocks with BatchNorm and ReLU activation
        # Input shape: (B, 1, 64, 64) — single-channel 64x64 spectrograms
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels, spatial 64 -> 32
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Block 2: 32 -> 64 channels, spatial 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Block 3: 64 -> 128 channels, spatial 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        # After 3 MaxPool2d(2): 64 -> 32 -> 16 -> 8
        # So flattened feature vector size = 128 * 8 * 8 = 8192
        self.classifier = nn.Sequential(
            # Fully connected: 8192 -> 128, with ReLU and 50% dropout
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
            # Output layer: 128 -> num_classes
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Extract spatial features through conv blocks
        x = self.features(x)
        # Flatten from (B, 128, 8, 8) to (B, 8192) for the FC layers
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class V2XBinaryCNN(nn.Module):
    """Binary: occupied vs vacant.
    
    Wraps V2XSpectrumCNN with num_classes=2 for binary spectrum sensing
    (channel occupied vs. vacant). Reuses the same architecture to ensure
    fair comparison between multi-class and binary classification.
    """
    def __init__(self):
        super().__init__()
        self.cnn = V2XSpectrumCNN(num_classes=2)
    def forward(self, x):
        return self.cnn(x)


class EarlyStopping:
    """Stops training when validation loss stops improving.
    
    Tracks the best (lowest) validation loss and counts consecutive epochs
    without improvement. Training stops when the counter reaches `patience`.
    
    Uses a minimum improvement threshold of 1e-4 to ignore negligible
    fluctuations that don't represent real improvement.
    """
    def __init__(self, patience=5):
        self.patience, self.counter, self.best, self.stop = patience, 0, None, False
    def step(self, val_loss):
        if self.best is None or val_loss < self.best - 1e-4:
            # New best loss found — reset counter
            self.best = val_loss; self.counter = 0
        else:
            # No improvement — increment counter
            self.counter += 1
            if self.counter >= self.patience: self.stop = True


def load_data(split="train", batch_size=64):
    """Load pre-generated spectrogram data from .npz files.
    
    Each .npz contains:
      - X: spectrograms of shape (N, 1, 64, 64) as float16
      - y: integer class labels
      - snr: SNR index (0-6 mapping to [-10, -5, 0, 5, 10, 15, 20] dB)
    
    Shuffling is only enabled for training to avoid data order bias;
    test/val loaders preserve order for consistent evaluation.
    num_workers=0 avoids multiprocessing issues with small datasets.
    """
    path = os.path.join(DATA_DIR, f"{split}.npz")
    data = np.load(path)
    X = torch.from_numpy(data["X"].astype(np.float32))
    y = torch.from_numpy(data["y"])
    snr = torch.from_numpy(data["snr"])
    return DataLoader(TensorDataset(X, y, snr), batch_size=batch_size,
                      shuffle=(split == "train"), num_workers=0)


def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001,
                checkpoint_name="best_model.pt"):
    """Train a CNN model with Adam optimizer and cosine annealing LR schedule.
    
    Hyperparameter choices:
      - lr=0.001: standard Adam learning rate; combined with cosine annealing
        provides a good balance between convergence speed and final accuracy.
      - CosineAnnealingLR: decays LR following a cosine curve from lr to ~0,
        which empirically yields better final accuracy than step decay for CNNs.
      - EarlyStopping patience=5: allows some tolerance for noisy validation
        metrics while preventing excessive overfitting.
      - Best model selection by validation accuracy (not loss) because
        classification accuracy is the primary research metric.
    
    Returns the trained model with the best validation accuracy loaded.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing: LR follows cos(π * t/T_max) curve from lr to ~0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early = EarlyStopping(patience=5)
    best_acc, best_state = 0.0, None

    print(f"Training on {DEVICE} ({num_epochs} epochs)")

    for epoch in range(1, num_epochs + 1):
        model.train()
        t_loss, t_ok, t_n = 0.0, 0, 0
        for Xb, yb, _ in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            # Accumulate loss * batch_size for proper averaging
            t_loss += loss.item() * Xb.size(0)
            t_ok += (logits.argmax(1) == yb).sum().item()
            t_n += Xb.size(0)
        scheduler.step()

        # Validation: compute accuracy without gradients for efficiency
        model.eval()
        v_ok, v_n = 0, 0
        with torch.no_grad():
            for Xb, yb, _ in val_loader:
                p = model(Xb.to(DEVICE)).argmax(1)
                v_ok += (p == yb.to(DEVICE)).sum().item(); v_n += yb.size(0)
        vacc = v_ok / max(v_n, 1)
        print(f"  Ep {epoch:3d} | TrAcc {t_ok/max(t_n,1):.4f} | ValAcc {vacc:.4f}")
        # Deep copy state_dict to avoid reference issues during continued training
        if vacc > best_acc: best_acc = vacc; best_state = copy.deepcopy(model.state_dict())
        # Feed 1-accuracy as the "loss" to early stopping (higher is better)
        early.step(1.0 - vacc)
        if early.stop: print("  >> Early stop"); break

    # Restore best model weights before returning
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, checkpoint_name))
    print(f"Best val acc: {best_acc:.4f}")
    return model


def evaluate_by_snr(model, test_loader, snr_levels=7):
    """Evaluate model accuracy stratified by SNR level.
    
    This is critical for V2X spectrum sensing research because adversarial
    robustness varies significantly with signal quality. Low SNR signals
    are inherently harder to classify and may be more vulnerable to attacks.
    
    SNR levels 0-6 correspond to [-10, -5, 0, 5, 10, 15, 20] dB.
    
    Returns:
      - accs: list of accuracy values, one per SNR level
      - all predictions, labels, and SNR indices as flat tensors
    """
    model.eval()
    # Per-SNR correct/total counters
    sc, st = [0]*snr_levels, [0]*snr_levels
    # Collect all predictions and labels for confusion matrix computation
    ap, al, asn = [], [], []
    with torch.no_grad():
        for Xb, yb, snr in test_loader:
            p = model(Xb.to(DEVICE)).argmax(1).cpu()
            ap.append(p); al.append(yb); asn.append(snr)
            # Count correct predictions per SNR bucket
            for p_, y_, s_ in zip(p, yb, snr):
                sc[s_.item()] += (p_ == y_).item(); st[s_.item()] += 1
    accs = [sc[i]/max(st[i],1) for i in range(snr_levels)]
    return accs, torch.cat(ap), torch.cat(al), torch.cat(asn)
