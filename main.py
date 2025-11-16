"""
Federated Learning Prototype: Multimodal PCOS Prediction with CNN, ViT, XGBoost and Explainability

File: fl_pcos_prototype.py
Purpose: Minimal, runnable prototype that simulates federated clients with
- synthetic ultrasound image data,
- synthetic facial images,
- synthetic tabular hormonal/clinical features.

This revised version addresses import-time failures caused by optional visualization
libraries (e.g., matplotlib). Important changes:
- Removed any top-level import that can trigger matplotlib import errors (shap is now
  imported lazily inside the SHAP routine).
- Added a fast smoke-test runner (no plotting) that verifies core functionality.
- Ensures the script runs in environments without matplotlib or shap installed.

Notes / Requirements:
- Python 3.8+
- pip install the mandatory libs: torch torchvision captum xgboost flwr numpy scikit-learn
  (shap is OPTIONAL; if absent the code uses XGBoost gain-based fallback.)

Use this as a development prototype. Replace synthetic data with real data and add
secure aggregation / DP before production use.
"""

import os
import random
import copy
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Flower for federated orchestration
import flwr as fl
# XGBoost for tabular
import xgboost as xgb
# Captum for attribution (no matplotlib)
from captum.attr import Saliency

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

# -------------------------------
# Configuration
# -------------------------------
SEED = 42
NUM_CLIENTS = 4
LOCAL_EPOCHS = 1
ROUNDS = 1  # smoke-test default
BATCH_SIZE = 16
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------
# Dataset
# -------------------------------
class MultimodalPCOSDataset(Dataset):
    def __init__(self, data_dir: str = "data", split: str = "train", img_size: int = IMG_SIZE, real_dataset=None):
        """
        Wrapper dataset used by the training code. By default it loads a real dataset
        from `data_loader.MultimodalPCOSDataset`. If `real_dataset` is provided we use
        that directly (useful for creating client-specific subsets).
        """
        if real_dataset is not None:
            # real_dataset is expected to be an instance returned by data_loader.MultimodalPCOSDataset
            self.dataset = real_dataset
            self.n_samples = len(self.dataset)
        else:
            from data_loader import MultimodalPCOSDataset as RealDataset
            self.dataset = RealDataset(data_dir=data_dir, split=split, img_size=img_size, load_real_data=True)
            self.n_samples = len(self.dataset)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        us_tensor, tab, label = self.dataset[idx]
        # Return empty tensor for face since we don't use it
        face = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        return us_tensor, face, tab, label

def create_federated_datasets(num_clients: int = NUM_CLIENTS, samples_per_client: int = 200) -> List[MultimodalPCOSDataset]:
    """
    Create per-client datasets by splitting the real training dataset.

    Each client receives up to `samples_per_client` samples sampled without replacement
    from the full train split. If the total samples are insufficient, sampling with
    replacement is used to fill remaining slots.
    """
    from data_loader import MultimodalPCOSDataset as RealDataset

    full = RealDataset(data_dir="data", split="train", img_size=IMG_SIZE, load_real_data=True)
    n_total = len(full)
    indices = np.arange(n_total)
    ptr = 0
    clients: List[MultimodalPCOSDataset] = []
    for i in range(num_clients):
        # Prefer stratified sampling for each client to preserve label balance
        labels = np.asarray(full.labels)
        try:
            if samples_per_client <= n_total:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=samples_per_client, random_state=SEED + i)
                sel = next(sss.split(indices, labels))[0]
            else:
                # Need to sample with replacement; sample per-class proportional to class sizes
                unique, counts = np.unique(labels, return_counts=True)
                probs = counts / counts.sum()
                sel = []
                for cls, p in zip(unique, probs):
                    k = int(round(p * samples_per_client))
                    cls_idx = np.where(labels == cls)[0]
                    if len(cls_idx) == 0:
                        continue
                    chosen = np.random.choice(cls_idx, size=k, replace=True)
                    sel.extend(chosen.tolist())
                sel = np.array(sel, dtype=int)
                # If rounding produced different size, adjust by sampling uniformly
                if len(sel) < samples_per_client:
                    extra = np.random.choice(n_total, samples_per_client - len(sel), replace=True)
                    sel = np.concatenate([sel, extra])
                elif len(sel) > samples_per_client:
                    sel = sel[:samples_per_client]
        except Exception:
            # Fallback: sample randomly with replacement if stratified split fails
            sel = np.random.choice(n_total, samples_per_client, replace=True)

        # Create a RealDataset subset by copying arrays
        sub = RealDataset(data_dir="data", split="train", img_size=IMG_SIZE, load_real_data=True)
        # If the real dataset exposes processed arrays, overwrite them with the subset
        if hasattr(full, 'tabular'):
            sub.tabular = full.tabular[sel]
        if hasattr(full, 'labels'):
            sub.labels = full.labels[sel]
        sub.n_samples = len(sel)

        clients.append(MultimodalPCOSDataset(real_dataset=sub))

    return clients

# -------------------------------
# Models
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TinyViT(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=8, in_chans=3, embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, batch_first=True)
            for _ in range(depth)
        ])
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0,2,1,3,4)
        patches = patches.contiguous().view(B, patches.shape[1], -1)
        x_p = self.proj(patches)
        cls = self.cls_token.expand(B, -1, -1)
        x_cat = torch.cat([cls, x_p], dim=1)
        x_cat = x_cat + self.pos_emb
        for block in self.transformer_blocks:
            x_cat = block(x_cat)
        cls_out = x_cat[:,0]
        return self.fc(cls_out)

class TabularMLP(nn.Module):
    def __init__(self, in_dim=5, out_dim=32):  # Changed in_dim to 5 for your actual features
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),  # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),     # Added Dropout
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class FusionHead(nn.Module):
    def __init__(self, cnn_dim=128, vit_dim=128, tab_dim=32, hidden=192):
        super().__init__()
        # Ensemble CNN + ViT features for ultrasound + tabular features
        in_dim = cnn_dim + vit_dim + tab_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, f_cnn, f_vit, f_tab):
        # Concatenate CNN (ultrasound), ViT (ultrasound), and tabular features
        x = torch.cat([f_cnn, f_vit, f_tab], dim=1)
        return self.mlp(x).squeeze(1)

# -------------------------------
# Training utilities
# -------------------------------
class LocalTrainer:
    def __init__(self, cnn, vit, tab_mlp, fusion, lr=1e-3, device=DEVICE):
        self.cnn = cnn.to(device)
        self.vit = vit.to(device)
        self.tab_mlp = tab_mlp.to(device)
        self.fusion = fusion.to(device)
        params = list(self.cnn.parameters()) + list(self.vit.parameters()) + list(self.tab_mlp.parameters()) + list(self.fusion.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def train_epoch(self, dataloader):
        self.cnn.train(); self.tab_mlp.train(); self.fusion.train()
        losses = []
        for us, _, tab, label in dataloader:  # _ is unused face tensor
            us = us.to(self.device)
            tab = tab.to(self.device)
            label = label.float().to(self.device)
            f_cnn = self.cnn(us)
            # Compute ViT features from the same ultrasound input (ensemble)
            f_vit = self.vit(us)
            f_tab = self.tab_mlp(tab)
            logits = self.fusion(f_cnn, f_vit, f_tab)
            loss = self.criterion(logits, label)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def predict_probs(self, dataloader):
        self.cnn.eval(); self.tab_mlp.eval(); self.fusion.eval()
        ys = []
        ys_prob = []
        with torch.no_grad():
            for us, _, tab, label in dataloader:  # _ is unused face tensor
                us = us.to(self.device)
                tab = tab.to(self.device)
                f_cnn = self.cnn(us)
                f_vit = self.vit(us)
                f_tab = self.tab_mlp(tab)
                logits = self.fusion(f_cnn, f_vit, f_tab)
                probs = torch.sigmoid(logits).cpu().numpy()
                ys_prob.extend(probs.tolist())
                ys.extend(label.numpy().tolist())
        return np.array(ys), np.array(ys_prob)

# -------------------------------
# Federated client wrapper
# -------------------------------

def get_state_dicts(cnn, vit, tab_mlp, fusion) -> Dict[str, dict]:
    return {
        'cnn': cnn.state_dict(),
        'vit': vit.state_dict(),
        'tab_mlp': tab_mlp.state_dict(),
        'fusion': fusion.state_dict()
    }

def set_state_dicts(cnn, vit, tab_mlp, fusion, state_dicts: Dict[str, dict]):
    cnn.load_state_dict(state_dicts['cnn'])
    vit.load_state_dict(state_dicts['vit'])
    tab_mlp.load_state_dict(state_dicts['tab_mlp'])
    fusion.load_state_dict(state_dicts['fusion'])

class FLClient(fl.client.NumPyClient):
    def __init__(self, train_ds: MultimodalPCOSDataset, val_ds: MultimodalPCOSDataset):
        self.cnn = SimpleCNN(out_dim=128)
        # Use ViT on grayscale ultrasound (in_chans=1) so it can process the same US tensor
        self.vit = TinyViT(in_chans=1)
        self.tab_mlp = TabularMLP(in_dim=5, out_dim=32)  # 5 features from the CSV
        self.fusion = FusionHead(cnn_dim=128, vit_dim=128, tab_dim=32)
        self.trainer = LocalTrainer(self.cnn, self.vit, self.tab_mlp, self.fusion)
        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        self.xgb_model = None

    def get_parameters(self):
        import pickle
        state = get_state_dicts(self.cnn, self.vit, self.tab_mlp, self.fusion)
        pickled = pickle.dumps(state)
        return [np.frombuffer(pickled, dtype=np.uint8)]

    def set_parameters(self, parameters):
        import pickle
        pickled = parameters[0].tobytes()
        state = pickle.loads(pickled)
        set_state_dicts(self.cnn, self.vit, self.tab_mlp, self.fusion, state)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for epoch in range(LOCAL_EPOCHS):
            loss = self.trainer.train_epoch(self.train_loader)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        ys, ys_prob = self.trainer.predict_probs(self.val_loader)
        auc = roc_auc_score(ys, ys_prob) if len(np.unique(ys)) > 1 else 0.5
        loss = float(1.0 - auc)
        return loss, len(self.val_loader.dataset), {"auc": float(auc)}

# -------------------------------
# XGBoost + optional SHAP (lazy import)
# -------------------------------

def train_local_xgboost(train_ds: MultimodalPCOSDataset, val_ds: MultimodalPCOSDataset):
    # Collect all tabular data from the dataloader
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    
    # Get tabular data from a single batch (contains all data)
    _, _, X_train, y_train = next(iter(train_loader))
    _, _, X_val, y_val = next(iter(val_loader))
    
    # Convert to numpy for XGBoost
    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_val = X_val.numpy()
    y_val = y_val.numpy()
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {'objective':'binary:logistic', 'eval_metric':'auc', 'verbosity':0}
    bst = xgb.train(params, dtrain, num_boost_round=50, evals=[(dval, 'val')], early_stopping_rounds=5)
    return bst


def compute_shap_for_xgb(model: xgb.Booster, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute SHAP values if shap is installed; otherwise return a shap-like fallback
    based on XGBoost feature gain. This function performs lazy import of shap to
    avoid import-time failures in environments missing matplotlib.
    """
    try:
        import shap as _shap
        # Some shap versions import matplotlib lazily; importing inside try guards against failure
        explainer = _shap.TreeExplainer(model)
        try:
            shap_vals = explainer.shap_values(X)
        except Exception:
            shap_res = explainer(X)
            shap_vals = getattr(shap_res, 'values', None)
        return shap_vals
    except Exception:
        # Fallback: use XGBoost feature gain normalized to sum=1 and broadcast to samples
        fmap = model.get_score(importance_type='gain')
        n_features = X.shape[1]
        gains = np.zeros(n_features, dtype=float)
        for k, v in fmap.items():
            if k.startswith('f'):
                idx = int(k[1:])
                if idx < n_features:
                    gains[idx] = v
        if gains.sum() > 0:
            gains = gains / gains.sum()
        # Broadcast to (n_samples, n_features)
        shap_like = np.tile(gains.reshape(1, -1), (X.shape[0], 1))
        return shap_like

# -------------------------------
# CNN saliency (Captum) - returns raw attribution array (no plotting)
# -------------------------------

def cnn_saliency_map(cnn: nn.Module, input_image: torch.Tensor, target_class: int = 1) -> np.ndarray:
    cnn.eval()
    saliency = Saliency(cnn)
    input_image = input_image.requires_grad_()
    attributions = saliency.attribute(input_image, target=target_class)
    return attributions.detach().cpu().numpy()

# -------------------------------
# Federated simulation helper
# -------------------------------

def start_simulation(clients_data: List[Tuple[MultimodalPCOSDataset, MultimodalPCOSDataset]], rounds: int = ROUNDS):
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        train_ds, val_ds = clients_data[idx]
        return FLClient(train_ds, val_ds)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=len(clients_data),
        min_available_clients=len(clients_data)
    )

    fl.simulation.start_simulation(client_fn=client_fn,
                                   num_clients=len(clients_data),
                                   config=fl.server.ServerConfig(num_rounds=rounds),
                                   strategy=strategy)

# -------------------------------
# Local single-process FedAvg (Windows-friendly)
# -------------------------------

def _average_state_dicts(dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg: Dict[str, torch.Tensor] = {}
    for k in dicts[0].keys():
        tensors = [d[k].detach().cpu() for d in dicts]
        t0 = tensors[0]
        if t0.is_floating_point():
            stacked = torch.stack(tensors)
            avg_val = stacked.mean(dim=0)
        else:
            # For integer/bool buffers (e.g., BatchNorm num_batches_tracked), keep first
            avg_val = t0.clone()
        avg[k] = avg_val
    return avg

def _average_multi_models(states: List[Dict[str, dict]]) -> Dict[str, dict]:
    keys = states[0].keys()  # ['cnn','vit','tab_mlp','fusion']
    out: Dict[str, dict] = {}
    for key in keys:
        module_dicts = [s[key] for s in states]
        out[key] = _average_state_dicts(module_dicts)
    return out

def run_local_fedavg(clients_data: List[Tuple[MultimodalPCOSDataset, MultimodalPCOSDataset]], rounds: int = ROUNDS):
    # Create one client per dataset (single-process, no Ray)
    clients: List[FLClient] = []
    for train_ds, val_ds in clients_data:
        clients.append(FLClient(train_ds, val_ds))

    # Initialize global parameters from first client
    global_state = get_state_dicts(clients[0].cnn, clients[0].vit, clients[0].tab_mlp, clients[0].fusion)
    for c in clients:
        set_state_dicts(c.cnn, c.vit, c.tab_mlp, c.fusion, global_state)

    for r in range(rounds):
        local_states: List[Dict[str, dict]] = []
        for c in clients:
            for _ in range(LOCAL_EPOCHS):
                _ = c.trainer.train_epoch(c.train_loader)
            local_states.append(get_state_dicts(c.cnn, c.vit, c.tab_mlp, c.fusion))

        # Average and broadcast
        global_state = _average_multi_models(local_states)
        for c in clients:
            set_state_dicts(c.cnn, c.vit, c.tab_mlp, c.fusion, global_state)

    # Simple eval
    aucs = []
    for c in clients:
        ys, ys_prob = c.trainer.predict_probs(c.val_loader)
        auc = roc_auc_score(ys, ys_prob) if len(np.unique(ys)) > 1 else 0.5
        aucs.append(float(auc))
    print(f"Local FedAvg done. Mean AUC over {len(aucs)} clients: {np.mean(aucs):.3f}")
    return clients  # Return trained clients so they can be used for testing

# -------------------------------
# Model inference/testing function
# -------------------------------

def test_model_on_sample(cnn, vit, tab_mlp, fusion, ultrasound, face, tabular):
    """
    Test the trained model on a single sample.
    
    Args:
        cnn: Trained CNN model for ultrasound
        vit: Trained ViT model for face
        tab_mlp: Trained MLP for tabular data
        fusion: Trained fusion head
        ultrasound: numpy array shape (1, H, W) or (H, W)
        face: numpy array shape (3, H, W) or (H, W, 3)
        tabular: numpy array shape (10,)
    
    Returns:
        float: Probability of PCOS (0-1)
    """
    # Prepare inputs
    if len(ultrasound.shape) == 2:
        ultrasound = ultrasound.reshape(1, 1, ultrasound.shape[0], ultrasound.shape[1])
    elif len(ultrasound.shape) == 3:
        ultrasound = ultrasound.reshape(1, *ultrasound.shape)
    if not isinstance(ultrasound, torch.Tensor):
        ultrasound = torch.tensor(ultrasound, dtype=torch.float32)
    
    if len(face.shape) == 3 and face.shape[0] != 3:
        face = face.transpose(2, 0, 1)  # Convert (H, W, 3) to (3, H, W)
    if len(face.shape) == 3:
        face = face.reshape(1, *face.shape)
    if not isinstance(face, torch.Tensor):
        face = torch.tensor(face, dtype=torch.float32)
    
    if not isinstance(tabular, torch.Tensor):
        tabular = torch.tensor(tabular, dtype=torch.float32)
    if tabular.dim() == 1:
        tabular = tabular.reshape(1, -1)
    
    # Get predictions
    cnn.eval()
    vit.eval()
    tab_mlp.eval()
    fusion.eval()
    
    with torch.no_grad():
        # Ensemble: compute both CNN and ViT features from the ultrasound image
        f_cnn = cnn(ultrasound)
        f_vit = vit(ultrasound)
        f_tab = tab_mlp(tabular)
        logits = fusion(f_cnn, f_vit, f_tab)
        prob = torch.sigmoid(logits).item()
    
    return prob

# -------------------------------
# Simple smoke tests
# -------------------------------

def run_smoke_test():
    print("Running smoke test with real data...")
    
    # Create datasets using real data
    train_ds = MultimodalPCOSDataset(data_dir="data", split="train")
    val_ds = MultimodalPCOSDataset(data_dir="data", split="val")
    test_ds = MultimodalPCOSDataset(data_dir="data", split="test")
    
    # For smoke test, we'll just use a small portion
    train_idx = np.random.choice(len(train_ds), size=100, replace=False)
    val_idx = np.random.choice(len(val_ds), size=20, replace=False)
    
    # Create a single client with the sampled data
    client_train = MultimodalPCOSDataset(data_dir="data", split="train")
    client_val = MultimodalPCOSDataset(data_dir="data", split="val")
    clients_data = [(client_train, client_val)]

    # Run a single-round local training (no federated for smoke test)
    trained_clients = run_local_fedavg(clients_data, rounds=1)

    # Train XGBoost locally on client 0's tabular and compute fallback SHAP-like values
    xgb_model = train_local_xgboost(clients_data[0][0], clients_data[0][1])
    # Extract tabular features from the validation dataset for SHAP
    val_loader = DataLoader(clients_data[0][1], batch_size=min(32, len(clients_data[0][1])), shuffle=False)
    us_batch, face_batch, X_val_full, y_val_full = next(iter(val_loader))
    X_shap = X_val_full[:10].numpy() if X_val_full.shape[0] >= 1 else X_val_full.numpy()
    shap_vals = compute_shap_for_xgb(xgb_model, X_shap)
    assert shap_vals is not None, "SHAP-like output must not be None"
    print("SHAP-like shape:", getattr(shap_vals, 'shape', None))

    # Saliency map check using a sample ultrasound from the val batch
    cnn = SimpleCNN()
    sample_img = us_batch[0:1]  # shape: [1, 1, H, W]
    sal = cnn_saliency_map(cnn, sample_img)
    assert isinstance(sal, np.ndarray)
    print("Saliency shape:", sal.shape)

    # Example: Test model on a sample
    print("\nTesting model on a sample...")
    # Prepare sample inputs for the test helper
    sample_ultrasound = sample_img[0].squeeze(0).detach().cpu().numpy()  # (H, W)
    sample_face = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    sample_tabular = X_val_full[0].numpy()

    # Get a trained model from a client
    client0_cnn = trained_clients[0].cnn
    client0_vit = trained_clients[0].vit
    client0_tab = trained_clients[0].tab_mlp
    client0_fusion = trained_clients[0].fusion

    prob = test_model_on_sample(client0_cnn, client0_vit, client0_tab, client0_fusion,
                                 sample_ultrasound, sample_face, sample_tabular)
    true_label = int(y_val_full[0].item())
    print(f"Sample prediction: {prob:.3f} (true label: {true_label})")
    
    print("Smoke test passed.")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == '__main__':
    # Default: run smoke test to verify core functionality without plotting
    run_smoke_test()
    print("Done. If you want a full federated run, set FULL_RUN=1 in the environment.")
    if os.environ.get('FULL_RUN') == '1':
        print("Starting full federated simulation (this may take a while)...")
        clients = create_federated_datasets(num_clients=NUM_CLIENTS, samples_per_client=400)
        clients_data = []
        for ds in clients:
            # ds is a MultimodalPCOSDataset wrapper around a RealDataset instance
            full_sub = ds.dataset
            n = len(full_sub)
            # Stratified split to preserve label proportions in train/val
            train_idx, val_idx = train_test_split(
                np.arange(n), test_size=0.2, random_state=SEED, stratify=full_sub.labels
            )

            from data_loader import MultimodalPCOSDataset as RealDataset

            # Create train subset
            train_sub = RealDataset(data_dir="data", split="train", img_size=IMG_SIZE, load_real_data=True)
            if hasattr(full_sub, 'tabular'):
                train_sub.tabular = full_sub.tabular[train_idx]
            if hasattr(full_sub, 'labels'):
                train_sub.labels = full_sub.labels[train_idx]
            train_sub.n_samples = len(train_idx)

            # Create val subset
            val_sub = RealDataset(data_dir="data", split="train", img_size=IMG_SIZE, load_real_data=True)
            if hasattr(full_sub, 'tabular'):
                val_sub.tabular = full_sub.tabular[val_idx]
            if hasattr(full_sub, 'labels'):
                val_sub.labels = full_sub.labels[val_idx]
            val_sub.n_samples = len(val_idx)

            # Wrap them for the training loop
            train_wrapper = MultimodalPCOSDataset(real_dataset=train_sub)
            val_wrapper = MultimodalPCOSDataset(real_dataset=val_sub)
            clients_data.append((train_wrapper, val_wrapper))
        start_simulation(clients_data, rounds=3)

