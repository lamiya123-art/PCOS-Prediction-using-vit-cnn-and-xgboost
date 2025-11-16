"""
Training script for Federated Multimodal PCOS Prediction.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys

# Import models from main.py (or separate model files)
from main import SimpleCNN, TinyViT, TabularMLP, FusionHead, LocalTrainer, LocalClient

# Import data loader
from data_loader import MultimodalPCOSDataset


def create_federated_datasets(
    num_clients: int = 4,
    samples_per_client: int = 200,
    use_real_data: bool = True,
    data_dir: str = "data",
    use_processed_tabular: bool = False,
    processed_tabular_dir: str = "data/tabular_processed",
):
    """
    Create federated datasets for each client.
    
    Args:
        num_clients: Number of federated clients
        samples_per_client: Samples per client (for synthetic data)
        use_real_data: Whether to use real data or synthetic
        data_dir: Directory containing real data
    
    Returns:
        List of (train_dataset, val_dataset) tuples for each client
    """
    clients_data = []
    
    for i in range(num_clients):
        if use_real_data:
            # TODO: If you have separate files per client, load them here
            # For now, we assume all clients share the same dataset
            # In real federated learning, each client would have different data
            
            # Split data into train/val for this client
            full_dataset = MultimodalPCOSDataset(
                data_dir=data_dir,
                split="train",
                load_real_data=True,
                use_processed_tabular=use_processed_tabular,
                processed_tabular_dir=processed_tabular_dir,
            )
            
            # Random split for this client (in real FL, each client has different data)
            indices = np.arange(len(full_dataset))
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=0.2, 
                random_state=42+i  # Different random state per client
            )
            
            # Create train dataset
            train_ds = MultimodalPCOSDataset(
                data_dir=data_dir,
                split="train",
                load_real_data=False  # We'll manually set the data
            )
            # Copy selected samples
            train_ds.n = len(train_idx)
            if hasattr(full_dataset, 'ultrasounds'):
                train_ds.ultrasounds = full_dataset.ultrasounds[train_idx]
                train_ds.faces = full_dataset.faces[train_idx]
                train_ds.tabular = full_dataset.tabular[train_idx]
                train_ds.labels = full_dataset.labels[train_idx]
            
            # Create val dataset
            val_ds = MultimodalPCOSDataset(
                data_dir=data_dir,
                split="val",
                load_real_data=False
            )
            val_ds.n = len(val_idx)
            if hasattr(full_dataset, 'ultrasounds'):
                val_ds.ultrasounds = full_dataset.ultrasounds[val_idx]
                val_ds.faces = full_dataset.faces[val_idx]
                val_ds.tabular = full_dataset.tabular[val_idx]
                val_ds.labels = full_dataset.labels[val_idx]
        
        else:
            # Use synthetic data
            from main import MultimodalPCOSDataset as SyntheticDataset
            ds = SyntheticDataset(n_samples=samples_per_client)
            ds.tabular[:, 0] += (i - num_clients/2) * 0.1
            
            indices = np.arange(len(ds))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            train_ds = MultimodalPCOSDataset(load_real_data=False)
            train_ds.n = len(train_idx)
            train_ds.ultrasounds = ds.ultrasounds[train_idx]
            train_ds.faces = ds.faces[train_idx]
            train_ds.tabular = ds.tabular[train_idx]
            train_ds.labels = ds.labels[train_idx]
            
            val_ds = MultimodalPCOSDataset(load_real_data=False)
            val_ds.n = len(val_idx)
            val_ds.ultrasounds = ds.ultrasounds[val_idx]
            val_ds.faces = ds.faces[val_idx]
            val_ds.tabular = ds.tabular[val_idx]
            val_ds.labels = ds.labels[val_idx]
        
        clients_data.append((train_ds, val_ds))
    
    return clients_data


def train(args):
    """Main training function."""
    print(f"🚀 Starting federated training...")
    print(f"   Clients: {args.num_clients}")
    print(f"   Rounds: {args.rounds}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Using {'real data' if args.use_real_data else 'synthetic data'}")
    
    # Create federated datasets
    clients_data = create_federated_datasets(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        use_real_data=args.use_real_data,
        data_dir=args.data_dir,
        use_processed_tabular=args.use_processed_tabular,
        processed_tabular_dir=args.processed_tabular_dir,
    )
    
    # Run federated training using the local FedAvg implementation
    from main import run_local_fedavg
    trained_clients = run_local_fedavg(clients_data, rounds=args.rounds)
    
    print("\n✅ Training complete!")
    return trained_clients


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Federated PCOS Prediction Model")
    
    # Federated learning parameters
    parser.add_argument("--num_clients", type=int, default=4, help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round")
    
    # Data parameters
    parser.add_argument("--samples_per_client", type=int, default=200, help="Samples per client (for synthetic)")
    parser.add_argument("--use_real_data", action="store_true", help="Use real data from data/ directory")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data")
    parser.add_argument("--use_processed_tabular", action="store_true", help="Use preprocessed tabular arrays (run preprocess_data.py first)")
    parser.add_argument("--processed_tabular_dir", type=str, default="data/tabular_processed", help="Directory of processed tabular arrays")
    
    args = parser.parse_args()
    
    # Update global settings in main.py
    from main import LOCAL_EPOCHS, BATCH_SIZE
    LOCAL_EPOCHS = args.local_epochs
    BATCH_SIZE = args.batch_size
    
    # Run training
    trained_clients = train(args)

