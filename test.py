"""
Testing and inference script for PCOS prediction.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Import from main.py
from main import test_model_on_sample, SimpleCNN, TinyViT, TabularMLP, FusionHead


def test_single_sample(
    model_path: str,
    ultrasound_path: str,
    facial_path: str,
    tabular_features: list
):
    """
    Test the model on a single patient sample.
    
    Args:
        model_path: Path to saved model
        ultrasound_path: Path to ultrasound image
        facial_path: Path to facial image
        tabular_features: List of tabular feature values
    """
    print(f"🔍 Testing model from: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path)
    
    cnn = SimpleCNN()
    vit = TinyViT()
    tab_mlp = TabularMLP(in_dim=len(tabular_features))
    fusion = FusionHead()
    
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    vit.load_state_dict(checkpoint['vit_state_dict'])
    tab_mlp.load_state_dict(checkpoint['tab_mlp_state_dict'])
    fusion.load_state_dict(checkpoint['fusion_state_dict'])
    
    # Load and preprocess images
    img_size = 64
    
    # Ultrasound
    us_img = Image.open(ultrasound_path).convert('L')
    us_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ultrasound = us_transform(us_img).unsqueeze(0)  # Add batch dimension
    
    # Facial
    face_img = Image.open(facial_path).convert('RGB')
    face_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    face = face_transform(face_img).unsqueeze(0)
    
    # Tabular features
    tabular = torch.tensor(tabular_features, dtype=torch.float32).unsqueeze(0)
    
    # Get prediction
    probability = test_model_on_sample(
        cnn, vit, tab_mlp, fusion,
        ultrasound.numpy(),
        face.numpy(),
        tabular.numpy().flatten()
    )
    
    print(f"\n📊 Prediction Results:")
    print(f"   PCOS Probability: {probability:.3f}")
    print(f"   Likelihood: {'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'}")
    
    return probability


def test_dataset(model_path: str, data_dir: str = "data"):
    """
    Test the model on the validation/test dataset.
    """
    from data_loader import MultimodalPCOSDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    
    print(f"🧪 Testing on dataset from: {data_dir}")
    
    # Load model
    checkpoint = torch.load(model_path)
    
    cnn = SimpleCNN()
    vit = TinyViT()
    # Get tabular dimension from checkpoint or default
    tab_dim = checkpoint.get('n_tabular_features', 10)
    tab_mlp = TabularMLP(in_dim=tab_dim)
    fusion = FusionHead()
    
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    vit.load_state_dict(checkpoint['vit_state_dict'])
    tab_mlp.load_state_dict(checkpoint['tab_mlp_state_dict'])
    fusion.load_state_dict(checkpoint['fusion_state_dict'])
    
    # Load test dataset
    try:
        dataset = MultimodalPCOSDataset(data_dir=data_dir, split="test", load_real_data=True)
    except FileNotFoundError:
        print(f"⚠ No test data found, using validation set...")
        dataset = MultimodalPCOSDataset(data_dir=data_dir, split="val", load_real_data=True)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    cnn.eval()
    vit.eval()
    tab_mlp.eval()
    fusion.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.to(device)
    vit.to(device)
    tab_mlp.to(device)
    fusion.to(device)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for us, face, tab, label in dataloader:
            us = us.to(device)
            face = face.to(device)
            tab = tab.to(device)
            
            f_cnn = cnn(us)
            f_vit = vit(face)
            f_tab = tab_mlp(tab)
            logits = fusion(f_cnn, f_vit, f_tab)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(label.numpy())
    
    # Metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    predictions = (all_probs > 0.5).astype(int)
    
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    accuracy = accuracy_score(all_labels, predictions)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   AUC: {auc:.3f}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, predictions, target_names=['No PCOS', 'PCOS']))
    
    return auc, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PCOS Prediction Model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_mode", choices=["single", "dataset"], default="dataset",
                       help="Test on single sample or full dataset")
    
    # Single sample arguments
    parser.add_argument("--ultrasound", type=str, help="Path to ultrasound image (for single mode)")
    parser.add_argument("--facial", type=str, help="Path to facial image (for single mode)")
    parser.add_argument("--tabular", nargs="+", type=float, help="Tabular features (for single mode)")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory (for dataset mode)")
    
    args = parser.parse_args()
    
    if args.test_mode == "single":
        if not (args.ultrasound and args.facial and args.tabular):
            parser.error("--ultrasound, --facial, and --tabular are required for single mode")
        
        test_single_sample(
            args.model_path,
            args.ultrasound,
            args.facial,
            args.tabular
        )
    else:
        test_dataset(args.model_path, args.data_dir)

