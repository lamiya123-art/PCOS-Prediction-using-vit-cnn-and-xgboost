"""
Quick test script to verify tabular data loading without requiring images.
"""
import pandas as pd
from pathlib import Path

def verify_tabular_data(data_dir="data"):
    data_dir = Path(data_dir)
    splits = ['train', 'val', 'test']
    
    print("\nVerifying tabular data structure...")
    all_ok = True
    
    for split in splits:
        csv_path = data_dir / "tabular" / f"{split}.csv"
        try:
            df = pd.read_csv(csv_path)
            n_samples = len(df)
            features = [c for c in df.columns if c != 'label']
            n_features = len(features)
            n_pcos = df['label'].sum()
            n_healthy = n_samples - n_pcos
            
            print(f"\n✓ {split}.csv loaded successfully:")
            print(f"  • {n_samples} total samples")
            print(f"  • {n_features} features: {', '.join(features)}")
            print(f"  • Labels: {n_pcos} PCOS, {n_healthy} healthy")
            
        except FileNotFoundError:
            print(f"\n✗ Error: {split}.csv not found at {csv_path}")
            all_ok = False
        except Exception as e:
            print(f"\n✗ Error loading {split}.csv: {str(e)}")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    verify_tabular_data()