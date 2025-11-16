"""
Helper script to set up your data structure for PCOS prediction.

This script will help you:
1. Create example CSV files with the expected format
2. Provide instructions for organizing your images
"""

import pandas as pd
from pathlib import Path
import numpy as np


def create_example_csv(data_dir: str = "data"):
    """Create example CSV files showing the expected format."""
    
    # Define example features (customize these based on your actual data)
    features = [
        'LH',           # Luteinizing hormone
        'FSH',          # Follicle stimulating hormone
        'AMH',          # Anti-Müllerian hormone
        'BMI',          # Body mass index
        'age',          # Age in years
        'waist_hip',    # Waist-to-hip ratio
        'testosterone', # Testosterone level
        'insulin',      # Insulin level
        'glucose',      # Glucose level
        'cholesterol'   # Cholesterol level
    ]
    
    # Create example data for train split
    n_train = 100
    n_val = 30
    n_test = 30
    
    # Generate example data
    np.random.seed(42)
    
    def generate_data(n_samples, pcos_prob=0.3):
        """Generate synthetic PCOS data."""
        data = {}
        
        # Simulate realistic ranges for each feature
        data['LH'] = np.random.normal(4.5, 2.0, n_samples)
        data['FSH'] = np.random.normal(4.0, 1.5, n_samples)
        data['AMH'] = np.abs(np.random.normal(5.0, 3.0, n_samples))
        data['BMI'] = np.random.normal(25.0, 4.0, n_samples)
        data['age'] = np.random.randint(18, 40, n_samples)
        data['waist_hip'] = np.random.uniform(0.7, 1.0, n_samples)
        data['testosterone'] = np.abs(np.random.normal(2.0, 0.8, n_samples))
        data['insulin'] = np.abs(np.random.normal(15.0, 8.0, n_samples))
        data['glucose'] = np.random.normal(90.0, 15.0, n_samples)
        data['cholesterol'] = np.random.normal(180.0, 35.0, n_samples)
        
        # Create labels (1 = PCOS, 0 = No PCOS)
        # Higher LH, lower FSH, higher BMI more likely to be PCOS
        pcos_score = (
            (data['LH'] > 5) * 0.3 +
            (data['FSH'] < 4) * 0.2 +
            (data['BMI'] > 25) * 0.3 +
            (data['waist_hip'] > 0.85) * 0.2
        )
        
        pcos = (pcos_score + np.random.random(n_samples)) > (1 - pcos_prob)
        data['label'] = pcos.astype(int)
        
        return pd.DataFrame(data)
    
    # Generate data for each split
    train_df = generate_data(n_train)
    val_df = generate_data(n_val)
    test_df = generate_data(n_test)
    
    # Save to CSV
    output_dir = Path(data_dir) / "tabular"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    print(f"✅ Created example CSV files in {output_dir}/")
    print(f"   train.csv: {n_train} samples")
    print(f"   val.csv: {n_val} samples")
    print(f"   test.csv: {n_test} samples")
    print(f"\n📋 Features included: {', '.join(features)}")
    print(f"\n⚠️  Replace these with YOUR real data!")


def print_instructions():
    """Print instructions for organizing data."""
    
    instructions = """
═══════════════════════════════════════════════════════════════
  📁 HOW TO SET UP YOUR DATA
═══════════════════════════════════════════════════════════════

1️⃣  TABULAR DATA (CSV files)
   Location: data/tabular/
   
   ✅ Run: python setup_data.py --create_csv
      This creates example CSVs showing the format
   
   📝 Replace with YOUR real data:
      - Each row = one patient
      - Columns = your clinical/hormonal features
      - One column named 'label' (0 or 1 for PCOS)
   
   
2️⃣  ULTRASOUND IMAGES
   Location: data/ultrasound/train/, data/ultrasound/val/, etc.
   
   Place your ultrasound images (DICOM, JPG, PNG) here
   - Filename doesn't matter, but should match row order in CSV
   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .dcm
   
   
3️⃣  FACIAL IMAGES  
   Location: data/facial/train/, data/facial/val/, etc.
   
   Place your facial images here
   - Same filename/order as ultrasound images
   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff
   
   
💡 TIP: Name files with IDs that match your CSV rows for consistency


═══════════════════════════════════════════════════════════════
"""
    print(instructions)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up data structure for PCOS prediction")
    parser.add_argument("--create_csv", action="store_true", help="Create example CSV files")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    
    args = parser.parse_args()
    
    if args.create_csv:
        create_example_csv(args.data_dir)
    else:
        print("Choose an option:\n")
        print("1. Create example CSV files:")
        print("   python setup_data.py --create_csv")
        print("\n2. Show instructions:")
        print("   python setup_data.py")
        
        print_instructions()


