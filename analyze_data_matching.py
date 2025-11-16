"""
Analyze and display the relationship between ultrasound images and tabular data.
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

def extract_id_from_filename(filename):
    """Extract numeric ID from image filename."""
    # Try to extract ID from patterns like img_0_1023.jpg or img1.jpg
    match = re.search(r'img_0_(\d+)|img(\d+)', filename)
    if match:
        # Return the first non-None group
        return next(g for g in match.groups() if g is not None)
    return None

def analyze_data_split(split, data_dir="data"):
    """Analyze one data split (train/val/test)."""
    data_dir = Path(data_dir)
    print(f"\nAnalyzing {split} split:")
    
    # Load tabular data
    tabular_path = data_dir / "tabular" / f"{split}.csv"
    df = pd.read_csv(tabular_path)
    print(f"Tabular samples: {len(df)}")
    
    # Get ultrasound images
    img_dir = data_dir / "ultrasound" / split
    infected_dir = img_dir / "infected"
    noninfected_dir = img_dir / "non infected"
    
    # Collect image info
    images = defaultdict(list)  # label -> [image_paths]
    id_to_path = {}  # image_id -> path
    
    for label, directory in [(1, infected_dir), (0, noninfected_dir)]:
        if directory.exists():
            for img_path in directory.glob("*.jpg"):
                images[label].append(img_path)
                img_id = extract_id_from_filename(img_path.name)
                if img_id:
                    id_to_path[img_id] = img_path
    
    total_images = sum(len(imgs) for imgs in images.values())
    print(f"Total images: {total_images}")
    print(f"  PCOS (infected): {len(images[1])}")
    print(f"  Healthy (non-infected): {len(images[0])}")
    
    # Compare distributions
    tabular_pcos = (df['label'] == 1).sum()
    tabular_healthy = (df['label'] == 0).sum()
    print("\nClass distribution:")
    print(f"Tabular PCOS:    {tabular_pcos:4d} ({tabular_pcos/len(df)*100:.1f}%)")
    print(f"Tabular Healthy: {tabular_healthy:4d} ({tabular_healthy/len(df)*100:.1f}%)")
    
    image_pcos = len(images[1])
    image_total = total_images
    print(f"Image PCOS:      {image_pcos:4d} ({image_pcos/image_total*100:.1f}%)")
    print(f"Image Healthy:   {len(images[0]):4d} ({len(images[0])/image_total*100:.1f}%)")
    
    return df, images, id_to_path

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        df, images, id_to_path = analyze_data_split(split)
        print("\nImage IDs found:", len(id_to_path))
        if id_to_path:
            print("Sample IDs:", sorted(list(id_to_path.keys()))[:5], "...")