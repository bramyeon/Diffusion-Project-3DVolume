import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "./data"

def split(root: str, dataset: str, voxel_dir: str, pc_dir: str, category: list):
    dataset_dir = os.path.join(root, dataset)
    
    voxel_save_dir = Path(os.path.join(root, voxel_dir))
    voxel_save_dir.mkdir(exist_ok=True, parents=True)
    
    pc_save_dir = Path(os.path.join(root, pc_dir))
    pc_save_dir.mkdir(exist_ok=True, parents=True)

    for split in ["train", "val", "test"]: 
        voxel_save_split_dir = Path(os.path.join(voxel_save_dir, split))
        voxel_save_split_dir.mkdir(exist_ok=True, parents=True)
        
        pc_save_split_dir = Path(os.path.join(pc_save_dir, split))
        pc_save_split_dir.mkdir(exist_ok=True, parents=True)
        
        for cat in tqdm(category):
            voxel_save_cat_dir = Path(os.path.join(voxel_save_split_dir, cat))
            voxel_save_cat_dir.mkdir(exist_ok=True, parents=True)
            voxel_path = os.path.join(dataset_dir, f"{cat}_voxels_{split}.npy")
            voxel_data = np.load(voxel_path)
            
            pc_save_cat_dir = Path(os.path.join(pc_save_split_dir, cat))            
            pc_save_cat_dir.mkdir(exist_ok=True, parents=True)
            pc_path = os.path.join(dataset_dir, f"{cat}_pointclouds_{split}.npy")
            pc_data = np.load(pc_path)
            
            # os.remove(voxel_path)
            # os.remove(pc_path)

            assert voxel_data.shape[0] == pc_data.shape[0]
            for i in range(voxel_data.shape[0]):
                np.save(os.path.join(voxel_save_cat_dir, f"{cat}_{i}.npy"), voxel_data[i])
                np.save(os.path.join(pc_save_cat_dir, f"{cat}_{i}.npy"), pc_data[i])

if __name__ == "__main__":
    category = ["airplane", "chair", "table"]
    split(DATA_DIR, "hdf5_data", "preprocessed/voxel", "preprocessed/pc", category)