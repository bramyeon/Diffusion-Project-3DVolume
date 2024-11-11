import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

data_root ="./data"
shapenet_root = os.path.join(data_root, "./")


def split(root: str, dataset: str, category: list):

    save_dir = Path(os.path.join(root, "voxel_data"))
    save_dir.mkdir(exist_ok=True, parents=True)

    dataset_root = os.path.join(root, dataset)

    for split in ["train", "val"]:  # Remove test - No need test dataset for training
        save_split_dir = Path(os.path.join(save_dir, split))
        save_split_dir.mkdir(exist_ok=True, parents=True)
        for cat in tqdm(category):
            save_cat_dir = Path(os.path.join(save_split_dir, cat))
            save_cat_dir.mkdir(exist_ok=True, parents=True)

            data = np.load(os.path.join(dataset_root, f"{cat}_voxels_{split}.npy"))
            # os.remove(os.path.join(dataset_root, f"{cat}_voxels_{split}.npy"))  # For memory
            for i in range(data.shape[0]):
                np.save(os.path.join(save_cat_dir, f"{cat}_{i}.npy"), data[i])



if __name__ == "__main__":
    category = ["airplane", "chair", "table"]
    split("./data", "hdf5_data", category)