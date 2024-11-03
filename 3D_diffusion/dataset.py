import os
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path

import torch
import numpy as np ## Project ##


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class ShapeNetVoxelDataset(torch.utils.data.Dataset):
    def __init__(
            self, root: str, split: str, max_num_voxels_per_cat=-1, label_offset=1
    ):
        super().__init__()
        self.root = root
        self.split = split
        # self.transform = transform
        self.max_num_voxels_per_cat = max_num_voxels_per_cat
        self.label_offset = label_offset

        ############ Project ############
        # categories = ["airplane", "chair", "table"]
        categories = ["airplane"] # Memory test
        self.num_classes = len(categories)

        print(f"Start loading {split} Data....")
        voxels, labels = [], []
        for idx, cat in enumerate(sorted(categories)):
            cat_voxel = np.load(os.path.join(self.root, f"{cat}_voxels_{self.split}.npy"))
            if self.max_num_voxels_per_cat > 0:
                # cat_voxel = cat_voxel[: self.max_num_voxels_per_cat]
                cat_voxel = cat_voxel[: self.max_num_voxels_per_cat, 48:80, 48:80, 48:80] # For mini size test (B, 32, 32, 32)
            cat_voxel = (cat_voxel - 0.5) / 0.5  # Normalize
            cat_voxel = torch.Tensor(cat_voxel)

            voxels.append(cat_voxel)
            labels += [idx + label_offset] * len(cat_voxel) # label 0 is for null class.

        self.voxels = torch.cat(voxels)
        # self.voxels = np.concatenate(voxels)
        self.labels = labels
        # print(self.voxels.shape)
        # print(len(self.labels))
        print(f"======Complete loading {split} Data======")
        #################################

    def __getitem__(self, idx):
        ####### Project #######
        voxel = self.voxels[idx]
        label = self.labels[idx]
        #######################
        return voxel, label

    def __len__(self):
        return len(self.labels)


class ShapeNetVoxelDataModule(object):
    def __init__(
            self,
            root: str = "../data",  # Project
            batch_size: int = 32,
            num_workers: int = 4,
            max_num_voxels_per_cat: int = 1000,  # Project
            voxel_resolution: int = 128,  # Project
            label_offset=1,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shapenet_root = os.path.join(root, "hdf5_data")  # Project
        self.max_num_voxels_per_cat = max_num_voxels_per_cat  # Project
        self.voxel_resolution = voxel_resolution  # Project
        self.label_offset = label_offset

        if not os.path.exists(self.shapenet_root):
            print(f"{self.shapenet_root} is empty. Downloading Shapenet dataset...")

        self._set_dataset()

    def _set_dataset(self):
        self.train_ds = ShapeNetVoxelDataset(
            self.shapenet_root,
            "train",
            max_num_voxels_per_cat=self.max_num_voxels_per_cat,
            label_offset=self.label_offset
        )
        self.val_ds = ShapeNetVoxelDataset(
            self.shapenet_root,
            "val",
            max_num_voxels_per_cat=self.max_num_voxels_per_cat,
            label_offset=self.label_offset,
        )
        self.num_classes = self.train_ds.num_classes


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )


if __name__ == "__main__":
    # Below code works well  when loading only one category data
    # However, this occurs error when loading all categories data...(nothing to do with training, just need to test only this file. I'm solving..)
    voxel_resolution = 128
    ds_module = ShapeNetVoxelDataModule(
        "../data",
        batch_size=16,
        num_workers=4,
        max_num_voxels_per_cat=2000,
        voxel_resolution=voxel_resolution
    )
    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    voxel, label = next(train_it)
    print(voxel.shape)
    print(label)
