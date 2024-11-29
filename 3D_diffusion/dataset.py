import os
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path

import torch
import numpy as np ## Project ##

def listdir(dname):
    fnames = list(
        chain(
            *[
                list(Path(dname).rglob("*.npy"))  # Project
            ]
        )
    )
    return fnames

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

        categories = os.listdir(os.path.join(root, split))
        self.num_classes = len(categories)

        file_names, labels = [], []

        # if only train one category  airplane = 0, chair = 1, table = 2
        category_idx = 2
        cat = sorted(categories)[category_idx]
        category_dir = os.path.join(root, split, cat)
        cat_file_names = listdir(category_dir)
        cat_file_names = sorted(cat_file_names)
        if self.max_num_voxels_per_cat > 0:
            cat_file_names = cat_file_names[: self.max_num_voxels_per_cat]
        file_names += cat_file_names
        labels += [category_idx + label_offset] * len(cat_file_names)  # label 0 is for null class.

        # if train all categories
        # for idx, cat in enumerate(sorted(categories)):
        #     category_dir = os.path.join(root, split, cat)
        #     cat_file_names = listdir(category_dir)
        #     cat_file_names = sorted(cat_file_names)
        #     if self.max_num_voxels_per_cat > 0:
        #         cat_file_names = cat_file_names[: self.max_num_voxels_per_cat]
        #     file_names += cat_file_names
        #     labels += [idx + label_offset] * len(cat_file_names)  # label 0 is for null class.

        self.file_names = file_names
        self.labels = labels


    def __getitem__(self, idx):
        ####### Project #######
        voxel = np.load(self.file_names[idx])
        # voxel = (voxel-0.5) / 0.5  # Normalize
        label = self.labels[idx]
        assert label >= self.label_offset
        #######################
        return voxel, label

    def __len__(self):
        return len(self.labels)
    
class ShapeNetAugmentedVoxelDataset(torch.utils.data.Dataset):
    def __init__(
            self, root: str, split: str, max_num_voxels_per_cat=-1, label_offset=1, augment_dir: str=None
    ):
        super().__init__()
        self.root = root
        self.split = split
        # self.transform = transform
        self.max_num_voxels_per_cat = max_num_voxels_per_cat
        self.label_offset = label_offset
        self.augment_dir = augment_dir

        categories = os.listdir(os.path.join(root, split))
        self.num_classes = len(categories)

        file_names, labels = [], []

        # if only train one category  airplane = 0, chair = 1, table = 2
        category_idx = 2
        cat = sorted(categories)[category_idx]
        category_dir = os.path.join(root, split, cat)
        cat_file_names = listdir(category_dir)
        cat_file_names = sorted(cat_file_names)
        if self.max_num_voxels_per_cat > 0:
            cat_file_names = cat_file_names[: self.max_num_voxels_per_cat]
        file_names += cat_file_names
        labels += [category_idx + label_offset] * len(cat_file_names)  # label 0 is for null class.

        # if train all categories
        # for idx, cat in enumerate(sorted(categories)):
        #     category_dir = os.path.join(root, split, cat)
        #     cat_file_names = listdir(category_dir)
        #     cat_file_names = sorted(cat_file_names)
        #     if self.max_num_voxels_per_cat > 0:
        #         cat_file_names = cat_file_names[: self.max_num_voxels_per_cat]
        #     file_names += cat_file_names
        #     labels += [idx + label_offset] * len(cat_file_names)  # label 0 is for null class.

        self.file_names = file_names
        self.labels = labels
        self.ori_len = len(self.labels)
        
        self.syn_data = np.load(os.path.join(self.augment_dir, "???.npy"))
        self.syn_len = self.syn_data.shape[0]


    def __getitem__(self, idx):
        ####### Project #######
        if idx < self.ori_len:
            voxel = np.load(self.file_names[idx])
            # voxel = (voxel-0.5) / 0.5  # Normalize
            label = self.labels[idx]
        else:
            syn_idx = idx - self.ori_len
            voxel = self.syn_data[syn_idx]
            label = self.labels[syn_idx] # anyway, all belong to the same category in our project
        assert label >= self.label_offset
        #######################
        return voxel, label

    def __len__(self):
        return self.ori_len + self.syn_len #len(self.labels)


class ShapeNetVoxelDataModule(object):
    def __init__(
            self,
            root: str = "../data",  # Project
            batch_size: int = 32,
            num_workers: int = 4,
            max_num_voxels_per_cat: int = 1000,  # Project
            voxel_resolution: int = 128,  # Project
            label_offset=1,
            augment_dir: str = None
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shapenet_root = os.path.join(root, "voxel_data")  # Project
        self.max_num_voxels_per_cat = max_num_voxels_per_cat  # Project
        self.voxel_resolution = voxel_resolution  # Project
        self.label_offset = label_offset
        self.augment_dir = augment_dir

        if not os.path.exists(self.shapenet_root):
            print(f"{self.shapenet_root} is empty. Make voxel dataset...")

        self._set_dataset()

    def _set_dataset(self):
        self.train_ds = ShapeNetAugmentedVoxelDataset(
            self.shapenet_root,
            "train",
            max_num_voxels_per_cat=self.max_num_voxels_per_cat,
            label_offset=self.label_offset, augment_dir=self.augment_dir
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
    # train_dl = ds_module.train_dataloader()
    train_dl = ds_module.val_dataloader()
    train_it = get_data_iterator(train_dl)

    voxel, label = next(train_it)
    print(voxel.shape)
    print(label)
