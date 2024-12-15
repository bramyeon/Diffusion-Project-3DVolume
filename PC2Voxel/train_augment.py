'''
This script was developed by Bryan Nathanael Wijaya.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from tqdm import tqdm
from visualize import visualize

from train import ShapeNetDataset, PointCloudToVoxel

# Dataset class
class ShapeNetAugmentDataset(Dataset):
    def __init__(self, voxel_dir, pc_dir, category, augment_dir, vox_res=(64, 64, 64)):
        self.voxel_paths = []
        self.pc_paths = []
        self.augment_paths = []
        
        voxel_category_dir = os.path.join(voxel_dir, category)
        pc_category_dir = os.path.join(pc_dir, category)
        files = os.listdir(voxel_category_dir)
        self.voxel_paths += [os.path.join(voxel_category_dir, f) for f in files]
        self.pc_paths += [os.path.join(pc_category_dir, f) for f in files]
        self.ori_len = len(self.voxel_paths)
        
        augment_files = os.listdir(augment_dir)
        self.augment_paths += [os.path.join(augment_dir, f) for f in augment_files if ".npy" in f]
        self.augment_pc = [np.load(path) for path in self.augment_paths]
        self.augment_pc = np.concatenate(self.augment_pc, axis=0)
        self.augment_len = self.augment_pc.shape[0]
        
        self.vox_res = vox_res
        
    def __len__(self):
        return self.ori_len + self.augment_len

    # copied from load_data.py in the original project repository
    def voxelize(self, pts):
        """
        pts: np.ndarray [N,3]
        """
        vox_res = np.array(self.vox_res)
        min_bounds = np.min(pts, axis=0)
        max_bounds = np.max(pts, axis=0)

        normalized_pts = (pts - min_bounds) / (max_bounds - min_bounds)
        scaled_pts = normalized_pts * (vox_res - 1)

        voxel_grid = np.zeros(vox_res, dtype=np.float32)
        voxel_indices = np.floor(scaled_pts).astype(np.int32)

        for idx in voxel_indices:
            voxel_grid[tuple(idx)] = 1
        return voxel_grid

    def __getitem__(self, idx):
        # original data
        if idx < self.ori_len:
            voxel = np.load(self.voxel_paths[idx])
            pc = np.load(self.pc_paths[idx])
        
        # augmented data (synthetic)
        else:
            idx_ = idx - self.ori_len
            pc = self.augment_pc[idx_]
            voxel = self.voxelize(pc)
            
        return torch.tensor(voxel, dtype=torch.float32), torch.tensor(pc, dtype=torch.float32)
    
    
def train_model(model, train_loader, val_loader, device, epochs, lr, save_dir):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    eval_save_dir = os.path.join(save_dir, "samples")
    os.makedirs(eval_save_dir, exist_ok=True)
    
    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0
        for voxel_grids, point_clouds in train_loader:
            voxel_grids, point_clouds = voxel_grids.to(device), point_clouds.to(device)
            optimizer.zero_grad()
            predictions = model(point_clouds)
            loss = criterion(predictions, voxel_grids)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        if (epoch+1) % 20 == 0 or (epoch+1) == epochs:
            if (epoch+1) == epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, f'pc2voxel_aug_last.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, f'pc2voxel_aug_epoch_{epoch+1}.pth'))
 
            model.eval()
            val_loss = 0
            # samples = []
            with torch.no_grad():
                for voxel_grids, point_clouds in val_loader:
                    voxel_grids, point_clouds = voxel_grids.to(device), point_clouds.to(device)
                    predictions = model(point_clouds)
                    # samples.append(predictions.cpu().numpy())
                    samples = predictions.cpu().numpy()
                    loss = criterion(predictions, voxel_grids)
                    val_loss += loss.item()
            
            print(f"\tEpoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")
            # samples = np.stack(samples, axis=0)
            if samples.shape[0] == 1:
                samples = samples[0]
            samples = samples[:16]
            if (epoch+1) == epochs:
                np.save(os.path.join(eval_save_dir, f"sample_last.npy"), samples)
                visualize(samples, os.path.join(eval_save_dir, f"sample_last.png"), warn=False)
            else:
                np.save(os.path.join(eval_save_dir, f"sample_epoch_{epoch+1}.npy"), samples)
                # visualize(samples, os.path.join(eval_save_dir, f"sample_epoch_{epoch+1}.png"), warn=False)
                
        else:
            print(f"\tEpoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           default='./output-augment')
    parser.add_argument('--voxel_dir',          default='../data/preprocessed/voxel')
    parser.add_argument('--pc_dir',             default='../data/preprocessed/pc')
    parser.add_argument('--augment_dir',        default=None, required=True)
    parser.add_argument('--category',           default=None)
    parser.add_argument('--bs',     type=int,   default=8,      help='input batch size')
    parser.add_argument('--lr',     type=float, default=1e-4,   help='learning rate')
    parser.add_argument('--niter',  type=int,   default=300,    help='number of epochs to train for')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # set training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set category
    assert opt.category in ['chair', 'airplane', 'table']
    opt.save_dir = os.path.join(opt.save_dir, opt.category)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # print settings
    print(f"Save directory: {opt.save_dir}")
    print(f"Voxel directory: {opt.voxel_dir}")
    print(f"PC directory: {opt.pc_dir}")
    print(f"Category: {opt.category}")
    print(f"Batch size: {opt.bs}")
    print(f"Learning rate: {opt.lr}")
    print(f"Epochs: {opt.niter}")
    print(f"Device: {device}")
    print()

    # DataLoader
    print("Loading training dataset...")
    train_dataset = ShapeNetAugmentDataset(os.path.join(opt.voxel_dir, "train"), os.path.join(opt.pc_dir, "train"), opt.category, opt.augment_dir)
    train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True)
    
    print("\nLoading evaluation dataset...")
    val_dataset = ShapeNetDataset(os.path.join(opt.voxel_dir, "val"), os.path.join(opt.pc_dir, "val"), opt.category)
    val_loader = DataLoader(val_dataset, batch_size=opt.bs, shuffle=True)

    # model
    model = PointCloudToVoxel().to(device)

    # train
    print("\nTraining...")
    train_model(model, train_loader, val_loader, device, opt.niter, opt.lr, opt.save_dir)