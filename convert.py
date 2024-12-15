'''
This script was developed by Bryan Nathanael Wijaya.
'''

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from tqdm import tqdm

from PC2Voxel.visualize import visualize
from PC2Voxel.train import PointCloudToVoxel

class ShapeNetPCDataset(Dataset):
    def __init__(self, pc_path):
        self.pc_path = pc_path
        self.pc = torch.tensor(np.load(self.pc_path), dtype=torch.float32)
        
    def __len__(self):
        return self.pc.shape[0]

    def __getitem__(self, idx):
        return self.pc[idx]

def pc2voxel(model, sample_loader, device, category, save_dir, nsamp=1000):
    eval_save_dir = os.path.join(save_dir, "voxel_samples")
    os.makedirs(eval_save_dir, exist_ok=True)
    
    # hyperparameters
    if category == 'airplane':
        threshold = 0.02
        color = '#167ea2'
    elif category == 'chair':
        threshold = 0.01
        color = '#bb729f'
    elif category == 'table':
        threshold = 0.02
        color = '#559281'
    
    model.eval()
    samples = []
    with torch.no_grad():
        # convert point clouds
        for point_clouds in tqdm(sample_loader):
            point_clouds = point_clouds.to(device)
            predictions = model(point_clouds)
            samples.append(predictions.cpu().numpy())
    
    # save as npy file with shape (nsamp, ...)
    samples = np.concatenate(samples, axis=0)
    samples = samples[:nsamp]
    print(f"Sample shape: {samples.shape}")
    np.save(os.path.join(eval_save_dir, "samples_voxel_raw.npy"), samples)
    
    binary = (samples >= threshold)
    np.save(os.path.join(eval_save_dir, f"samples_voxel_{threshold}.npy"), binary)

    # visualize samples
    for i in range(2): # 10
        try:
            print(f"\nVisualizing ({i+1}/10)...")
            visualize(binary[i * 16:(i+1) * 16], os.path.join(eval_save_dir, f"samples_voxel_{threshold}_{i+1}.png"), color=color)
        except:
            print(f"\nAll data has been visualized!")
            
if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           default='./output')
    parser.add_argument('--category',           default='chair')
    parser.add_argument('--model_path',         default='./PC2Voxel/output/chair/pc2voxel_last.pth')
    parser.add_argument('--pc_path',            default='./PVD/output/sample/2024-11-21-13-39-23/syn/samples.npy')
    parser.add_argument('--nsamp',  type=int,   default=1000,   help='number of samples')
    parser.add_argument('--bs',     type=int,   default=8,      help='input batch size')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)    
    
    # set inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set category
    assert opt.category in ['chair', 'airplane', 'table']
    opt.save_dir = os.path.join(opt.save_dir, opt.category)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # print settings
    print(f"Save directory: {opt.save_dir}")
    print(f"Category: {opt.category}")
    print(f"Model path: {opt.model_path}")
    print(f"PC path: {opt.pc_path}")
    print(f"Num samples: {opt.nsamp}")
    print(f"Batch size: {opt.bs}")
    print(f"Device: {device}")
    print()

    print("Loading point cloud samples...")
    sample_dataset = ShapeNetPCDataset(opt.pc_path)
    sample_loader = DataLoader(sample_dataset, batch_size=opt.bs, shuffle=True)

    # model
    model = PointCloudToVoxel().to(device)
    weights = torch.load(opt.model_path)
    model.load_state_dict(weights)

    # train
    print("\nConverting PC samples to voxels...")
    pc2voxel(model, sample_loader, device, opt.category, opt.save_dir, opt.nsamp)