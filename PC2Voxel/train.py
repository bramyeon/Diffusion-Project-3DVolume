'''
This script was developed by Bryan Nathanael Wijaya.
AI Tool Used: ChatGPT (only for initial decoder design; much improvements were added by the author, especially on the network design)

While using ChatGPT, the following works were mentioned as references to the initial decoder design:
1. PointNet++: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (2017)
2. 3D-GAN: Wu et al., "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling" (2016)
3. AtlasNet: Groueix et al., "AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation" (2018)
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

# Dataset class
class ShapeNetDataset(Dataset):
    def __init__(self, voxel_dir, pc_dir, category):
        self.voxel_paths = []
        self.pc_paths = []
        
        if category is None:
            categories = os.listdir(voxel_dir)
            for category in tqdm(categories):
                voxel_category_dir = os.path.join(voxel_dir, category)
                pc_category_dir = os.path.join(pc_dir, category)
                files = os.listdir(voxel_category_dir)
                self.voxel_paths += [os.path.join(voxel_category_dir, f) for f in files]
                self.pc_paths += [os.path.join(pc_category_dir, f) for f in files]

        else:
            voxel_category_dir = os.path.join(voxel_dir, category)
            pc_category_dir = os.path.join(pc_dir, category)
            files = os.listdir(voxel_category_dir)
            self.voxel_paths += [os.path.join(voxel_category_dir, f) for f in files]
            self.pc_paths += [os.path.join(pc_category_dir, f) for f in files]
        
    def __len__(self):
        return len(self.voxel_paths)

    def __getitem__(self, idx):
        voxel = np.load(self.voxel_paths[idx])
        pc = np.load(self.pc_paths[idx])
        return torch.tensor(voxel, dtype=torch.float32), torch.tensor(pc, dtype=torch.float32)

# Spatial attention for encoder
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 2, 1)    # (batch_size, in_channel, 2048) --> (batch_size, in_channel//2, 2048)
        self.conv2 = nn.Conv1d(in_channels // 2, 1, 1)              # (batch_size, in_channel//2, 2048) --> (batch_size, 1, 2048)
        
    def forward(self, x):
        attention = F.relu(self.conv1(x))                   # (batch_size, in_channel//2, 2048)
        attention = torch.sigmoid(self.conv2(attention))    # (batch_size, 1, 2048)
        return x * attention                                # (batch_size, in_channel, 2048)
    
# Positional embeddings for encoder
class PositionalEncoding(nn.Module):
    def __init__(self, max_freq_log2=10, num_bands=6):
        super(PositionalEncoding, self).__init__()
        self.num_bands = num_bands
        self.max_freq = 2 ** max_freq_log2
    
    def forward(self, coords):
        # coords: Shape (batch, 2048, 3)
        freqs = torch.linspace(1.0, self.max_freq, self.num_bands, device=coords.device)    # (6,)
        angles = coords[..., None] * freqs                                                  # (batch_size, 2048, 3, 6)
        positional_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)     # (batch_size, 2048, 3, 12)
        return positional_features.flatten(-2)                                              # (batch_size, 2048, 36)
    
# Encode point cloud to some latent representation
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat_dim=1024, num_bands=6):
        super(PointNetEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(num_bands=num_bands)  # (batch_size, 2048, 3) --> (batch_size, 2048, 36)
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3 + 3 * 2 * num_bands, 64, 1), nn.ReLU(),     # (batch_size, 39, 2048) --> (batch_size, 64, 2048)
            nn.Conv1d(64, 128, 1), nn.ReLU(), nn.Dropout(p=0.3)     # (batch_size, 64, 2048) --> (batch_size, 128, 2048)
        )
        self.spatial_attention1 = SpatialAttention(128)
        self.conv_skip1 = nn.Conv1d(39, 128, 1)                     # (batch_size, 39, 2048) --> (batch_size, 128, 2048)
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1), nn.ReLU(),                      # (batch_size, 128, 2048) --> (batch_size, 256, 2048)
            nn.Conv1d(256, global_feat_dim, 1)                      # (batch_size, 256, 2048) --> (batch_size, 1024, 2048)
        )
        self.spatial_attention2 = SpatialAttention(global_feat_dim)
        self.conv_skip2 = nn.Conv1d(128, global_feat_dim, 1)        # (batch_size, 128, 2048) --> (batch_size, 1024, 2048)
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)                  # (batch_size, 1024, 2048) --> (batch_size, 1024, 1)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        positional_features = self.positional_encoding(x)   # (batch_size, 2048, 36)
        x = torch.cat([x, positional_features], dim=-1)     # (batch_size, 2048, 39)
        x = x.transpose(1, 2)                               # (batch_size, 39, 2048)
         
        intermediate_feat = self.mlp1(x)                                # (batch_size, 128, 2048)
        intermediate_feat = self.spatial_attention1(intermediate_feat)  # (batch_size, 128, 2048)
        intermediate_feat_skip = self.conv_skip1(x)                     # (batch_size, 128, 2048)
        intermediate_feat = intermediate_feat + intermediate_feat_skip  # (batch_size, 128, 2048)
        
        global_feat = self.mlp2(intermediate_feat)              # (batch_size, 1024, 2048)
        global_feat = self.spatial_attention2(global_feat)      # (batch_size, 1024, 2048)
        global_feat_skip = self.conv_skip2(intermediate_feat)   # (batch_size, 1024, 2048)
        global_feat = global_feat + global_feat_skip            # (batch_size, 1024, 2048)
        
        global_feat = self.global_pool(global_feat).squeeze(-1) # (batch_size, 1024)
        global_feat = self.dropout(global_feat)                 # (batch_size, 1024)
        return global_feat 

# Decode latent representation to voxel grids
class VoxelDecoder(nn.Module):
    def __init__(self, input_dim=1024, grid_size=64):
        super(VoxelDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.ReLU(),                      # (batch_size, 1024) --> (batch_size, 2048)
            nn.Linear(2048, 4 * 4 * 4), nn.ReLU(), nn.Dropout(p=0.3)    # (batch_size, 2048) --> (batch_size, 64)
        )
        self.fc_skip = nn.Linear(input_dim, 4 * 4 * 4)                  # (batch_size, 1024) --> (batch_size, 64)
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),                       # (batch_size, 1, 4, 4, 4) --> (batch_size, 32, 8, 8, 8)
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(p=0.3))   # (batch_size, 32, 8, 8, 8) --> (batch_size, 16, 16, 16, 16)
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # (batch_size, 16, 16, 16, 16) --> (batch_size, 8, 32, 32, 32)
            nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1)                # (batch_size, 8, 32, 32, 32) --> (batch_size, 1, 64, 64, 64)
        )
        self.deconv_skip = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1), nn.ReLU()  # (batch_size, 16, 64, 64, 64) --> (batch_size, 1, 64, 64, 64)
        )
        self.sigmoid = nn.Sigmoid()     # map values to between 0 and 1
        self.grid_size = grid_size
    
    def forward(self, x):
        x_fc = self.fc(x)                   # (batch_size, 64)
        x_fc_skip = self.fc_skip(x)         # (batch_size, 64)
        x_fc = x_fc + x_fc_skip             # (batch_size, 64)
        x_fc = x_fc.view(-1, 1, 4, 4, 4)    # (batch_size, 1, 4, 4, 4)
        
        x_deconv1 = self.deconv1(x_fc)                                                                  # (batch_size, 16, 16, 16, 16)
        x_deconv1_skip = F.interpolate(x_fc, size=(16, 16, 16), mode='trilinear', align_corners=False)  # (batch_size, 1, 16, 16, 16)
        x_deconv1 = x_deconv1 + x_deconv1_skip                                                          # (batch_size, 16, 16, 16, 16)
        
        x_deconv2 = self.deconv2(x_deconv1)                                                                                     # (batch_size, 1, 64, 64, 64)
        x_deconv2_skip = self.deconv_skip(F.interpolate(x_deconv1, size=(64, 64, 64), mode='trilinear', align_corners=False))   # (batch_size, 1, 64, 64, 64)
        x_deconv2 = x_deconv2 + x_deconv2_skip                                                                                  # (batch_size, 1, 64, 64, 64)
        
        x = self.sigmoid(x_deconv2)         # (batch_size, 1, 64, 64, 64)
        return x

# Convert point clouds to voxel grids
class PointCloudToVoxel(nn.Module):
    def __init__(self):
        super(PointCloudToVoxel, self).__init__()
        self.encoder = PointNetEncoder()    # (batch_size, 2048, 3) --> (batch_size, 1024)
        self.decoder = VoxelDecoder()       # (batch_size, 1024) --> (batch_size, 1, 64, 64, 64)
    
    def forward(self, x):
        latent = self.encoder(x)                # (batch_size, 1024)
        voxels = self.decoder(latent)           # (batch_size, 1, 64, 64, 64)
        return voxels.reshape(-1, 64, 64, 64)   # (batch_size, 64, 64, 64)
    
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
                torch.save(model.state_dict(), os.path.join(save_dir, f'pc2voxel_last.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, f'pc2voxel_epoch_{epoch+1}.pth'))
 
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
    parser.add_argument('--save_dir',           default='./output')
    parser.add_argument('--voxel_dir',          default='../data/preprocessed/voxel')
    parser.add_argument('--pc_dir',             default='../data/preprocessed/pc')
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
    assert opt.category in [None, 'chair', 'airplane', 'table']
    if opt.category is not None:
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
    train_dataset = ShapeNetDataset(os.path.join(opt.voxel_dir, "train"), os.path.join(opt.pc_dir, "train"), opt.category)
    train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True)
    
    print("\nLoading evaluation dataset...")
    val_dataset = ShapeNetDataset(os.path.join(opt.voxel_dir, "val"), os.path.join(opt.pc_dir, "val"), opt.category)
    val_loader = DataLoader(val_dataset, batch_size=opt.bs, shuffle=True)

    # model
    model = PointCloudToVoxel().to(device)

    # train
    print("\nTraining...")
    train_model(model, train_loader, val_loader, device, opt.niter, opt.lr, opt.save_dir)