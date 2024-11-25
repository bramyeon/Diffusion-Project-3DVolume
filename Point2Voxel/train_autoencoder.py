import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import init


# Dataset class
class ShapeNetDataset(Dataset):
    def __init__(self, voxel_dir, pc_dir):
        self.voxel_paths = []
        self.pc_paths = []
        categories = os.listdir(voxel_dir)
        for category in tqdm(categories):
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
        # Normalize point cloud
        # min_bounds = np.min(pc, axis=0)
        # max_bounds = np.max(pc, axis=0)
        # vox_res = 128
        #
        # normalized_pts = (pc - min_bounds) / (max_bounds - min_bounds)
        # pc = normalized_pts * (vox_res - 1)

        return torch.tensor(voxel, dtype=torch.float32), torch.tensor(pc, dtype=torch.float32)


# Encoder: (128, 128, 128) voxel grids -> (2048, 3) point clouds
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16 * 16, latent_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(latent_dim, 2048 * 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.fc(x)
        return x.view(-1, 2048, 3)


# Mimicking nn.Unflatten because we use an older PyTorch version to accommodate PyTorchEMD for PVD
class CustomUnflatten(nn.Module):
    def __init__(self, dim, unflattened_size):
        """
        Args:
            dim (int): The dimension to unflatten.
            unflattened_size (tuple): The new shape for the unflattened dimension.
        """
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, x):
        return x.view(*x.shape[:self.dim], *self.unflattened_size, *x.shape[self.dim + 1:])


# Decoder: (2048, 3) point clouds -> (128, 128, 128) voxel grids
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048 * 3, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16 * 16),
            nn.ReLU(),
            CustomUnflatten(1, (64, 16, 16, 16)),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.decoder[-2].weight, gain=1e-5)
        init.kaiming_uniform_(self.decoder[-4].weight, nonlinearity='relu')
        init.kaiming_uniform_(self.decoder[-6].weight, nonlinearity='relu')
        init.kaiming_uniform_(self.decoder[0].weight, nonlinearity='relu')
        # init.zeros_(self.decoder[-2].bias)
        # init.zeros_(self.decoder[-4].bias)
        # init.zeros_(self.decoder[-6].bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.decoder(x)
        return x.squeeze(1)


# autoencoder consists of the encoder and decoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

    def forward(self, x):
        point_cloud = self.encoder(x)
        reconstructed_voxel = self.decoder(point_cloud)
        return reconstructed_voxel, point_cloud


# save evaluation samples
def save_samples(encoder, decoder, val_loader, device, save_dir, epoch, num_samples=10):
    # encoder['model'].eval()
    decoder['model'].eval()

    concatenated_pcs = []
    concatenated_voxels = []
    samples_saved = 0

    with torch.no_grad():
        for _, pc in val_loader:
            if samples_saved >= num_samples:
                break

            # voxel = voxel.to(device)
            pc = pc.to(device)
            # encoded_pc = encoder['model'](voxel)
            reconstructed_voxel = decoder['model'](pc)

            for i in range(pc.size(0)):
                if samples_saved >= num_samples:
                    break

                # concatenated_pcs.append(encoded_pc[i].cpu().numpy())
                concatenated_voxels.append(reconstructed_voxel[i].cpu().numpy())
                samples_saved += 1

    # save concatenated npy
    # concatenated_pcs = np.stack(concatenated_pcs, axis=0)
    concatenated_voxels = np.stack(concatenated_voxels, axis=0)
    # np.save(os.path.join(save_dir, f"sample_epoch_{epoch}_pc.npy"), concatenated_pcs)
    np.save(os.path.join(save_dir, f"sample_epoch_{epoch}_voxel.npy"), concatenated_voxels)


# training script
def train_encdec(train_loader, val_loader, encoder, decoder, device, save_dir, epochs=200, num_samples=10):
    eval_save_dir = os.path.join(save_dir, "samples")
    os.makedirs(eval_save_dir, exist_ok=True)

    # encoder['model'].to(device)
    decoder['model'].to(device)

    # encoder['model'].train()
    decoder['model'].train()

    for epoch in tqdm(range(epochs), desc="train"):
        # encoder['total_loss'] = 0
        decoder['total_loss'] = 0
        for voxel, point_cloud in train_loader:
            voxel, point_cloud = voxel.to(device), point_cloud.to(device)

            # encoder['optimizer'].zero_grad()
            # predicted_pc = encoder['model'](voxel)
            # encoder_loss = encoder['criterion'](predicted_pc, point_cloud)
            # encoder_loss.backward()
            # encoder['optimizer'].step()
            # encoder['total_loss'] += encoder_loss.item()
            decoder['optimizer'].zero_grad()
            reconstructed_voxel = decoder['model'](point_cloud)
            decoder_loss = decoder['criterion'](reconstructed_voxel, voxel)
            decoder_loss.backward()
            decoder['optimizer'].step()
            decoder['total_loss'] += decoder_loss.item()

        print(
            # f"Epoch [{epoch}/{epochs}], Encoder Loss: {encoder['total_loss'] / len(train_loader):.4f}, Decoder Loss: {decoder['total_loss'] / len(train_loader):.4f}")
            f"Epoch [{epoch}/{epochs}], Decoder Loss: {decoder['total_loss'] / len(train_loader):.4f}")
        if (epoch) % 20 == 0:
            # torch.save(encoder['model'].state_dict(), os.path.join(save_dir, f'encoder_epoch_{epoch}.pth'))
            torch.save(decoder['model'].state_dict(), os.path.join(save_dir, f'decoder_epoch_{epoch}.pth'))
            save_samples(encoder, decoder, val_loader, device, eval_save_dir, epoch=epoch, num_samples=num_samples)

    # torch.save(encoder['model'].state_dict(), os.path.join(save_dir, 'encoder_last.pth'))
    torch.save(decoder['model'].state_dict(), os.path.join(save_dir, 'decoder_last.pth'))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--voxel_dir', default='./data/preprocessed/voxel')
    parser.add_argument('--pc_dir', default='./data/preprocessed/pc')
    parser.add_argument('--bs', type=int, default=128, help='input batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0001')
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--nsamp', type=int, default=10, help='number of samples for evaluation')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # set training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Save directory: {opt.save_dir}")
    print(f"Voxel directory: {opt.voxel_dir}")
    print(f"PC directory: {opt.pc_dir}")
    print(f"Batch size: {opt.bs}")
    print(f"Learning rate: {opt.lr}")
    print(f"Epochs: {opt.niter}")
    print(f"Num samples: {opt.nsamp}")
    print(f"Device: {device}")
    print()

    # DataLoader
    print("Loading training dataset...")
    train_dataset = ShapeNetDataset(os.path.join(opt.voxel_dir, "train"), os.path.join(opt.pc_dir, "train"))
    train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True)

    print("\nLoading evaluation dataset...")
    val_dataset = ShapeNetDataset(os.path.join(opt.voxel_dir, "val"), os.path.join(opt.pc_dir, "val"))
    val_loader = DataLoader(val_dataset, batch_size=opt.bs, shuffle=True)

    # model, loss, and optimizer
    encoder = {
        'model': Encoder(),
        'criterion': nn.MSELoss(),
    }
    encoder['optimizer'] = optim.Adam(encoder['model'].parameters(), lr=opt.lr)

    decoder = {
        'model': Decoder(),
        'criterion': nn.functional.binary_cross_entropy,
    }
    decoder['optimizer'] = optim.Adam(decoder['model'].parameters(), lr=opt.lr)

    # train
    print("\nTraining...")
    train_encdec(train_loader, val_loader, encoder, decoder, device, opt.save_dir, epochs=opt.niter,
                 num_samples=opt.nsamp)