import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Input data : (2658, 128, 128, 128)
input_data = torch.from_numpy(np.load('./data/hdf5_data/table_voxels_train.npy')).float()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        # Input data : (batch_size, 128, 128, 128)
        x = x.unsqueeze(1)  # (batch_size, 1, 128, 128, 128)
        x = self.conv1(x)  # (batch_size, 1, 32, 32, 32)
        x = torch.sigmoid(x)  # constraints to 0 or 1
        x = (x > 0.5).float()  # threshold : 0.5
        return x.squeeze(1)  # (batch_size, 32, 32, 32)

# Encoder
model = Encoder()
downsampled_data = model(input_data)

# print(downsampled_data.shape)  # torch.Size([2658, 32, 32, 32])
# print(downsampled_data.min(), downsampled_data.max(), downsampled_data.mean())

downsampled_data_np = downsampled_data.numpy()

# save to numpy file
np.save('./data/sampled_voxel/table_voxels_train32.npy', downsampled_data_np)
print("Completed!")