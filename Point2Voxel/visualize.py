import numpy as np
import os
import torch
import h5py
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image
import argparse


def visualize(data, save):
    # Voxel grids
    if 128 in data.shape:
        if len(data.shape) > 3:
            data = data[0]

        # Get the coordinates of occupied voxels
        print(f"Min/max/mean: {data.min():.4f} / {data.max():.4f} / {data.mean():.4f}")
        data = np.argwhere(data > 0.5)  # data == 1

    # Point clouds
    else:
        print(f"Min/max/mean: {data.min():.4f} / {data.max():.4f} / {data.mean():.4f}")
        if len(data.shape) > 2:
            data = data[0]

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    plt.tight_layout()

    ax = fig.add_subplot(111, projection='3d')

    # Plot occupied voxels as scatter points
    ax.scatter(data[:, 0], data[:, 2], data[:, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Move the buffer cursor to the beginning

    plt.savefig(save)
    print(f"Save path: {save}")
    plt.close()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='./output/samples/sample_epoch_100_voxel.npy')
    parser.add_argument('--save_path', '-o', default=None)
    parser.add_argument('--save_dir', '-d', default=None)
    opt = parser.parse_args()

    # Load 3D tensor data
    data = np.load(opt.input)
    shape = data.shape
    print(f"Input tensor shape: {data.shape}")

    # If save path is specified
    if opt.save_path is not None:
        save = (opt.save_path + ".png").replace(".png.png", ".png")

    # Generate save path otherwise
    else:
        if 128 in shape:
            # Voxel grids
            save = opt.input.split("/")[-1].replace('.npy', '_voxel.png').replace('_voxel_voxel', '_voxel')
            print("Input type: Voxel grids")
        else:
            # Point clouds
            save = opt.input.split("/")[-1].replace('.npy', '_pc.png').replace('_pc_pc', '_pc')
            print("Input type: Point clouds")

    # If save directory is specified
    if opt.save_dir is not None:
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        save = os.path.join(opt.save_dir, save)

    visualize(data, save)