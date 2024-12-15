'''
This script was developed by Bryan Nathanael Wijaya.
AI Tool Used: ChatGPT (only for initial implementation)
Also insipired by the visualize.ipynb file of the project's main repository.
'''

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def visualize(data, save, threshold = 0.5, warn=True, color=None):

    if warn: print(f"Input tensor shape: {data.shape}")

    # Voxel grids
    if data.shape[-1] == 64:
        if warn: 
            print("Input type: Voxel grids")
            print(f"Threshold: {threshold}")
        if len(data.shape) == 4:
            num_objects = data.shape[0]
        else:
            data = data[np.newaxis, ...]
            num_objects = 1
            
        # Get the coordinates of occupied voxels
        objects = [np.argwhere(data[i] >= threshold) for i in range(num_objects)] # data[i] == 1
    
    # Point clouds
    else:
        if warn: print("Input type: Point clouds")
        if len(data.shape) == 3:
            num_objects = data.shape[0]
        else:
            data = data[np.newaxis, ...]
            num_objects = 1
        
        objects = [data[i] for i in range(num_objects)]

    if warn:
        print(f"Num objects: {num_objects}")
        print(f"Min/max/mean: {data.min():.4f} / {data.max():.4f} / {data.mean():.4f}")
    
    cols = int(np.ceil(np.sqrt(num_objects)))
    rows = int(np.ceil(num_objects / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    for i, obj in enumerate(objects):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.scatter(obj[:, 0], obj[:, 2], obj[:, 1], s=1, c=color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save, format='png', bbox_inches='tight', pad_inches=0)
    if warn: print(f"Save path: {save}")
    plt.close()
    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      '-i', default='./data/preprocessed/pc/train/chair/chair_14.npy')
    parser.add_argument('--save_path',  '-o', default=None)
    parser.add_argument('--save_dir',   '-d', default=None)
    parser.add_argument('--color',      '-c', default=None)
    parser.add_argument('--threshold',  '-t', default=0.5, type=float)
    opt = parser.parse_args()
    
    # Load 3D tensor data
    data = np.load(opt.input)
    shape = data.shape
    
    # If save path is specified
    if opt.save_path is not None:
        save = (opt.save_path + ".png").replace(".png.png", ".png")
        
    # Generate save path otherwise
    else:
        if shape[-1] == 64:
            # Voxel grids    
            save = opt.input.split("/")[-1].replace('.npy', '_voxel.png').replace('_voxel_voxel', '_voxel')
        else:
            # Point clouds
            save = opt.input.split("/")[-1].replace('.npy', '_pc.png').replace('_pc_pc', '_pc')
        
    # If save directory is specified
    if opt.save_dir is not None:
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        save = os.path.join(opt.save_dir, save)
    
    visualize(data, save, opt.threshold, color=opt.color)