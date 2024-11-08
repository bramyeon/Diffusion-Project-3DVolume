import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import ShapeNetVoxelDataModule, get_data_iterator
from dotmap import DotMap
from model import DiffusionModule
from network import UNet3D
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from tqdm import tqdm


matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    if args.use_cfg:
        save_dir = Path(f"results/cfg_diffusion-{args.sample_method}-{now}")
    else:
        save_dir = Path(f"results/diffusion-{args.sample_method}-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    ####### Project #######
    voxel_resolution = 128
    ds_module = ShapeNetVoxelDataModule(
        "../data",
        batch_size=config.batch_size,
        num_workers=4,
        max_num_voxels_per_cat=config.max_num_voxels_per_cat,
        voxel_resolution=voxel_resolution
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Set up the scheduler
    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )

    network = UNet3D(
        T=config.num_diffusion_train_timesteps,
        voxel_resolution=voxel_resolution,
        #ch=64, ## Project ## (128->64)
        ch=16, # for test
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    ddpm = DiffusionModule(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                print("HEllow!!!")
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                if args.use_cfg:  # Conditional, CFG training
                    samples = ddpm.sample(
                        4,
                        class_label=torch.randint(1, 4, (4,)).to(config.device),
                        return_traj=False,
                    )
                else:  # Unconditional training
                    samples = ddpm.sample(4, return_traj=False)


                ### Project ###
                # for i, voxel in enumerate(samples):
                #     voxel = voxel.squeeze(1)  # Remove channel
                #     np.save(save_dir/ f"step={step}-{i}", voxel.cpu().numpy())
                ###############

                # ddpm.save(f"{save_dir}/last.ckpt")
                # print("111111")
                ddpm.train()

            ################## Project ##################

            voxel, label = next(train_it)
            voxel = voxel.unsqueeze(1)  # Adding channel
            voxel, label = voxel.to(config.device), label.to(config.device)
            # print("222222")
            if args.use_cfg:  # Conditional, CFG training
                loss = ddpm.get_loss(voxel, class_label=label)
            else:  # Unconditional training
                loss = ddpm.get_loss(voxel)
            pbar.set_description(f"Loss: {loss.item():.4f}")
            # print("3333333")
            optimizer.zero_grad()
            # print("444444")
            loss.backward()
            # print("555555")
            optimizer.step()
            # print("666666")
            scheduler.step()
            # print("777777")
            losses.append(loss.item())
            # print("888888")

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=2)  # 200
    parser.add_argument(
        "--max_num_voxels_per_cat",
        type=int,
        default=3000,
        help="max number of images per category for Shapenet dataset",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=2,  # 1000
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--voxel_resolution", type=int, default=128)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
