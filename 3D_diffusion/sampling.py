import argparse

import numpy as np
import torch
# from dataset import tensor_to_pil_image
from model import DiffusionModule
from scheduler import DDPMScheduler
from pathlib import Path


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    num_train_timesteps = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=1e-4,
        beta_T=0.02,
        mode="linear",
    ).to(device)

    total_num_samples = 1000
    num_categories = 3
    num_batches = int(np.ceil(total_num_samples / args.batch_size))
    samples_list_all = [[],[],[]]
    samples_list = []
    category_idx = 2  # For one category sampling, airplane = 0, chair = 1, table = 2

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:  # Enable CFG sampling
            assert ddpm.network.use_cfg, f"The model was not trained to support CFG."
            for i in range(num_categories):
                samples = ddpm.sample(
                    B,
                    class_label=torch.full((B,), i+1),
                    guidance_scale=args.cfg_scale,
                )
                voxel = samples.squeeze(1).clamp(0, 1).detach()  # Remove channel
                samples_list_all[i].append(voxel)

        else:
            samples = ddpm.sample(
                B,
                class_label=torch.full((B,), category_idx+1),
                guidance_scale=0.0,
            )

            voxel = samples.squeeze(1).clamp(0, 1).detach()  # Remove channel
            samples_list.append(voxel)

    if args.use_cfg:
        for i in range(num_categories):
            category_samples = torch.cat(samples_list[i])
            np.save(save_dir / f"{category_idx}", category_samples.cpu().numpy())
            print(f"Saved the {category_idx}-th category's voxels.")
    else:
        samples_list = torch.cat(samples_list)
        np.save(save_dir / f"{category_idx}", samples_list.cpu().numpy())
        print(f"Saved the {category_idx}-th category's voxels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
