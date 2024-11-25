from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)

        xt, eps = self.var_scheduler.add_noise(x0, timestep)

        # eps_pred = self.network(xt, timestep)
        x0_pred = self.network(xt, timestep)

        # eps = torch.sigmoid(eps)
        # eps_pred = torch.sigmoid(eps_pred)

        # loss = F.mse_loss(eps_pred, eps, reduction='mean')

        loss = F.binary_cross_entropy(x0_pred, x0)
        loss = loss.mean()
        ######################
        return loss

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def voxel_resolution(self):
        return self.network.voxel_resolution

    @torch.no_grad()
    def sample(
            self,
            batch_size,
            return_traj=False,
            class_label: Optional[torch.Tensor] = None,
            guidance_scale: Optional[float] = 0.0,
    ):
        x_T = torch.randn([batch_size, 1, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution]).to(self.device)  # Project

        do_classifier_free_guidance = guidance_scale > 0.0

        if do_classifier_free_guidance:
            ######## TODO ########
            # Assignment 2-3. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"

            null_class_label = torch.zeros_like(class_label)
            total_class_labels = torch.cat([null_class_label, class_label]).to(self.device)
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.

                # x_t_cat = torch.cat([x_t, x_t]).to(self.device)
                # noise_pred = self.network(x_t_cat, t.to(self.device), total_class_labels)  # training
                #
                # noise_null = noise_pred[:batch_size]
                # noise_condition = noise_pred[batch_size:]
                #
                # noise_pred = (1 + guidance_scale) * noise_condition - guidance_scale * noise_null

                x_t_cat = torch.cat([x_t, x_t]).to(self.device)
                x0_pred = self.network(x_t_cat, t.to(self.device), total_class_labels)  # training

                noise_null = x0_pred[:batch_size]
                noise_condition = x0_pred[batch_size:]

                x0_pred = (1 + guidance_scale) * noise_condition - guidance_scale * noise_null

                #######################
            else:
                # noise_pred = self.network(
                #     x_t,
                #     timestep=t.to(self.device),
                #     class_label=class_label,
                # )
                x0_pred = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label,
                )

            # x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)
            x_t_prev = self.var_scheduler.step(x_t, t, x0_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
