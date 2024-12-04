import math

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from diffusers import DDIMScheduler

from models import base_block as bb
from help_func import tensor2array
from plane_func import batch_tangent2plane


class SPLDiffusionModel(bb.BaseModule):
    def __init__(self,
                 diffusion_config,
                 denoise_config,
                 loss_weight):
        super(SPLDiffusionModel, self).__init__()

        self.feat_extractor = bb.PlaneFeatExtractor()

        # freeze the feature extractor [optional]
        # for param in self.feat_extractor.parameters():
        #     param.requires_grad = False

        self.diffusion_config = diffusion_config

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.diffusion_config['TrainTimeSteps'],
            beta_schedule=self.diffusion_config['Scheduler'],
            beta_start=self.diffusion_config['BetaStart'],
            beta_end=self.diffusion_config['BetaEnd'],
            clip_sample=False,
        )

        self.noise_scheduler.set_timesteps(diffusion_config['InferenceSteps'])

        self.denoise_config = denoise_config

        self.denoise_net = bb.DenoiseUNet()

        self.prior_mean = torch.nn.Parameter(
            torch.FloatTensor(np.load(diffusion_config['Prior']['Mean'])), requires_grad=False)
        self.prior_std = torch.nn.Parameter(
            torch.FloatTensor(np.load(diffusion_config['Prior']['Std'])), requires_grad=False)

        self.loss_weight = loss_weight

    def forward(self, tangent, slicer, bsize):

        tangent = torch.FloatTensor(tangent).to(self.model_device)

        tangent = tangent / slicer.img3d_shape * 2

        tangent = tangent.unsqueeze(0).repeat(bsize, 1)

        noise = torch.randn_like(tangent).to(self.model_device) * self.prior_std

        time_step = torch.randint(low=0, high=self.noise_scheduler.num_train_timesteps - 1,
                                  size=(bsize,)).to(self.model_device)

        # This is the version for adaptively adding the noise on the normalized tangent.
        noisy_tangent = self.noise_scheduler.add_noise(tangent - self.prior_mean,
                                                       noise, time_step)

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.model_device)
        sqrt_alpha_prod = alphas_cumprod[time_step] ** 0.5
        noisy_plane_param = batch_tangent2plane(noisy_tangent + self.prior_mean * sqrt_alpha_prod.unsqueeze(1),
                                                slicer.img3d_shape / 2)

        # This is the original version for adding noise on the tangent param and plane transform
        # noisy_tangent = self.noise_scheduler.add_noise(tangent,
        #                                                noise, time_step)
        #
        # noisy_plane_param = batch_tangent2plane(noisy_tangent, slicer.img3d_shape / 2)

        sliced_planes = slicer.slice(noisy_plane_param)

        plane_feats = self.feat_extractor(sliced_planes.unsqueeze(1))

        noise_pred = self.denoise_net(noisy_tangent, plane_feats, time_step)

        fwd = {'Noise': noise, 'PredNoise': noise_pred}

        return fwd

    def loss_func(self, fwd):
        noise_score = F.mse_loss(fwd['PredNoise'].squeeze(), fwd['Noise'].squeeze())

        loss_total = noise_score / self.prior_std.mean()
        loss_info = {'NoiseScore': noise_score.item()}
        print_info = f'NoiseScore: {noise_score.item():.5f} '

        return loss_total, loss_info, print_info

    def evaluate(self, tangent, slicer, bsize):
        tangent = torch.FloatTensor(tangent).to(self.model_device)
        tangent = tangent.unsqueeze(0).repeat(bsize, 1)

        tangent = tangent / slicer.img3d_shape * 2

        noise = torch.randn_like(tangent).to(self.model_device) * self.prior_std

        time_step = torch.LongTensor([self.diffusion_config['TrainTimeSteps']-1]).to(self.model_device)
        noisy_tangent = self.noise_scheduler.add_noise(tangent - self.prior_mean, noise, time_step)

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.model_device)
        sqrt_alpha_prod = alphas_cumprod[time_step] ** 0.5
        noisy_plane_param = batch_tangent2plane(noisy_tangent + self.prior_mean * sqrt_alpha_prod.unsqueeze(1),
                                                slicer.img3d_shape / 2)

        sliced_planes = slicer.slice(noisy_plane_param)
        plane_feats = self.feat_extractor(sliced_planes.unsqueeze(1))

        for t in self.noise_scheduler.timesteps:

            pred_noise = self.denoise_net(noisy_tangent, plane_feats, t.float().expand(bsize,).to(self.model_device))
            out = self.noise_scheduler.step(pred_noise.squeeze(-1).squeeze(-1), t, noisy_tangent)
            noisy_tangent = out.prev_sample

            sqrt_alpha_prod = alphas_cumprod[time_step] ** 0.5
            noisy_plane_param = batch_tangent2plane(noisy_tangent + self.prior_mean * sqrt_alpha_prod.unsqueeze(1),
                                                    slicer.img3d_shape / 2)
            sliced_planes = slicer.slice(noisy_plane_param)
            plane_feats = self.feat_extractor(sliced_planes.unsqueeze(1))

        tangent = tangent * slicer.img3d_shape / 2
        noisy_tangent = noisy_tangent + self.prior_mean
        noisy_tangent = noisy_tangent * slicer.img3d_shape / 2
        tangent_error = torch.linalg.norm(tangent - noisy_tangent, dim=-1).mean().item()

        noisy_plane_param = batch_tangent2plane(noisy_tangent)
        plane_param = batch_tangent2plane(tangent)

        cosine = torch.cosine_similarity(plane_param[:, :3], noisy_plane_param[:, :3], dim=1)
        angle = torch.arccos(torch.abs(cosine)).mean().item() * 180 / math.pi
        distance = torch.abs(plane_param[:, 3:] - noisy_plane_param[:, 3:]).mean().item()

        return {'TangentDistance': tangent_error, 'Angle': angle, 'Distance': distance}
