import os
import math
import torch
import torch.nn as nn
from torchvision.models import resnet18

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.device_param = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, *args, **kwargs):
        return

    def loss_func(self, fwd):
        return

    @property
    def model_device(self):
        return self.device_param.device

    def load_weight(self, weight_path):
        if not os.path.exists(weight_path):
            raise ValueError('Path Not Exist!')
        pretrained_dict = torch.load(weight_path, map_location=self.device_param.device)
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        if len(pretrained_dict) == len(model_dict):
            print('No dropped weights')
        else:
            print('Weights dropped!!')

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class PlaneFeatExtractor(nn.Module):
    def __init__(self):
        super(PlaneFeatExtractor, self).__init__()
        net = resnet18()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3:5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


def conv_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self.init_conv = nn.ConvTranspose2d(96, 256, kernel_size=7)

        self.dec1 = conv_block(256, 128)
        self.dec2 = conv_block(128, 64)
        self.dec3 = conv_block(64, 32)
        self.dec4 = conv_block(32, 16)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.up(x)
        x = self.dec1(x)
        x = self.up(x)
        x = self.dec2(x)
        x = self.up(x)
        x = self.dec3(x)
        x = self.up(x)
        x = self.dec4(x)
        x = self.up(x)
        x = self.out(x)
        return torch.sigmoid(x)


class TemporalResBlock(nn.Module):
    def __init__(self, in_chan, out_chan, t_chan):
        super(TemporalResBlock, self).__init__()
        self.t_emd = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(t_chan, in_chan, 1),
        )

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.norm1 = nn.GroupNorm(32, out_chan)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.norm2 = nn.GroupNorm(32, out_chan)

        if in_chan == out_chan:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_chan, out_chan, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.t_emd(t)
        x = x + t_emb

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        out = self.skip_connection(x) + h
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoiseUNet(BaseModule):
    def __init__(self):
        super(DenoiseUNet, self).__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(16)

        self.time_mlp = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.Conv2d(64, 64, 1)
        )

        self.plane_enc = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=7),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 96, kernel_size=1),
        )

        self.tangent_enc = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),

        )

        self.image_decoder = ImageDecoder()

        self.enc0 = nn.Conv2d(128, 128, kernel_size=1)

        self.enc1 = TemporalResBlock(128, 256, 64)
        self.enc2 = TemporalResBlock(256, 512, 64)
        self.mid = TemporalResBlock(512, 512, 64)
        self.dec2 = TemporalResBlock(512+512, 256, 64)
        self.dec1 = TemporalResBlock(256+256, 128, 64)

        self.dec0 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1, bias=False)
        )

    def forward(self, x, plane_feat, time_feat):
        plane_feat = self.plane_enc(plane_feat)

        time_feat = self.time_embedding(time_feat)
        time_feat = self.time_mlp(time_feat.unsqueeze(-1).unsqueeze(-1))

        enc_x = self.tangent_enc(x.unsqueeze(-1).unsqueeze(-1))

        x_in = torch.cat([plane_feat, enc_x], dim=1)

        enc0 = self.enc0(x_in)

        enc1 = self.enc1(enc0, time_feat)
        enc2 = self.enc2(enc1, time_feat)

        mid = self.mid(enc2, time_feat)

        dec2 = self.dec2(torch.cat([mid, enc2], dim=1), time_feat)
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1), time_feat)

        dec0 = self.dec0(dec1)

        recon_img = self.image_decoder(plane_feat)

        return {'PredNoise': dec0, 'ReconImage': recon_img}