import torch
import numpy as np


def tangent2plane(coord, in_gpu, return_dict=False):
    x, y, z = coord

    if (x == 0.0) and (y == 0.0) and (z == 0.0):
        x, y, z = 1e-6, 1e-6, 1e-6

    if in_gpu:
        p = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(z, 2))
    else:
        p = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))

    cos_a = x / p
    cos_b = y / p
    cos_c = z / p

    if return_dict:
        return {'Normal': np.asarray([cos_a, cos_b, cos_c]), 'P': np.asarray([p])}
    else:
        return np.asarray([cos_a, cos_b, cos_c, p])


def batch_tangent2plane(coords, normalize=None):
    if normalize is not None:
        if len(normalize.shape) == 1:
            normalize = normalize.unsqueeze(0)
        coords = coords * normalize

    p = torch.sqrt(torch.pow(coords[:, 0], 2) + torch.pow(coords[:, 1], 2) + torch.pow(coords[:, 2], 2))

    cos_a = coords[:, 0] / p
    cos_b = coords[:, 1] / p
    cos_c = coords[:, 2] / p
    return torch.stack([cos_a, cos_b, cos_c, p], dim=-1)


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def cart2sph_tensor(cart_coord):
    x = cart_coord[:, 0].unsqueeze(-1)
    y = cart_coord[:, 1].unsqueeze(-1)
    z = cart_coord[:, 2].unsqueeze(-1)

    hxy = torch.hypot(x, y)
    r = torch.hypot(hxy, z)
    el = torch.atan2(z, hxy)
    az = torch.atan2(y, x)
    sph_coord = torch.cat([az, el, r], dim=-1)

    return sph_coord


def sph2cart_tensor(sph_coord):
    az = sph_coord[:, 0].unsqueeze(-1)
    el = sph_coord[:, 1].unsqueeze(-1)
    r = sph_coord[:, 2].unsqueeze(-1)

    rcos_theta = r * torch.cos(el)
    x = rcos_theta * torch.cos(az)
    y = rcos_theta * torch.sin(az)
    z = r * torch.sin(el)
    cart_coord = torch.cat([x, y, z], dim=-1)
    return cart_coord


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


class Slicer(object):
    """Volume slicer.

    Example:
    # slicer = Slicer(im, img_size=200, device="cuda")
    # out = slicer(planes)
    """

    def __init__(self, im: np.ndarray, out_size: int = 320, device: str = "cpu"):
        """Init slicer.

        Arguments:
            im {numpy.ndarray} -- image volume to be sliced.

        Keyword Arguments:
            out_size {int} -- size of sliced plane (default: {320})
            device {str} -- torch device, "cpu" or "cuda" (default: {"cpu"})
        """
        self.device = device

        self.half_size = np.asarray(im.shape) / 2

        self.img3d_shape = torch.as_tensor(im.shape, dtype=torch.float32, device=self.device)
        self.im = torch.as_tensor(im, dtype=torch.float32, device=self.device)

        # self.set_outsize(out_size)
        self.img2d_shape = torch.as_tensor([out_size, out_size], dtype=torch.float32, device=self.device)
        y_i = torch.arange(out_size, dtype=torch.float32, device=self.device).expand(out_size, -1)
        self.ids = torch.stack((y_i.transpose(1, 0), y_i), dim=-1).view(1, out_size, out_size, 2, 1)

        # self._warmup()

    # def slice(self, planes: numpy.ndarray) -> numpy.ndarray:
    def slice(self, planes):
        """slice plane from planes params

        Arguments:
            planes {numpy.ndarray} -- planes params of shape (N, 4), viewing center of volume as zero point.
                                      p[0] * x + p[1] * y + p[2] * z = p[3]

        Returns:
            [numpy.ndarray] -- sliced planes of shape (N, s, s)
        """
        N = planes.shape[0]
        if isinstance(planes, np.ndarray):
            planes = torch.as_tensor(planes, dtype=torch.float32, device=self.device)
        else:
            planes = planes.to(self.device)
        n = planes[:, :3]
        a = (n[:, 1]).pow(2) / (1 + n[:, 2] + 1e-6) + n[:, 2]
        b = (n[:, 0]).pow(2) / (1 + n[:, 2] + 1e-6) + n[:, 2]
        c = -(n[:, 0] * n[:, 1]) / (1 + n[:, 2] + 1e-6)
        e1 = torch.stack((a, c, -n[:, 0]), dim=-1)
        e2 = torch.stack((c, b, -n[:, 1]), dim=-1)
        # (N, 2, 3), float32
        e = torch.stack((e1, e2), dim=1)

        # (N, 3), float32
        project_point = planes[:, :3] * planes[:, 3:] + self.img3d_shape.view(1, 3) / 2
        # (N, 3), float32
        op = project_point - (e * (self.img2d_shape / 2).view(1, 2, 1)).sum(dim=1)

        # (N, s, s, 3), float32
        plane_points = (self.ids * e.view(-1, 1, 1, 2, 3)).sum(dim=-2) + op.view(-1, 1, 1, 3)

        plane_points = 2 * plane_points / (self.img3d_shape.view(1, 1, 1, 3) - 1) - 1
        plane_points = torch.flip(plane_points, dims=[3])
        plane_points.unsqueeze_(1)

        result = torch.nn.functional.grid_sample(input=self.im.repeat(N, 1, 1, 1, 1),
                                                 grid=plane_points,
                                                 mode='bilinear',
                                                 padding_mode='zeros',
                                                 align_corners=False).squeeze(1).squeeze(1)

        return result

    def __call__(self, planes: np.ndarray):
        return self.slice(planes)
