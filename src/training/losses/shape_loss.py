import torch
from torch import nn
from igl import read_obj
import numpy as np
from src.configs.train_config import GuideConfig
from src.models.mesh_utils import MeshOBJ
from src.models.mesh import Mesh
import torch.functional as F
import os, sys
os.chdir(sys.path[0])

def ce_pq_loss(p, q, weight=None):  # p: gt   q: predict
    def clamp(v, T=0.01):
        return v.clamp(T, 1 - T)
    ce = -1 * (p * torch.log(clamp(q)) + (1 - p) * torch.log(clamp(1 - q)))
    if weight is not None:
        ce *= weight
    return ce.sum()

class ShapeLoss(nn.Module):
    def __init__(self, cfg:GuideConfig, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        v, _, _, f, _, _ = read_obj(self.cfg.topology_path, float)
        mesh = MeshOBJ(v, f)
        self.sketchshape = mesh.normalize_mesh(cfg.shape_scale, cfg.dx, cfg.dy, cfg.dz)

    # reverseweight shapeloss newsigma
    def forward(self, xyzs, sigmas, train_nerf_step, ):
        if self.cfg.proximal_surface > 0: 
            weight = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
        else:
            weight = None
        nerf_occ = sigmas
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        indicator = torch.zeros_like(nerf_occ)
        loss = ce_pq_loss(nerf_occ, indicator, weight=weight)  # order is important for CE loss + second argument may not be optimized
        return loss