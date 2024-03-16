import math
import torch
import torch.nn as nn
from loguru import logger
from pytorch3d.ops.knn import knn_points
from src.configs.train_config import NeRFRenderConfig
from .render_utils import sample_pdf, safe_normalize
import torch.nn.functional as F
import numpy as np

class NeRFRenderer(nn.Module):
    def __init__(self, cfg:NeRFRenderConfig, latent_mode:bool=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = cfg
        self.bound = cfg.bound
        self.cascade = 1 + math.ceil(math.log2(cfg.bound))
        self.grid_size = 128
        self.min_near = cfg.min_near
        self.bg_radius = cfg.bg_radius
        self.latent_mode = latent_mode
        self.img_dims = 3+1 if self.latent_mode else 3
        logger.info('Loading CUDA ray marching module (compiling might take a while)...')
        from src.raymarching import raymarchingrgb, raymarchinglatent
        logger.info('\tDone.')
        self.raymarching = raymarchinglatent if self.latent_mode else raymarchingrgb

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-cfg.bound, -cfg.bound, -cfg.bound, cfg.bound, cfg.bound, cfg.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)
        
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def batch_index_select(self, data, inds):
        bs, nv = data.shape[:2]
        device = data.device
        inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        data = data.reshape(bs*nv, *data.shape[2:])
        return data[inds.long()]

    def smplx_lbsmap_top_k(self, lbs_weights, verts_transform, points, template_points, source_points=None, K=1, addition_info=None):
        '''ref: https://github.com/JanaldoChen/Anim-NeRF
        Args:  
        '''
        with torch.no_grad():
            results = knn_points(points, template_points, K=K)
            dists, idxs = results.dists, results.idx
        neighbs_dist = dists
        neighbs = idxs
        weight_std = 0.1
        weight_std2 = 2. * weight_std ** 2
        xyz_neighbs_lbs_weight = lbs_weights[neighbs] 
        xyz_neighbs_weight_conf = torch.exp(-torch.sum(torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1)/weight_std2) 
        xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
        xyz_neighbs_weight = torch.exp(-neighbs_dist) 
        xyz_neighbs_weight *= xyz_neighbs_weight_conf
        xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True)
        xyz_neighbs_transform = self.batch_index_select(verts_transform, neighbs)
        xyz_transform = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) * xyz_neighbs_transform, dim=2) 
        xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True) 

        if addition_info is not None:
            xyz_neighbs_info = self.batch_index_select(addition_info, neighbs)
            xyz_info = torch.sum(xyz_neighbs_weight.unsqueeze(-1)* xyz_neighbs_info, dim=2)
            return xyz_dist, xyz_transform, xyz_info
        else:
            return xyz_dist, xyz_transform

    def transfer(self, xyz, target_scale=3, dx=0, dy=-0.78, dz=0):  
        xyz = xyz[0]
        xyz[:, 2] -= dz
        xyz[:, 1] -= dy
        xyz[:, 0] -= dx
        xyz /= target_scale
        return xyz.unsqueeze(0)

    def backward_skinning(self, xyz, viewdir,posed_verts, mesh_verts_view, curr_vertices_transform, curr_offsets, canonical_transform, canonical_offsets, lbs_weights):
        '''
            xyz: [B, n_ray, n_point, 3]
        '''
        batch_size=1
        xyz = self.transfer(xyz.clone())
        vertices_transform = torch.inverse(curr_vertices_transform)
        vertices_transform_3x1 = vertices_transform[...,:3,3] - curr_offsets + canonical_offsets
        vertices_transform_3x3 = torch.cat([vertices_transform[...,:3,:3], vertices_transform_3x1[...,None]], dim=-1)
        vertices_transform = torch.cat([vertices_transform_3x3, vertices_transform[...,3:,:]], dim=-2)
        vertices_transform = torch.matmul(canonical_transform.clone().repeat(batch_size, 1, 1, 1), vertices_transform)
        surface_verts = posed_verts
        xyz_dist, xyz_transform = self.smplx_lbsmap_top_k(lbs_weights, vertices_transform, 
                                xyz, surface_verts, K=self.opt.k_neigh,) 
        xyz = self.batch_transform(xyz_transform, xyz) 
        return xyz  
    
    def batch_transform(self, P, v, pad_ones=True):
        if pad_ones:
            homo = torch.ones((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
        else:
            homo = torch.zeros((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
        v_homo = torch.cat((v, homo), dim=-1)
        v_homo = torch.matmul(P, v_homo.unsqueeze(-1))
        v_ = v_homo[..., :3, 0]
        return v_
    
    def normalize_mesh(self, vertices, target_scale=3, dx=0, dy=-0.78, dz=0):
        verts = vertices[0]
        verts *= target_scale
        verts[:, 2] += dz
        verts[:, 1] += dy
        verts[:, 0] += dx
        return verts.unsqueeze(0)
    
    def run(self, step, posed_verts, mesh_verts_view, mesh_faces_view, curr_vertices_transform, curr_offsets, canonical_transform, canonical_offsets,lbs_weights, mesh_latent, mesh_depth, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, train=None, **kwargs): 
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        mesh_depth = mesh_depth.contiguous().view(-1, 1)
        mesh_latent = mesh_latent.contiguous().view(-1, self.img_dims)
        if bg_color != None:
            bg_color = bg_color.contiguous().view(-1, self.img_dims)
        num_steps = self.opt.num_steps
        upsample_steps = self.opt.upsample_steps
        N = rays_o.shape[0] 
        device = rays_o.device
        results = {}
        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer 
        # sample steps
        nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near) 
        if self.opt.mesh_depth_as_index:
            mesh_intersection_idx = torch.nonzero(mesh_depth.squeeze(1)!=0).squeeze()
            import cubvh
            BVH = cubvh.cuBVH(mesh_verts_view, mesh_faces_view) 
            intersections, face_id, depth = BVH.ray_trace(rays_o, rays_d) 
            intersection_idx = mesh_intersection_idx
            fars[intersection_idx] = depth[intersection_idx] + self.opt.far_offset
            nears.unsqueeze_(-1)
            fars.unsqueeze_(-1)
        else:
            nears.unsqueeze_(-1)
            fars.unsqueeze_(-1)
        # random sample light_d if not provided
        if light_d is None:
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) 
        z_vals = z_vals.expand((N, num_steps)) 
        z_vals = nears + (fars - nears) * z_vals
        # perturb z_vals
        sample_dist = (fars - nears) / num_steps 
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) 
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])
        # skinning
        xyzs_init = xyzs.reshape(-1,3).unsqueeze(0)
        xyzs = self.backward_skinning(xyzs_init, rays_d, posed_verts, mesh_verts_view, curr_vertices_transform, curr_offsets, canonical_transform, canonical_offsets, lbs_weights)
        xyzs = self.normalize_mesh(xyzs)
        xyzs = xyzs.reshape(N,-1,3)
        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)
        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1] 
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) 
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] 
                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) 
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() 
                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1)
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])
                # skinning
                xyzs_upsample = new_xyzs.reshape(-1,3).unsqueeze(0)
                xyzs_upsample = self.backward_skinning(xyzs_upsample, rays_d, posed_verts, mesh_verts_view, curr_vertices_transform, curr_offsets, canonical_transform, canonical_offsets, lbs_weights)
                xyzs_upsample = self.normalize_mesh(xyzs_upsample.clone())
                new_xyzs = xyzs_upsample.reshape(N,-1,3)
            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) 
            z_vals, z_index = torch.sort(z_vals, dim=1)
            xyzs = torch.cat([xyzs, new_xyzs], dim=1) 
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))
            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))
        
        if self.opt.perturb_z_vals:
            offset = (torch.rand(z_vals.shape, device=device)) * sample_dist*0.5
            z_vals[:,:-1] = z_vals[:,:-1]+offset[:,:-1]
            xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        deltas_init = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        deltas = torch.cat([deltas, self.opt.last_point_weight * torch.ones_like(deltas[..., :1])], dim=-1)
        if self.opt.perturb_sigma:
            noise = torch.randn(density_outputs['sigma'].shape, device=device)*(np.maximum((1 - step/150),0))
            alphas = 1 - torch.exp(-deltas * F.relu((density_outputs['sigma']+noise).squeeze(-1)))
            alphas_init = 1 - torch.exp(-deltas_init * F.relu((density_outputs['sigma']+noise).squeeze(-1))) 
        else:
            alphas = 1 - torch.exp(-deltas * F.relu((density_outputs['sigma']).squeeze(-1)))
            alphas_init = 1 - torch.exp(-deltas_init * F.relu((density_outputs['sigma']).squeeze(-1))) 
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])
        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, self.img_dims) 
        mesh_latent = mesh_latent.reshape([N,1,self.img_dims])
        rgbs = torch.cat([rgbs[:,:-1,:], mesh_latent], dim=1)
        
        # orientation loss
        if normals is not None:
            normals = normals.view(N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.mean()
        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)
        weights_except_last = weights[:,:-1].sum(dim=-1)
        weights_sum = weights.sum(dim=-1)
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)
        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) 
        bg_color = 1
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, self.img_dims)
        depth = depth.view(*prefix)
        mask = (nears < fars).reshape(*prefix)
        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['weights_except_last'] = weights_except_last
        results['mask'] = mask 
        results['xyzs'] = xyzs.unsqueeze(0)  
        results['sigmas'] = density_outputs['sigma'].view(N, num_steps+upsample_steps).unsqueeze(0)
        results['alphas'] = alphas 
        results['alphas_init'] = alphas_init
        results['intersection_idx'] = intersection_idx
        return results

    def render(self, step, posed_verts, mesh_verts_view, mesh_faces_view, curr_vertices_transform, curr_offsets, canonical_transform, canonical_offsets,lbs_weights, mesh_latent,mesh_depth,  rays_o, rays_d, staged=False, max_ray_batch=65536, bg_color=None, train=None, **kwargs):
        _run = self.run
        B, N = rays_o.shape[:2]
        device = rays_o.device
        if staged:
            depth = torch.empty((B, N), device=device)
            weights_except_last = torch.empty((B, N), device=device)
            weights_sum = torch.empty((B, N), device=device)
            image = torch.empty((B, N, self.img_dims), device=device)
            xyzs = torch.empty((B, N, self.opt.num_steps+self.opt.upsample_steps, 3), device=device)
            sigmas = torch.empty((B, N, self.opt.num_steps+self.opt.upsample_steps), device=device)
            alphas = torch.empty((B, N, self.opt.num_steps+self.opt.upsample_steps), device=device)
            alphas_init = torch.empty((B, N, self.opt.num_steps+self.opt.upsample_steps), device=device)
            mesh_depth_stage = mesh_depth.reshape(B,1,N).permute(0,2,1)
            mesh_latent_stage = mesh_latent.reshape(B,-1,N).permute(0,2,1)
            intersection_idx = []
            
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(step, posed_verts, mesh_verts_view, mesh_faces_view,curr_vertices_transform, curr_offsets, canonical_transform, canonical_offsets,lbs_weights, mesh_latent_stage[b:b+1, head:tail], mesh_depth_stage[b:b+1, head:tail], rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs) # , bg_color=bg_color[b:b+1, head:tail] mesh_depth_stage[b:b+1, head:tail], parsing_hair_mask_stage[b:b+1, head:tail] parsing_hair_mask_stage[b:b+1, head:tail], mesh_hair_mask_stage[b:b+1, head:tail],
                    depth[b:b+1, head:tail] = results_['depth']
                    weights_except_last[b:b+1, head:tail] = results_['weights_except_last']
                    weights_sum[b:b+1, head:tail] = results_['weights_sum']
                    image[b:b+1, head:tail] = results_['image']
                    xyzs[b:b+1, head:tail] = results_['xyzs']
                    sigmas[b:b+1, head:tail] = results_['sigmas']
                    alphas[b:b+1, head:tail] = results_['alphas']
                    alphas_init[b:b+1, head:tail] = results_['alphas_init']
                    intersection_idx.append(results_['intersection_idx'])
                    head += max_ray_batch
            results = {}
            results['depth'] = depth
            results['weights_except_last'] = weights_except_last
            results['image'] = image
            results['weights_sum'] = weights_sum
            results['xyzs'] = xyzs
            results['sigmas'] = sigmas
            results['alphas'] = alphas
            results['alphas_init'] = alphas_init
            results['intersection_idx'] = intersection_idx
        else:
            results = _run(step, posed_verts, mesh_verts_view, mesh_faces_view, mesh_latent, rays_o, rays_d, **kwargs)
        return results