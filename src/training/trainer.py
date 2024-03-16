from pathlib import Path
from typing import Any, Dict, Union, List
import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch.utils.data import DataLoader
from tqdm import tqdm
from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.models.mesh import Mesh
from src.models.smplx import SMPLX
from src.stable_diffusion_depth import StableDiffusion
from src.stable_diffusion_2base import NeRFStableDiffusion
from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy
from src.training.nerf_dataset import NeRFDataset

class TECA:
    def __init__(self, cfg: TrainConfig, shape=None, expression=None, jaw_pose=None, pose=None):
        self.cfg = cfg
        self.paint_step = 0
        self.nerf_train_step = 0
        self.jaw_pose = jaw_pose
        self.pose = pose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = self.cfg.optim.seed
        utils.seed_everything(self.seed)
        self.view_dirs = ['front', 'left side', 'back', 'right side', 'overhead', 'bottom']

        # make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.cfg.log.ckpt_dir / f'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        # init log
        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))
        
        # load shape and expression parameters and init mesh
        self.view_pose = self.mesh_view_pose()
        if shape is not None:
            self.shape = shape.to(device=self.device)
        else:
            self.shape = torch.zeros([1,self.cfg.nerfrender.n_shape]).to(device=self.device)
        if expression is not None:
            self.expression = expression.to(device=self.device)
        else:
            self.expression = torch.zeros([1,self.cfg.nerfrender.n_exp]).to(device=self.device)

        # init models
        self.canonical_pose = self.init_pose()
        self.canonical_mesh()
        self.view_mesh()
        self.mesh_model = self.init_mesh_model()
        self.nerf_model = self.init_nerf_model()
        self.mesh_diffusion = self.init_mesh_paint_diffusion()
        self.nerf_diffusion = self.init_nerf_diffusion()
        self.mesh_paint_text_z = self.cal_mesh_paint_text_embeddings()
        self.nerf_text_z = self.cal_nerf_text_embeddings()
        self.losses = self.init_losses()
        self.nerf_optimizer, self.nerf_scaler = self.init_nerf_optimizer()
        self.dataloaders = self.init_dataloaders()
        self.back_img = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0, 1) / 255.0
        
        # load ckpt
        self.past_checkpoints = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)
        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')
        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_nerf_model(self):
        if self.cfg.nerfrender.backbone == 'grid':
            from src.models.network_grid import NeRFNetwork
        else:
            raise ValueError(f'{self.cfg.nerfrender.backbone} is not a valid backbone name')
        model = NeRFNetwork(self.cfg.nerfrender).to(self.device)
        logger.info(
            f'Loaded {self.cfg.nerfrender.backbone} NeRF, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model
    
    def init_mesh_model(self):
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  initial_texture_path = None,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  vertices=self.mesh_verts_view,
                                  face=self.mesh_faces_view,
                                  vt=self.mesh_vt_view,
                                  ft=self.mesh_ft_view)
        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_mesh_paint_diffusion(self):
        diffusion_model = StableDiffusion(self.device, model_name='stabilityai/stable-diffusion-2-depth',
                                          latent_mode=True,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model
    
    def init_nerf_diffusion(self):
        nerf_diffusion_model = NeRFStableDiffusion(self.device, model_name='stabilityai/stable-diffusion-2-base',  #  'stabilityai/stable-diffusion-2-base' 'timbrooks/instruct-pix2pix' stabilityai/stable-diffusion-2-base CompVis/stable-diffusion-v1-4"stabilityai/stable-diffusion-2"
                                          latent_mode=self.nerf_model.latent_mode)
        for p in nerf_diffusion_model.parameters():
            p.requires_grad = False
        return nerf_diffusion_model

    def cal_mesh_paint_text_embeddings(self):
        ref_text = self.cfg.guide.paint_text
        if not self.cfg.guide.append_direction:
            text_z = self.mesh_diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in self.view_dirs:
                text = ref_text.format(d)
                logger.info(text)
                negative_prompt = self.cfg.guide.paint_neg_text # negative prompt is more powerful to avoid bad cases
                text_z.append(self.mesh_diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        return text_z

    def cal_nerf_text_embeddings(self):
        ref_text = self.cfg.guide.nerf_text 
        if not self.cfg.guide.append_direction:
            text_z = self.nerf_diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in self.view_dirs:
                text = ref_text.format(d)
                logger.info(text)
                negative_prompt = self.cfg.guide.nerf_neg_text
                text_z.append(self.nerf_diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        return text_z

    # can use different background color in nerf training stage
    def calc_background_color(self):
        from torchvision import transforms
        background = []
        H = W = self.cfg.nerfrender.train_mesh_rgb_dim
        H_latent = W_latent = self.cfg.nerfrender.train_mesh_depth_dim
        for i in range(7):
            # white in rgb space 3 channels
            if i == 0:
                bg_rgb = torch.ones(1,3,H,W).to(device=self.device)
                background.append((self.nerf_diffusion.encode_imgs(bg_rgb)))
            # white in rgb space 4 channels
            if i == 1:
                bg_latent = torch.ones(1,4,H_latent,W_latent).to(device=self.device)
                background.append(bg_latent)
            # black in rgb space 3 channels
            if i == 2:
                bg_rgb = torch.zeros(1,3,H,W).to(device=self.device)
                background.append((self.nerf_diffusion.encode_imgs(bg_rgb)))
            # black in rgb space 4 channels
            if i == 3:
                bg_latent = torch.zeros(1,4,H_latent,W_latent).to(device=self.device)
                background.append(bg_latent)
            # gaussian background1
            if i == 4:
                gaussian = torch.normal(torch.zeros([H, W, 1]) + 0.5, torch.zeros([H, W, 1]) + 0.2)
                gaussian = gaussian.unsqueeze(0).repeat(1, 1, 1, 3)
                bg_rgb = torch.clamp(gaussian, min=0, max=1).permute(0,3,1,2).to(device=self.device)
                background.append((self.nerf_diffusion.encode_imgs(bg_rgb)))
            # gaussian background1
            if i == 5:
                gaussian0 = torch.normal(torch.zeros([H, W, 1]) + 0.5, torch.zeros([H, W, 1]) + 0.2)
                gaussian1 = torch.normal(torch.zeros([H, W, 1]) + 0.5, torch.zeros([H, W, 1]) + 0.1)
                gaussian2 = torch.normal(torch.zeros([H, W, 1]) + 0.5, torch.zeros([H, W, 1]) + 0.5)
                bg_rgb = torch.cat((gaussian0, gaussian1, gaussian2),dim=2).unsqueeze(0).permute(0,3,1,2).to(device=self.device)
                background.append((self.nerf_diffusion.encode_imgs(bg_rgb)))
            # chess board
            if i == 6:
                chess_board = torch.zeros([H, W, 1]) + 0.2
                chess_length = H // np.random.choice(np.arange(10,20))
                i, j = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='xy')
                div_i, div_j = i // chess_length, j // chess_length
                white_i, white_j = i[(div_i + div_j) % 2 == 0], j[(div_i + div_j) % 2 == 0]
                chess_board[white_i, white_j] = 0.8
                blur_fn = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
                pred_back = blur_fn(chess_board.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0).reshape(-1,H,W).to(device=self.device)
                bg_rgb = pred_back.repeat(3, 1, 1).unsqueeze(0)
                background.append((self.nerf_diffusion.encode_imgs(bg_rgb)))
        return background 

    def init_losses(self):
        losses = {}
        if self.cfg.optim.lambda_shape > 0:
            from src.training.losses.shape_loss import ShapeLoss
            losses['shape_loss'] = ShapeLoss(self.cfg.guide, device=self.device)
        if self.cfg.optim.lambda_sparsity > 0:
            from src.training.losses.sparsity_loss import sparsity_loss
            losses['sparsity_loss'] = sparsity_loss
        if self.cfg.optim.lambda_blip_sim >0:
            from src.training.losses.blip_sim_loss import blip_sim_loss
            losses['blip_sim_loss'] = blip_sim_loss(self.cfg, device=self.device)
        if self.cfg.optim.lambda_mask > 0:
            from src.training.losses.mask_loss import mask_loss
            losses['mask_loss'] = mask_loss(self.cfg, device=self.device)
        return losses
    
    def init_nerf_optimizer(self):
        nerf_optimizer = torch.optim.Adam(self.nerf_model.get_params(self.cfg.optim.nerf_lr), betas=(0.9, 0.99), eps=1e-15)
        nerf_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)
        return nerf_optimizer, nerf_scaler
    
    def init_dataloaders(self):
        mesh_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()
        mesh_val_dataloader = ViewsDataset(self.cfg.render, device=self.device, size=self.cfg.log.eval_size).dataloader()
        # large means render the final video with more frames
        mesh_val_large_dataloader = ViewsDataset(self.cfg.render, device=self.device, size=self.cfg.log.full_eval_size).dataloader()
        nerf_train_dataloader = NeRFDataset(self.cfg.nerfrender, device=self.device, type='train', H=self.cfg.nerfrender.train_h,
                                       W=self.cfg.nerfrender.train_w, size=100).dataloader()
        nerf_val_dataloader = NeRFDataset(self.cfg.nerfrender, device=self.device, type='val', H=self.cfg.nerfrender.eval_h,
                                 W=self.cfg.nerfrender.eval_w, size=self.cfg.log.eval_size).dataloader()
        # large means render the final video with more frames
        nerf_val_large_loader = NeRFDataset(self.cfg.nerfrender, device=self.device, type='val', H=self.cfg.nerfrender.eval_h,
                                       W=self.cfg.nerfrender.eval_w, size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'mesh_train': mesh_train_dataloader, 'nerf_train': nerf_train_dataloader, 'mesh_val': mesh_val_dataloader, 'nerf_val': nerf_val_dataloader,
                       'mesh_val_large': mesh_val_large_dataloader, 'nerf_val_large': nerf_val_large_loader}
        return dataloaders
    
    def canonical_mesh(self):
        # set canonical space
        self.smplx = SMPLX(self.cfg).to(self.device)
        self.lbs_weights = self.smplx.lbs_weights
        xyz_c, _, _, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = self.canonical_pose, return_T=True)
        xyz_c = self.normalize_mesh(xyz_c.clone())
        self.v_template = xyz_c.squeeze().to(self.device)
        self.canonical_transform = T
        self.canonical_offsets = shape_offsets + pose_offsets
        mesh = Mesh(self.cfg.guide.topology_path, self.device)
        self.faces = mesh.faces[None,...].to(self.device)
        self.smplx_verts = self.verts = self.v_template
        self.smplx_faces = self.faces = self.smplx.faces_tensor.to(self.device)
        self.smplx_vt = mesh.vt.to(self.device)
        self.smplx_ft = mesh.ft.to(self.device)

    def view_mesh(self):
        self.posed_verts, _, _, joints_transform, self.curr_vertices_transform, shape_offsets, pose_offsets = \
                self.smplx(full_pose=self.view_pose, shape_params=self.shape, expression_params=self.expression, return_T=True)
        posed_verts_reshape = self.normalize_mesh(self.posed_verts.clone())
        self.v_template_view = posed_verts_reshape.squeeze().to(self.device)
        self.mesh_verts_view = self.v_template_view
        self.mesh_faces_view = self.smplx_faces
        self.mesh_vt_view = self.smplx_vt
        self.mesh_ft_view = self.smplx_ft
        self.curr_offsets = shape_offsets + pose_offsets

    def normalize_mesh(self, vertices, target_scale=3, dx=0, dy=-0.78, dz=0):
        verts = vertices[0]
        verts *= target_scale
        verts[:, 2] += dz
        verts[:, 1] += dy
        verts[:, 0] += dx
        return verts.unsqueeze(0)

    def init_pose(self):
        pose = torch.zeros([55,3], dtype=torch.float32, device=self.device)
        # put arms down
        angle = 60*np.pi/180.
        pose[16, 2] = -angle
        pose[17, 2] = angle
        init_pose = self.quaternion_to_angle_axis(self.euler_to_quaternion(pose))[None, ...]
        init_pose = (self.batch_axis2matrix(init_pose.reshape(-1,3))).reshape(1, 55, 3, 3)
        return init_pose
    
    def batch_axis2matrix(self, theta):
        return self.quaternion_to_rotation_matrix(self.angle_axis_to_quaternion(theta))

    def angle_axis_to_quaternion(self, angle_axis: torch.Tensor):
        if not torch.is_tensor(angle_axis):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(angle_axis)))
        if not angle_axis.shape[-1] == 3:
            raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}"
                            .format(angle_axis.shape))
        # unpack input and compute conversion
        a0: torch.Tensor = angle_axis[..., 0:1]
        a1: torch.Tensor = angle_axis[..., 1:2]
        a2: torch.Tensor = angle_axis[..., 2:3]
        theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2
        theta: torch.Tensor = torch.sqrt(theta_squared)
        half_theta: torch.Tensor = theta * 0.5
        mask: torch.Tensor = theta_squared > 0.0
        ones: torch.Tensor = torch.ones_like(half_theta)
        k_neg: torch.Tensor = 0.5 * ones
        k_pos: torch.Tensor = torch.sin(half_theta) / theta
        k: torch.Tensor = torch.where(mask, k_pos, k_neg)
        w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)
        quaternion: torch.Tensor = torch.zeros_like(angle_axis)
        quaternion[..., 0:1] += a0 * k
        quaternion[..., 1:2] += a1 * k
        quaternion[..., 2:3] += a2 * k
        return torch.cat([w, quaternion], dim=-1)

    def mesh_view_pose(self):
        pose = torch.zeros([55,3], dtype=torch.float32, device=self.device) 
        # put down arms
        angle = 60*np.pi/180.
        pose[16, 2] = -angle
        pose[17, 2] = angle
        mesh_pose = self.quaternion_to_angle_axis(self.euler_to_quaternion(pose))[None, ...] 
        # whether use jaw pose, jaw pose has a large impact on expression
        if self.cfg.guide.use_jaw_pose:
            mesh_pose[:,22] = self.jaw_pose
        mesh_pose = (self.batch_axis2matrix(mesh_pose.reshape(-1,3))).reshape(1, 55, 3, 3)
        if self.cfg.guide.animate:
            mesh_pose[:,12,:,:] = self.pose
        return mesh_pose
    
    def quaternion_to_rotation_matrix(self, quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
        B = quat.size(0)
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
        return rotMat
    
    def quaternion_to_angle_axis(self, quaternion: torch.Tensor):
        """Convert quaternion vector to angle axis of rotation.
        Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
        Args:
            quaternion (torch.Tensor): tensor with quaternions.
        Return:
            torch.Tensor: tensor with angle axis of rotation.
        Shape:
            - Input: :math:`(*, 4)` where `*` means, any number of dimensions
            - Output: :math:`(*, 3)`
        Example:
            >>> quaternion = torch.rand(2, 4)  # Nx4
            >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
        """
        if not torch.is_tensor(quaternion):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(quaternion)))

        if not quaternion.shape[-1] == 4:
            raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                            .format(quaternion.shape))
        # unpack input and compute conversion
        q1: torch.Tensor = quaternion[..., 1]
        q2: torch.Tensor = quaternion[..., 2]
        q3: torch.Tensor = quaternion[..., 3]
        sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

        sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
        cos_theta: torch.Tensor = quaternion[..., 0]
        two_theta: torch.Tensor = 2.0 * torch.where(
            cos_theta < 0.0,
            torch.atan2(-sin_theta, -cos_theta),
            torch.atan2(sin_theta, cos_theta))

        k_pos: torch.Tensor = two_theta / sin_theta
        k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta).to(quaternion.device)
        k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

        angle_axis: torch.Tensor = torch.zeros_like(quaternion).to(quaternion.device)[..., :3]
        angle_axis[..., 0] += q1 * k
        angle_axis[..., 1] += q2 * k
        angle_axis[..., 2] += q3 * k
        return angle_axis

    def euler_to_quaternion(self, r):
        x = r[..., 0]
        y = r[..., 1]
        z = r[..., 2]
        z = z/2.0
        y = y/2.0
        x = x/2.0
        cz = torch.cos(z)
        sz = torch.sin(z)
        cy = torch.cos(y)
        sy = torch.sin(y)
        cx = torch.cos(x)
        sx = torch.sin(x)
        quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
        quaternion[..., 0] += cx*cy*cz - sx*sy*sz
        quaternion[..., 1] += cx*sy*sz + cy*cz*sx
        quaternion[..., 2] += cx*cz*sy - sx*cy*sz
        quaternion[..., 3] += cx*cy*sz + sx*cz*sy
        return quaternion
    
    def init_logger(self):
        logger.remove()
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def mesh_paint(self, hybrid=None):
        logger.info('Starting training paint^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['mesh_val'], self.eval_renders_path, hybrid=False)
        self.mesh_model.train()
        pbar = tqdm(total=len(self.dataloaders['mesh_train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.rgb_output_list = []
        for data in self.dataloaders['mesh_train']:
            self.paint_step += 1
            pbar.update(1)
            self.paint_viewpoint(data)
            self.evaluate(self.dataloaders['mesh_val'], self.eval_renders_path, hybrid=False)
            self.mesh_model.train()
        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval(hybrid=False)
        logger.info('\tDone!')

    def nerf_hybrid(self, hybrid=None):
        logger.info('Starting training nerf ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['nerf_val'], self.eval_renders_path, hybrid=True)
        self.nerf_model.train()
        pbar = tqdm(total=self.cfg.optim.iters, initial=self.nerf_train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.nerf_train_step < self.cfg.optim.iters: 
            for data in self.dataloaders['nerf_train']:
                self.nerf_train_step += 1
                pbar.update(1)
                # render textured mesh with save camera view
                theta = data['theta']
                phi = -data['phi']
                radius = data['radius']
                phi = phi - np.deg2rad(self.cfg.render.front_offset)
                camera_projection_fovy = data['fov']
                from src.models.nerf_utils import NeRFType
                if self.cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
                    outputs = self.mesh_model.render(camera_projection_fovy=camera_projection_fovy, theta=theta, phi=phi, radius=radius,
                                                    dims=(self.cfg.nerfrender.train_mesh_rgb_dim, self.cfg.nerfrender.train_mesh_rgb_dim), background='white')
                    if self.cfg.nerfrender.train_mesh_depth_dim == self.cfg.nerfrender.train_mesh_rgb_dim:
                        mesh_depth = outputs['depth_nonnormal'].detach()
                    else:
                        outputs_depth = self.mesh_model.render(camera_projection_fovy=camera_projection_fovy, theta=theta, phi=phi, radius=radius,
                                                        dims=(self.cfg.nerfrender.train_mesh_depth_dim, self.cfg.nerfrender.train_mesh_depth_dim), background='white') #camera_projection_fovy=camera_projection_fovy,
                        mesh_depth = outputs_depth['depth_nonnormal'].detach()
                if self.cfg.nerfrender.nerf_type == NeRFType['fine_tunning']:
                    outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                    dims=(self.cfg.nerfrender.fine_tunning_w, self.cfg.nerfrender.fine_tunning_h), background='white')
                    mesh_depth = outputs['depth_nonnormal'].detach()
                from src.models.nerf_utils import NeRFType
                if self.cfg.nerfrender.nerf_type != NeRFType['latent']:
                    mesh_latent = outputs['image'].detach()
                    rgb = outputs['image'].detach()
                if self.cfg.nerfrender.nerf_type == NeRFType['latent']:
                    rgb = outputs['image']
                    mesh_latent = self.nerf_diffusion.encode_imgs(rgb).detach()

               # train nerf model 
                self.nerf_optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    loss = self.train_render(data, mesh_latent, mesh_depth) 
                self.nerf_scaler.scale(loss).backward()
                self.nerf_scaler.step(self.nerf_optimizer)
                self.nerf_scaler.update()

                # evaluate save video and ckpt
                if self.nerf_train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['nerf_val'], self.eval_renders_path, hybrid=True)
                    self.nerf_model.train()
        
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval(hybrid=True)
        logger.info('\tDone!')
               
    def train_render(self, data: Dict[str, Any], mesh_latent, mesh_depth): 
        rays_o, rays_d = data['rays_o'], data['rays_d']
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W'] 
        if self.cfg.optim.start_shading_iter is None or self.nerf_train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1
        step = self.nerf_train_step
        outputs = self.nerf_model.render(step, self.posed_verts, self.mesh_verts_view, self.mesh_faces_view,self.curr_vertices_transform, self.curr_offsets, self.canonical_transform, self.canonical_offsets, self.lbs_weights, mesh_latent, mesh_depth, rays_o, rays_d, staged=True, perturb=True, 
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, train=True) 
        pred_latent = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if pred_latent.shape[1] == 4:
            pred_rgb = self.nerf_diffusion.decode_latents(pred_latent)
        if pred_latent.shape[1] == 3:
            pred_rgb = pred_latent.clamp(0,1)
        pred_ws = outputs['weights_sum'].reshape(B, H, W)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']
            text_z = self.nerf_text_z[dirs]
        else:
            text_z = self.nerf_text_z            

        # sds loss
        if self.cfg.optim.sds_loss == True:
            text_guidance_scale = self.cfg.guide.nerf_guidance_scale
            step = self.nerf_train_step  
            sds_loss = self.nerf_diffusion.train_step(text_z, pred_latent, text_guidance_scale)
            loss = sds_loss
        else:
            loss = 0.0
        # sparsity loss
        if 'sparsity_loss' in self.losses:
            sparsity_loss = self.cfg.optim.lambda_sparsity * self.losses['sparsity_loss'](pred_ws)
            loss += sparsity_loss
        # shape loss
        if 'shape_loss' in self.losses:
            shape_loss = self.cfg.optim.lambda_shape * self.losses['shape_loss'](outputs['xyzs'], outputs['alphas_init'], self.nerf_train_step, )
            loss += shape_loss
        # blip loss
        if 'blip_sim_loss' in self.losses:
            blip_sim_loss = self.cfg.optim.lambda_blip_sim * self.losses['blip_sim_loss'](pred_rgb, data['dir'])
            loss += blip_sim_loss
        # mask loss
        if 'mask_loss' in self.losses:
            mask_loss = self.cfg.optim.lambda_mask * self.losses['mask_loss'](pred_rgb, outputs['weights_except_last'])
            loss += mask_loss
        return loss

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = True, hybrid=None):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        self.nerf_model.eval()
        save_path.mkdir(exist_ok=True)
        if save_as_video:
            all_preds = []
            total_vis = []
        for i, data in enumerate(dataloader):
            if not hybrid:
                preds, textures = self.eval_render(data)
                pred = tensor2numpy(preds[0])
                all_preds.append(cv2.resize(pred, (1024,1024)))
            if hybrid:
                # render mesh with same camera view
                theta = data['theta']
                phi = -data['phi']
                radius = data['radius']
                phi = phi - np.deg2rad(self.cfg.render.front_offset) 
                camera_projection_fovy = data['fov']
                from src.models.nerf_utils import NeRFType
                if self.cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
                    outputs = self.mesh_model.render(camera_projection_fovy=camera_projection_fovy, theta=theta, phi=phi, radius=radius,
                                                    dims=(self.cfg.nerfrender.eval_mesh_rgb_dim, self.cfg.nerfrender.eval_mesh_rgb_dim), background='white')
                    if self.cfg.nerfrender.eval_mesh_depth_dim == self.cfg.nerfrender.eval_mesh_rgb_dim:
                        mesh_depth = outputs['depth_nonnormal'].detach()
                    else:
                        outputs_depth = self.mesh_model.render(camera_projection_fovy=camera_projection_fovy, theta=theta, phi=phi, radius=radius,
                                                        dims=(self.cfg.nerfrender.eval_mesh_depth_dim, self.cfg.nerfrender.eval_mesh_depth_dim), background='white') 
                        mesh_depth = outputs_depth['depth_nonnormal'].detach()
                if self.cfg.nerfrender.nerf_type == NeRFType['fine_tunning']:
                    outputs = self.mesh_model.render(camera_projection_fovy=camera_projection_fovy,  theta=theta, phi=phi, radius=radius,
                                                    dims=(self.cfg.nerfrender.fine_tunning_w, self.cfg.nerfrender.fine_tunning_h), background='white')
                    mesh_depth = outputs['depth_nonnormal'].detach()
                if self.cfg.nerfrender.nerf_type == NeRFType['rgb']:
                    rgb = outputs['image']
                    mesh_rgb = outputs['image'].detach()
                if self.cfg.nerfrender.nerf_type == NeRFType['latent']:
                    rgb = outputs['image']
                    mesh_rgb = self.nerf_diffusion.encode_imgs(rgb).detach()
                if self.cfg.nerfrender.nerf_type == NeRFType['fine_tunning']:
                    rgb = outputs['image']
                    mesh_rgb = outputs['image'].detach()

                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    preds, pred_weights_except_last = self.eval_nerf_render(data, mesh_rgb, mesh_depth) 
                pred = tensor2numpy(preds[0])
                z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)  
                mesh_img = utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.3) 
                mesh_img = mesh_img*outputs['mask']+(1-outputs['mask'])
                mask_weight_except_last = F.interpolate(pred_weights_except_last.permute(0,3,1,2), (preds.shape[1], preds.shape[1]), mode='bilinear', antialias=True) 
                nerf_img = (mask_weight_except_last*preds.permute(0,3,1,2)+(1-mask_weight_except_last)*mesh_img).permute(0,2,3,1)  
                if not self.cfg.guide.animate:
                    hybrid_image = (mask_weight_except_last*preds.permute(0,3,1,2)+(1-mask_weight_except_last)*rgb).permute(0,2,3,1)  
                    total_img = torch.cat((hybrid_image, (F.interpolate(rgb, (preds.shape[1],preds.shape[1]), mode='bilinear', antialias=True).permute(0,2,3,1)), nerf_img), axis=2)  
                    total_vi = tensor2numpy(total_img[0])
                if self.cfg.guide.animate:
                    hybrid_image = (((mask_weight_except_last*preds.permute(0,3,1,2)))+(1-mask_weight_except_last)*rgb.flip(3)).permute(0,2,3,1)
                    total_vi = tensor2numpy(hybrid_image[0])
                if self.cfg.nerfrender.skinning_eval == True:
                    skinning_img = torch.cat((hybrid_image,(F.interpolate(rgb, (preds.shape[1],preds.shape[1]), mode='bilinear', antialias=True).permute(0,2,3,1)), mask_weight_except_last.permute(0,2,3,1)), axis=2)                
                    import os
                    from torchvision import transforms
                    unloader = transforms.ToPILImage()
                    image = skinning_img.permute(0,3,1,2).cpu().clone() 
                    image = image.squeeze(0)  
                    image = unloader(image)
                    save_skinning = save_path / "skinning/"
                    if not os.path.exists(save_skinning):
                        make_path(save_skinning)
                    image.save(save_skinning / f'{i:04d}.png')
                total_vis.append(total_vi)
                
        if not hybrid:
            # save texture map
            texture = tensor2numpy(textures[0])
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            if not hybrid:
                all_preds = np.stack(all_preds, axis=0)
                dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.cfg.log.exp_name}_step_{self.paint_step:05d}_{name}.mp4", video,
                                                            fps=10, quality=8, macro_block_size=1)
                dump_vid(all_preds, 'eval')
            if hybrid:
                total_vis = np.stack(total_vis, axis=0)
                dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.cfg.log.exp_name}_step_{self.nerf_train_step:05d}_{name}.mp4", video,
                                                            fps=10, quality=8, macro_block_size=1)
                dump_vid(total_vis, 'eval')
        logger.info('Done!')
        return total_vis

    def full_eval(self, output_dir: Path = None, hybrid=None):
        if output_dir is None:
            output_dir = self.final_renders_path
        if not hybrid:
            self.evaluate(self.dataloaders['mesh_val_large'], output_dir, save_as_video=True, hybrid=hybrid)
        if hybrid and not self.cfg.guide.animate:
            self.evaluate(self.dataloaders['nerf_val_large'], output_dir, save_as_video=True, hybrid=hybrid)
        if hybrid and self.cfg.guide.animate:
            frame = self.evaluate(self.dataloaders['nerf_val'], output_dir, save_as_video=True, hybrid=hybrid)
            return frame        
        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")
            self.mesh_model.export_mesh(save_path)
            logger.info(f"\tDone!")
    
    def paint_viewpoint(self, data: Dict[str, Any]):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')
        # set background image
        if self.cfg.guide.use_background_color:
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else:
            background = F.interpolate(self.back_img.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)
        # render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
        render_cache = outputs['render_cache']
        rgb_render_raw = outputs['image']
        depth_render = outputs['depth']
        # render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        # render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]
        self.log_train_image(rgb_render, 'rendered_input')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir'] 
            text_z = self.mesh_paint_text_z[dirs]
        else:
            text_z = self.mesh_paint_text_z

        # calculate trimap
        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'],
                                                                        step=self.paint_step)
        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return
        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
        self.log_train_image(rgb_render * refine_mask, name='refine_regions')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, np.maximum(min_h,0):np.maximum(max_h,0), np.maximum(min_w,0):np.maximum(max_w,0)]
        cropped_rgb_render = crop(rgb_render)
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
        self.log_train_image(cropped_rgb_render, name='cropped_input')

        checker_mask = None
        if self.paint_step > 1:
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
                                 'checkerboard_input')
        self.mesh_diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        cropped_rgb_output, steps_vis = self.mesh_diffusion.img2img_step(text_z, cropped_rgb_render.detach(),
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.mesh_paint_guidance_scale,
                                                                    strength=1.0, update_mask=cropped_update_mask,
                                                                    fixed_seed=self.seed,
                                                                    check_mask=checker_mask,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps,
                                                                    step=self.paint_step)
        self.log_train_image(cropped_rgb_output, name='direct_output')
        self.log_diffusion_steps(steps_vis)
        cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone()
        rgb_output[:, :, np.maximum(min_h,0):np.maximum(max_h,0), np.maximum(min_w,0):np.maximum(max_w,0)] = cropped_rgb_output
        self.log_train_image(rgb_output, name='full_output')
        self.rgb_output_list.append(rgb_output)
        # Project back
        object_mask = outputs['mask']
        fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache, refine_mask=refine_mask, step=self.paint_step)
        self.log_train_image(fitted_pred_rgb, name='fitted')
        return

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        dim = self.cfg.render.eval_grid_size 
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image'] 
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask
        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        return rgb_render, texture_rgb 

    def eval_nerf_render(self, data,mesh_rgb,mesh_depth, bg_color=None, perturb=False): 
        rays_o = data['rays_o'] 
        rays_d = data['rays_d'] 
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        step = self.nerf_train_step
        outputs = self.nerf_model.render(step, self.posed_verts, self.mesh_verts_view, self.mesh_faces_view, self.curr_vertices_transform, self.curr_offsets, self.canonical_transform, self.canonical_offsets, self.lbs_weights, mesh_rgb, mesh_depth, rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, train=False) # bg_color=bg_color, mesh_depth,parsing_hair_mask,mesh_hair_mask, 
        pred_depth = outputs['depth'].reshape(B, H, W) 
        pred_weights_except_last = outputs['weights_except_last'].reshape(B, H, W)
        if self.nerf_model.latent_mode:
            pred_latent = outputs['image'].reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
            if self.cfg.log.skip_rgb:
                # When rendering in a size that is too large for decoding
                pred_rgb = torch.zeros(B, H, W, 3, device=pred_latent.device)
            else:
                pred_rgb = self.nerf_diffusion.decode_latents(pred_latent).permute(0, 2, 3, 1).contiguous()     
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)
        pred_depth = pred_depth.unsqueeze(-1).repeat(1, 1, 1, 3) 
        pred_weights_except_last = pred_weights_except_last.unsqueeze(-1).repeat(1, 1, 1, 3)
        shading = 'normal'
        step = self.nerf_train_step
        return pred_rgb, pred_weights_except_last

    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor,
                         step=None):
        # calculate the difference between colored mesh and initial mesh as mask
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1) 
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)
        # erode and dilate are very important for avoid view conflict and artifacts
        # erode, eliminate some sparse points in the textured parts to avoid regenerating artifacts
        exact_generate_mask = torch.from_numpy(
            cv2.erode(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((10, 10), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)
        # dilate, inflate the boundaries of mask to avoid view conflict
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()
        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        refine_mask = torch.zeros_like(update_mask)
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        if self.cfg.guide.initial_texture is None:
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            refine_mask[z_normals < 0.4] = 0
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        # erode and dilate are very important for avoid view conflict and artifacts
        # erode, eliminate some sparse points in the textured parts to avoid regenerating artifacts
        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0) 
        # dilate, inflate the boundaries of mask to avoid view conflict
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1
        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0
        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)# torch.Size([1, 3, 1200, 1200])
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask
            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'shaded_input')
            self.log_train_image(trimap_vis, 'trimap')
        return update_mask, generate_mask, refine_mask

    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard, (512, 512))
        checker_mask = F.interpolate(update_mask_inner, (512, 512))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (512, 512))
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor, refine_mask: torch.Tensor, step=None): 
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        render_update_mask = object_mask.clone()
        render_update_mask[update_mask == 0] = 0
        blurred_render_update_mask = render_update_mask
        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)
        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0
        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
            blurred_render_update_mask[z_was_better] = 0
        render_update_mask = blurred_render_update_mask
        self.log_train_image(rgb_output * render_update_mask, 'project_back_input')
        # Update the normals
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])
        self.optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.paint_lr, betas=(0.9, 0.99),
                                     eps=1e-15)
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            self.optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
            rgb_render = outputs['image']
            loss = {}
            mask_img = render_update_mask
            mask = mask_img.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss['image'] = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean() + (
                    (masked_pred - masked_pred.detach()).pow(2) * (1 - masked_mask)).mean()
            # add symmetry loss
            if self.cfg.optim.flip_loss >0 and self.cfg.render.n_views % 2 ==0:
                if self.paint_step > 2 and self.paint_step % 2==1 and self.paint_step < self.cfg.render.n_views:
                    rgb_output_flip = self.rgb_output_list[self.paint_step-2]
                    rgb_output_flip = torch.flip(rgb_output_flip,[3]).detach()
                    masked_target = rgb_output_flip.reshape(1, rgb_output_flip.shape[1], -1)[:, :, mask > 0]
                    loss['symmetry']= (((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean() + ((masked_pred - masked_pred.detach()).pow(2) * (1 - masked_mask)).mean())*self.cfg.optim.flip_loss
                if self.paint_step == 1 or self.paint_step == self.cfg.render.n_views:
                    rgb_output_flip = torch.flip(rgb_output,[3]).detach()
                    masked_target = rgb_output_flip.reshape(1, rgb_output_flip.shape[1], -1)[:, :, mask > 0]
                    loss['symmetry']= (((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean() + ((masked_pred - masked_pred.detach()).pow(2) * (1 - masked_mask)).mean())*self.cfg.optim.flip_loss
            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            loss['normal']= (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            all_loss = 0.
            losses_key = loss.keys()
            for key in losses_key:
                all_loss = all_loss + loss[key]
            loss['all_loss'] = all_loss
            all_loss.backward()
            self.optimizer.step()
        return rgb_render, current_z_normals

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            tensor = np.maximum(tensor,0)
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        if 'model' not in checkpoint_dict:
            self.nerf_model.load_state_dict(checkpoint_dict)
            logger.info("loaded model.")
            return
        missing_keys, unexpected_keys = self.nerf_model.load_state_dict(checkpoint_dict['nerf_model'], strict=False)
        missing_keys, unexpected_keys = self.mesh_model.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")
        if model_only:
            return
        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.nerf_train_step = checkpoint_dict['nerf_train_step'] + 1
        logger.info(f"load at step {self.nerf_train_step}")
        if self.nerf_optimizer and 'nerf_optimizer' in checkpoint_dict:
            try:
                self.nerf_optimizer.load_state_dict(checkpoint_dict['nerf_optimizer'])
                logger.info("loaded nerf_optimizer.")
            except:
                logger.warning("Failed to load nerf_optimizer.")
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")
        if self.nerf_scaler and 'scaler' in checkpoint_dict:
            try:
                self.nerf_scaler.load_state_dict(checkpoint_dict['scaler'])
                logger.info("loaded scaler.")
            except:
                logger.warning("Failed to load scaler.")

    def save_checkpoint(self, full=False):
        name = f'step_{self.nerf_train_step:06d}'
        state = {
            'nerf_train_step': self.nerf_train_step,
            'paint_step': self.paint_step,
            'checkpoints': self.past_checkpoints,
        }

        if full:
            state['nerf_optimizer'] = self.nerf_optimizer.state_dict()
            from src.models.nerf_utils import NeRFType
            if self.cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
                state['optimizer'] = self.optimizer.state_dict()
            state['scaler'] = self.nerf_scaler.state_dict()

        state['nerf_model'] = self.nerf_model.state_dict()
        state['model'] = self.mesh_model.state_dict()
        file_path = f"{name}.pth"
        self.past_checkpoints.append(file_path)
        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)
        torch.save(state, self.ckpt_path / file_path)