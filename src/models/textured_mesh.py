import os
import os,sys
os.chdir(sys.path[0])
import kaolin as kal
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from .mesh import Mesh
from .mesh_render import Renderer
from src.configs.train_config import GuideConfig

def build_cotan_laplacian_torch(points_tensor: torch.Tensor, tris_tensor: torch.Tensor) -> np.ndarray:
    tris, points = tris_tensor.cpu().numpy(), points_tensor.cpu().numpy()
    a, b, c = (tris[:, 0], tris[:, 1], tris[:, 2]) 
    A = np.take(points, a, axis=1)
    B = np.take(points, b, axis=1)
    C = np.take(points, c, axis=1)

    eab, ebc, eca = (B - A, C - B, A - C)
    eab = eab / np.linalg.norm(eab, axis=0)[None, :]
    ebc = ebc / np.linalg.norm(ebc, axis=0)[None, :]
    eca = eca / np.linalg.norm(eca, axis=0)[None, :]

    alpha = np.arccos(-np.sum(eca * eab, axis=0))
    beta = np.arccos(-np.sum(eab * ebc, axis=0))
    gamma = np.arccos(-np.sum(ebc * eca, axis=0))

    wab, wbc, wca = (1.0 / np.tan(gamma), 1.0 / np.tan(alpha), 1.0 / np.tan(beta))
    rows = np.concatenate((a, b, a, b, b, c, b, c, c, a, c, a), axis=0)
    cols = np.concatenate((a, b, b, a, b, c, c, b, c, a, a, c), axis=0)
    vals = np.concatenate((wab, wab, -wab, -wab, wbc, wbc, -wbc, -wbc, wca, wca, -wca, -wca), axis=0)
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(points.shape[1], points.shape[1]), dtype=float).tocsc()
    return L 

def build_graph_laplacian_torch(tris_tensor: torch.Tensor) -> np.ndarray:
    tris = tris_tensor.cpu().numpy()
    n_verts = tris.max() + 1
    v2v = [[] for _ in range(n_verts)]
    for face in tris:
        for i in face:
            for j in face:
                if i != j:
                    if j not in v2v[i]:
                        v2v[i].append(j)

    valency = [len(x) for x in v2v]
    I, J, vals = [], [], []
    for i in range(n_verts):
        I.append(i)
        J.append(i)
        vals.append(1)
        for neighbor in v2v[i]:
            I.append(i)
            J.append(neighbor)
            vals.append(-1 / valency[i])
    L = sparse.csr_matrix((vals, (I, J)), shape=(n_verts, n_verts))
    return L

def eigen_problem(Lap, k=20, e=0.0) -> (torch.Tensor, torch.Tensor):
    shift = 1e-4
    eigenvalues, eigenvectors = eigsh(
        Lap + shift * scipy.sparse.eye(Lap.shape[0]),
        k=k + 1, which='LM', sigma=e, tol=1e-3) 
    eigenvalues += shift
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]
    return torch.from_numpy(eigenvalues).float(), torch.from_numpy(eigenvectors.T).float()

def choose_multi_modal(n: int, k: int):
    interval_length = n // k 
    n_intervals = n / interval_length
    if n_intervals == int(n_intervals):
        n_intervals = int(n_intervals)
    else:
        n_intervals = int(n_intervals) + 1
    chosen_numbers = []
    for i in range(n_intervals):
        current_interval_length = min(interval_length, n - i * interval_length)
        chosen_numbers.append(np.random.choice(current_interval_length) + i * interval_length)
    return chosen_numbers

class TexturedMeshModel(nn.Module):
    def __init__(self,
                 opt: GuideConfig,
                 render_grid_size=1024,
                 texture_resolution=1024,
                 initial_texture_path=None,
                 cache_path=None,
                 device=torch.device('cpu'),
                 vertices=None,
                 face=None,
                 vt=None,
                 ft=None,):
        super().__init__()
        self.device = device
        self.opt = opt
        self.dy = self.opt.dy
        self.dx = self.opt.dx
        self.dz = self.opt.dz
        self.mesh_scale = self.opt.shape_scale
        self.mesh_vertices = vertices
        self.mesh_faces = face
        self.mesh_vt = vt
        self.mesh_ft = ft
        self.texture_resolution = texture_resolution
        if initial_texture_path is not None:
            self.initial_texture_path = initial_texture_path
        else:
            self.initial_texture_path = self.opt.initial_texture
        self.cache_path = cache_path
        self.num_features = 3
        self.renderer = Renderer(device=self.device, dim=(render_grid_size, render_grid_size),
                                 interpolation_mode=self.opt.texture_interpolation_mode)
        self.env_sphere = self.init_meshes()
        self.default_color = [0.8,0.1,0.8] 
        self.background_sphere_colors, self.texture_img = self.init_paint()
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))
        if self.opt.reference_texture:
            base_texture = torch.Tensor(np.array(Image.open(self.opt.reference_texture).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
            change_mask = (
                    (base_texture.to(self.device) - self.texture_img).abs().sum(axis=1) > 0.1).float()
            with torch.no_grad():
                self.meta_texture_img[:, 1] = change_mask

        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.mesh_vt.unsqueeze(0),
            self.mesh_ft.long()).detach() 

        self.n_eigen_values = 20
        self._L = None
        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def L(self) -> np.ndarray:
        if self._L is None:
            self._L = build_cotan_laplacian_torch(self.mesh_vertices.T, self.mesh_faces)
        return self._L

    def eigens(self, k: int, e: float) -> (torch.Tensor, torch.Tensor):
        if self._eigenvalues is None or self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = eigen_problem(self.L, k, e) 
            self._eigenvalues, self._eigenvectors = \
                self._eigenvalues.to(self.device), self._eigenvectors.to(self.device)
        return self._eigenvalues, self._eigenvectors

    @staticmethod
    def normalize_vertices(vertices: torch.Tensor, mesh_scale: float = 1.0, dy: float = 0.0) -> torch.Tensor:
        vertices -= vertices.mean(dim=0)[None, :]
        vertices /= vertices.norm(dim=1).max()
        vertices *= mesh_scale
        vertices[:, 1] += dy
        return vertices

    def spectral_augmentations(self, vertices: torch.Tensor) -> torch.Tensor:
        eigen_values, basis_functions = self.eigens(self.n_eigen_values, 0.0)
        basis_functions /= basis_functions.max(dim=-1)[0][:, None] - basis_functions.min(dim=-1)[0][:, None]
        chosen_basis_function = choose_multi_modal(basis_functions.shape[0], 2)
        coeffs = torch.zeros(basis_functions.shape[0]).to(self.device)
        signs = ((torch.rand(len(chosen_basis_function), device=self.device) > 0.5).float() - 0.5) * 2
        coeffs[chosen_basis_function] = signs
        reconstructed = coeffs @ basis_functions.float()
        directions = vertices / torch.norm(vertices, dim=1)[:, None]
        deformed_v = vertices + 0.25 * reconstructed[:, None] * directions
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def axis_augmentations(self, vertices: torch.Tensor, stretch_factor: float = 1.6, squish_factor: float = 0.7):
        axis_indices = np.arange(0, 3) 
        axis_indices = np.random.permutation(axis_indices)
        stretch_axis = axis_indices[0]
        squish_axis = axis_indices[1]
        deformed_v = vertices.clone()
        deformed_v[:, stretch_axis] *= stretch_factor
        deformed_v[:, squish_axis] *= squish_factor
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def augment_vertices(self):
        verts = self.mesh_vertices.clone()
        if np.random.rand() < 0.5:
            verts = self.spectral_augmentations(verts)
        if np.random.rand() < 0.5:
            verts = self.axis_augmentations(verts) 
        return verts 

    def init_meshes(self, env_sphere_path='data/env_sphere.obj'):
        env_sphere = Mesh(env_sphere_path, self.device)
        return env_sphere

    def zero_meta(self):
        with torch.no_grad():
            self.meta_texture_img[:] = 0

    def init_paint(self, num_backgrounds=1):
        # random color face attributes for background sphere
        init_background_bases = torch.rand(num_backgrounds, 3).to(self.device)
        modulated_init_background_bases_latent = init_background_bases[:, None, None, :] * 0.8 + 0.2 * torch.randn(
            num_backgrounds, self.env_sphere.faces.shape[0],
            3, self.num_features, dtype=torch.float32).cuda()
        background_sphere_colors = nn.Parameter(modulated_init_background_bases_latent.cuda())
        # texture initialization for optimization
        if self.initial_texture_path is not None:
            texture = torch.Tensor(np.array(Image.open(self.initial_texture_path).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0#torch.Size([1, 3, 1024, 1024]) 
        else:
            texture = torch.ones(1, 3, self.texture_resolution, self.texture_resolution).cuda() * torch.Tensor(
                self.default_color).reshape(1, 3, 1, 1).cuda() 
        texture_img = nn.Parameter(texture) # the init texture will be optimized
        return background_sphere_colors, texture_img

    def invert_color(self, color: torch.Tensor) -> torch.Tensor:
        # inverse linear approx to find latent
        A = self.linear_rgb_estimator.T
        regularizer = 1e-2
        pinv = (torch.pinverse(A.T @ A + regularizer * torch.eye(4).cuda()) @ A.T)
        if len(color) == 1 or type(color) is torch.Tensor:
            init_color_in_latent = color @ pinv.T
        else:
            init_color_in_latent = pinv @ torch.tensor(
                list(color)).float().to(A.device)
        return init_color_in_latent

    def change_default_to_median(self):
        diff = (self.texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        default_mask = (diff < 0.1).float().unsqueeze(0)
        median_color = self.texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(axis=1)
        with torch.no_grad():
            self.texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)
    
    def init_texture_map(self):
        cache_path = self.cache_path
        if cache_path is None:
            cache_exists_flag = False
        else:
            vt_cache, ft_cache = cache_path / 'vt.pth', cache_path / 'ft.pth'
            cache_exists_flag = vt_cache.exists() and ft_cache.exists()

        if self.mesh_vt is not None and self.mesh_ft is not None \
                and self.mesh_vt.shape[0] > 0 and self.mesh_ft.min() > -1:
            vt = self.mesh_vt.cuda()
            ft = self.mesh_ft.cuda()
        elif cache_exists_flag:
            vt = torch.load(vt_cache).cuda()
            ft = torch.load(ft_cache).cuda()
        return vt, ft

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        return [self.background_sphere_colors, self.texture_img, self.meta_texture_img]

    @torch.no_grad()
    def export_mesh(self, path):
        v, f = self.mesh_vertices, self.mesh_faces.int()
        h0, w0 = 256, 256
        ssaa, name = 1, ''
        # v, f: torch Tensor
        v_np = v.cpu().numpy() 
        f_np = f.cpu().numpy() 
        colors = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        colors = colors[0].cpu().detach().numpy()
        colors = (colors * 255).astype(np.uint8)
        vt_np = self.mesh_vt.detach().cpu().numpy()
        ft_np = self.mesh_ft.detach().cpu().numpy()
        colors = Image.fromarray(colors)
        if ssaa > 1:
            colors = colors.resize((w0, h0), Image.LINEAR)
        colors.save(os.path.join(path, f'{name}albedo.png'))
        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'{name}mesh.obj')
        mtl_file = os.path.join(path, f'{name}mesh.mtl')
        logger.info('writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {name}mesh.mtl \n')
            logger.info('writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
            logger.info('writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                # fp.write(f'vt {v[0]} {1 - v[1]} \n')
                fp.write(f'vt {v[0]} {v[1]} \n')
            logger.info('writing faces {f_np.shape}')
            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {name}albedo.png \n')

    def render_mask(self, theta=None, phi=None, radius=None, background=None,
            use_meta_texture=False, render_cache=None, use_median=False, dims=None, texture_path=None):
        obj_path = self.opt.shape_path
        mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=True)
        vertices = mesh.vertices.to(self.device)
        faces = mesh.faces.to(self.device)
        ft = mesh.face_uvs_idx
        vt = mesh.uvs
        verts = vertices
        center = verts.mean(dim=0)
        verts = verts - center
        scale = torch.max(torch.norm(verts, p=2, dim=1))
        verts = verts / scale
        verts *= self.mesh_scale
        verts[:, 2] += self.dz
        verts[:, 1] += self.dy
        verts[:, 0] += self.dx
        vertices = verts
        face_attributes = kal.ops.mesh.index_vertices_by_faces(
            vt.unsqueeze(0),
            ft.long()).detach()
        texture_img_path = texture_path
        from skimage.io import imread
        texture_img = imread(texture_img_path).astype(np.float32)
        texture_img = torch.from_numpy(texture_img/255).to(self.device)
        texture_img = texture_img.unsqueeze(0).permute(0,3,1,2) #
        texture_img = F.interpolate(texture_img, (self.opt.texture_resolution, self.opt.texture_resolution))
        render_mask = self.renderer.render_mask(vertices,
                                                faces,
                                                face_attributes,
                                                texture_img,
                                                elev=theta,
                                                azim=phi,
                                                radius=radius,
                                                look_at_height=0,
                                                render_cache=render_cache,
                                                dims=dims,
                                                background_type=None)
        return render_mask

    def render(self, camera_projection_fovy=60, theta=None, phi=None, radius=None, background=None,
               use_meta_texture=False, render_cache=None, use_median=False, dims=None):
        if render_cache is None:
            assert theta is not None and phi is not None and radius is not None
        background_sphere_colors = self.background_sphere_colors[
            torch.randint(0, self.background_sphere_colors.shape[0], (1,))] 

        if use_meta_texture:
            texture_img = self.meta_texture_img
        else:
            texture_img = self.texture_img
        augmented_vertices = self.mesh_vertices

        if use_median:
            diff = (texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
                self.device)).abs().sum(axis=1) 
            default_mask = (diff < 0.1).float().unsqueeze(0)
            median_color = texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(
                axis=1)
            texture_img = texture_img.clone()
            with torch.no_grad():
                texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)#
        background_type = 'none'
        use_render_back = False
        if background is not None and type(background) == str:
            background_type = background
            use_render_back = True
        pred_features, mask, depth, normals, render_cache, depth_nonnormal = self.renderer.render_single_view_texture(augmented_vertices, 
                                                                                                     self.mesh_faces,
                                                                                                     self.face_attributes,
                                                                                                     texture_img,
                                                                                                     camera_projection_fovy = camera_projection_fovy,
                                                                                                     elev=theta,
                                                                                                     azim=phi,
                                                                                                     radius=radius,
                                                                                                     look_at_height=0,
                                                                                                     render_cache=render_cache,
                                                                                                     dims=dims,
                                                                                                     background_type=background_type)
        mask = mask.detach() 
        if use_render_back:
            pred_map = pred_features 
            pred_back = pred_features
        else:
            if background is None:
                pred_back, _, _ = self.renderer.render_single_view(self.env_sphere,
                                                                   background_sphere_colors,
                                                                   elev=theta,
                                                                   azim=phi,
                                                                   camera_projection_fovy = camera_projection_fovy,
                                                                   radius=radius,
                                                                   dims=dims,
                                                                   look_at_height=0, calc_depth=False) #look_at_height=self.dy
            elif len(background.shape) == 1:
                pred_back = torch.ones_like(pred_features) * background.reshape(1, 3, 1, 1)
            else:
                pred_back = background
            pred_map = pred_back * (1 - mask) + pred_features * mask 
        if not use_meta_texture:
            pred_map = pred_map.clamp(0, 1)
            pred_features = pred_features.clamp(0, 1)
        return {'image': pred_map, 'mask': mask, 'background': pred_back,
                'foreground': pred_features, 'depth': depth, 'normals': normals, 'render_cache': render_cache,
                'texture_map': texture_img, 'depth_nonnormal': depth_nonnormal}

    def draw(self, theta, phi, radius, target_rgb, camera_projection_fovy=60):
        # failed attempt to draw on the texture image
        uv_features, face_idx = self.renderer.project_uv_single_view(self.mesh_vertices,
                                                                     self.mesh_faces,
                                                                     self.face_attributes,
                                                                     elev=theta,
                                                                     camera_projection_fovy = camera_projection_fovy,
                                                                     azim=phi,
                                                                     radius=radius,
                                                                     look_at_height=0) # look_at_height=self.dy
        unique_face_idx = torch.unique(face_idx)
        print('')