import kaolin as kal
import torch
import numpy as np
from .util import vertex_normals
from .util import face_vertices

class Renderer:
    # from https://github.com/threedle/text2mesh
    def __init__(self, device, dim=(224, 224), interpolation_mode='nearest'):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'
        camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)
        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()

    @staticmethod
    def get_camera_from_view(elev, azim, r=3.0, look_at_height=0.0):
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)
        pos = torch.tensor([x, y, z]).unsqueeze(0).to(torch.float32)
        look_at = torch.zeros_like(pos).to(torch.float32)
        look_at[:, 1] = look_at_height
        direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(torch.float32)
        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
        return camera_proj

    def normalize_depth(self, depth_map):
        object_mask = depth_map != 0
        min_val = 0.5
        depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
                depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val
        return depth_map

    def render_single_view(self, mesh, face_attributes, elev=0, azim=0, camera_projection_fovy=60, radius=2, look_at_height=0.0,calc_depth=True,dims=None, background_type='none'):
        dims = self.dim if dims is None else dims
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height).to(self.device) 
        camera_projection = kal.render.camera.generate_perspective_projection(torch.deg2rad(torch.tensor(camera_projection_fovy))).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), camera_projection, camera_transform=camera_transform)
        if calc_depth:
            depth_map_nonnormal, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_depth(depth_map_nonnormal)
        else:
            depth_map = torch.zeros(1,64,64,1)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)
        mask = (face_idx > -1).float()[..., None]
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        if background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)
        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2)

    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, camera_projection_fovy=60, elev=0, azim=0, radius=2,
                                   look_at_height=0.0, dims=None, background_type='none', render_cache=None,use_overhead=True):
        dims = self.dim if dims is None else dims
        if render_cache is None:
            camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                    look_at_height=look_at_height).to(self.device) 
            camera_projection = kal.render.camera.generate_perspective_projection(torch.deg2rad(torch.tensor(camera_projection_fovy))).to(self.device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), camera_projection, camera_transform=camera_transform)
            depth_map_nonormal, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_depth(depth_map_nonormal) 
            
            padded_vertices = torch.nn.functional.pad(
                            verts.to(self.device), (0, 1), mode='constant', value=1.
                            )
            vertices_camera = (padded_vertices @ camera_transform)
            vertices_image = kal.render.camera.perspective_camera(vertices_camera, self.camera_projection)
            ## DEBUG: vertex normals rather than face normals
            vertices_camera_normals = vertex_normals(vertices_camera, faces.to(self.device)[None, ...])
            ## vertices to face vertices
            face_normals = face_vertices(vertices_camera_normals, faces.to(self.device)[None, ...])
            normals_image, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_normals)
            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr)
            uv_features = uv_features.detach()
        else:
            # logger.info('Using render cache')
            normals_image, face_normals, uv_features, face_idx, depth_map, depth_map_nonormal = render_cache['normals_image'], render_cache['face_normals'], render_cache['uv_features'], render_cache['face_idx'], render_cache['depth_map'], render_cache['depth_map_nonormal']
        mask = (face_idx > -1).float()[..., None]
        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode) 
        image_features = image_features * mask
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        elif background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)
        render_cache = {'normals_image':normals_image, 'uv_features':uv_features, 'face_normals':face_normals,'face_idx':face_idx, 'depth_map':depth_map, 'depth_map_nonormal':depth_map_nonormal}
        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2),\
               depth_map.permute(0, 3, 1, 2), normals_image.permute(0, 3, 1, 2), render_cache, depth_map_nonormal.permute(0, 3, 1, 2)

    def project_uv_single_view(self, verts, faces, uv_face_attr, elev=0, azim=0, radius=2,
                               look_at_height=0.0,camera_projection_fovy=60, dims=None, background_type='none'):
        # project the vertices and interpolate the uv coordinates
        dims = self.dim if dims is None else dims
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        camera_projection = kal.render.camera.generate_perspective_projection(torch.deg2rad(torch.tensor(camera_projection_fovy))).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), camera_projection, camera_transform=camera_transform)
        uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                          face_vertices_image, uv_face_attr)
        return face_vertices_image, face_vertices_camera, uv_features, face_idx

    def project_single_view(self, verts, faces, elev=0, azim=0, radius=2,camera_projection_fovy=60,
                               look_at_height=0.0):
        # only project the vertices
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        camera_projection = kal.render.camera.generate_perspective_projection(torch.deg2rad(torch.tensor(camera_projection_fovy))).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), camera_projection, camera_transform=camera_transform)
        return face_vertices_image

    def render_mask(self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2,camera_projection_fovy=60,
                                    look_at_height=0.0, dims=None, background_type='none', render_cache=None):
            dims = self.dim if dims is None else dims
            camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                    look_at_height=look_at_height).to(self.device) 
            camera_projection = kal.render.camera.generate_perspective_projection(torch.deg2rad(torch.tensor(camera_projection_fovy))).to(self.device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), camera_projection, camera_transform=camera_transform)
            face_vertices_camera = face_vertices_camera.to(device=self.device)
            face_vertices_image = face_vertices_image.to(device=self.device)
            uv_face_attr = uv_face_attr.to(device=self.device)
            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr)
            uv_features = uv_features.detach()
            image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
            return image_features