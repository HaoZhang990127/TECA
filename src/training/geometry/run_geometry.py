from ipaddress import ip_address
import os
from re import S
from socket import IP_DEFAULT_MULTICAST_LOOP
import numpy as np
from glob import glob
import pickle
import torch
import argparse
from skimage.io import imread, imsave
from tqdm import tqdm
import torch.nn.functional as F
import pytorch3d
from pytorch3d.renderer import (look_at_view_transform, FoVOrthographicCameras, PointsRasterizationSettings,
                                PointsRenderer, PulsarPointsRenderer, PointsRasterizer, AlphaCompositor,
                                NormWeightedCompositor)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)
from lib.utils.deca_render import SRenderY
from lib.utils import util, lossfunc
from lib.model.smplx import SMPLX
from lib.model.FLAME import FLAMETex, texture_flame2smplx
from lib.core.config import cfg, update_cfg
from lib.utils import rotation_converter
from lib.render.mesh_helper import pytorch3d_rasterize, render_shape, render_texture
from torch.utils.data import Dataset, ConcatDataset
cfg.model.smplx_model_path = os.path.join(cfg.data_dir, 'SMPLX_NEUTRAL_lockedhead_exp.npz')
model_cfg = cfg.model
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import face_alignment

class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', sample_step=10):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        else:
            print(f'please check the test path: {testpath}')
            exit()
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]
        h, w, _ = image.shape
        out = self.model.get_landmarks(image)
        image = image/255.
        lmk = torch.from_numpy(out[0])
        lmk = lmk / h
        lmk = lmk * 2. - 1
        lmk = np.concatenate([lmk, np.ones([lmk.shape[0], 1])], axis=-1)
        return {
                'imagepath': imagepath,
                'imagename': imagename,
                'image': torch.tensor(image.transpose(2,0,1)).float(),
                'lmk': lmk
                }

class SMPLX_optimizer(torch.nn.Module):
    def __init__(self, device='cuda:0', image_size=512, light_type='SH'):
        super(SMPLX_optimizer, self).__init__()
        self.cfg = cfg
        self.device = device
        self.image_size = image_size
        self.light_type = light_type
        self._setup_model()
        self._setup_renderer()
        self._setup_loss_weight()

    def _setup_model(self):
        ## smplx model
        self.smplx = SMPLX(model_cfg).to(self.device)
        self.flametex = FLAMETex(model_cfg).to(self.device)
        self.verts = self.smplx.v_template
        self.faces = self.smplx.faces_tensor
        ## iris index
        self.idx_iris = [9503, 10049]  #right, left

    def _setup_renderer(self):
        ## setup raterizer
        uv_size = 1024
        topology_path = self.cfg.model.topology_path
        # cache data for smplx texture
        self.smplx_texture = imread(self.cfg.model.smplx_tex_path) / 255.
        self.cached_data = np.load(self.cfg.model.flame2smplx_cached_path, allow_pickle=True, encoding='latin1').item()
        self.render = SRenderY(self.image_size,
                               obj_filename=topology_path,
                               uv_size=uv_size,
                               rasterizer_type='pytorch3d').to(self.device)
        mask = imread(self.cfg.model.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_eye_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)
        # face region mask in flame texture map
        mask = imread(self.cfg.model.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)
        ########### silhouette rendering
        ## camera
        R = torch.eye(3).unsqueeze(0)
        T = torch.zeros([1, 3])
        batch_size = 1
        self.cameras = pytorch3d.renderer.cameras.FoVOrthographicCameras(R=R.expand(batch_size, -1, -1),
                                                                         T=T.expand(batch_size, -1),
                                                                         znear=0.0).to(self.device)
        blend_params = BlendParams(sigma=1e-7, gamma=1e-4)
        raster_settings = RasterizationSettings(image_size=self.image_size,
                                                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                                                faces_per_pixel=50,
                                                bin_size=0)
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras,
                                                                          raster_settings=raster_settings),
                                                shader=SoftSilhouetteShader(blend_params=blend_params))

    def forward_model(self, batch, returnMask=False, returnRendering=False, returnNormal=False):
        ''' forward SMPLX model
        Args:
            batch: dict, batch data
                'beta': [B, n_shape(200)], shape parameters
                'exp': [B, n_exp(100)], expression parameters
                'full_pose': [B, n_pose(55), 3, 3], pose parameters, in Rotatation matrix format
                'cam': [B, 3], camera parameters, [scale, tx, ty], use orthographic projection
            returnMask: bool, whether to return mask
            returnRendering: bool, whether to return rendering
            returnNormal: bool, whether to return normal
        Returns:
            opdict: dict, output dict
        '''
        opdict = {}
        verts, landmarks, joints = self.smplx(shape_params=batch['beta'],
                                              expression_params=batch['exp'],
                                              full_pose=batch['full_pose'])
        cam = batch['cam']
        trans_verts = util.batch_orth_proj(verts, cam)
        pred_lmk = util.batch_orth_proj(landmarks, cam)[:, :, :2]
        # convert smpl-x landmarks to flame landmarks (right order)
        pred_lmk = torch.cat([pred_lmk[:, -17:], pred_lmk[:, :-17]], dim=1)
        opdict['verts'] = verts
        opdict['trans_verts'] = trans_verts
        opdict['pred_lmk'] = pred_lmk
        # render mask for silhouette loss
        batch_size = verts.shape[0]
        faces = self.faces.unsqueeze(0).expand(batch_size,-1,-1)
        if returnMask:
            trans_verts_mask = trans_verts.clone()
            trans_verts_mask[:,:,:2] = -trans_verts_mask[:,:,:2]
            trans_verts_mask[:,:,-1] = -trans_verts_mask[:,:,-1] + 50
            mesh = Meshes(
                verts = trans_verts_mask,
                faces = faces
            )
            mesh_mask = self.silhouette_renderer(meshes_world=mesh).permute(0, 3, 1, 2)[:,3:]
            opdict['mesh_mask'] = mesh_mask
        # render image for image loss
        if returnRendering:
            # calculate normal for shading
            normal_verts = trans_verts.clone()
            normal_verts[..., 0] = -normal_verts[..., 0]
            normals = util.vertex_normals(normal_verts, faces)                    
            trans_verts[..., -1] = trans_verts[..., -1] + 50
            albedo = self.flametex(batch['tex'])
            rendering_out = self.render(verts,
                                trans_verts,
                                albedo,
                                lights=batch['light'],
                                light_type=self.light_type,
                                given_normal=normals)
            opdict['image'] = rendering_out['images']
            opdict['albedo_image'] = rendering_out['albedo_images']
            opdict['shading_image'] = rendering_out['shading_images']
        return opdict

    def _setup_loss_weight(self):
        loss_cfg = cfg.loss
        loss_cfg.lmk = 1.
        loss_cfg.eyed = 2.
        loss_cfg.lipd = 0.
        # mask
        loss_cfg.inside_mask = 1.
        loss_cfg.mesh_mask = 1.
        # image
        loss_cfg.image = 2.
        loss_cfg.albedo = 2.
        self.loss_cfg = loss_cfg

    def configure_optimizers(self, batch):
        ###--- optimizer for training nerf model.
        init_beta = torch.zeros([1, model_cfg.n_shape]).float().to(self.device)
        self.register_parameter('beta', torch.nn.Parameter(init_beta))
        init_tex = torch.zeros([1, model_cfg.n_tex]).float().to(self.device)
        self.register_parameter('tex', torch.nn.Parameter(init_tex))
        init_exp = torch.zeros([1, model_cfg.n_exp]).float().to(self.device)
        self.register_parameter('exp', torch.nn.Parameter(init_exp))
        lmk = batch['lmk'][0]
        ymin = lmk[:, 1].min()
        ymax = lmk[:, 1].max()
        yrange = ymax - ymin
        init_scale = 7 * yrange 
        init_cam = torch.tensor([
            init_scale,
            0.02,
            1
        ]).float().reshape(1, 3).to(self.device)
        self.register_parameter('cam', torch.nn.Parameter(init_cam.to(self.device)))

        init_exp = torch.zeros([1, model_cfg.n_exp]).float().to(self.device)
        init_light = torch.zeros([1, 9, 3]).float().to(self.device)
        init_full_pose = torch.zeros([55, 3]).float().to(self.device) + 0.00001
        init_full_pose[0, 0] = np.pi
        ## init shoulder
        init_full_pose[16,2] = -np.pi*60/180
        init_full_pose[17,2] = np.pi*60/180
        init_full_pose = rotation_converter.batch_euler2axis(init_full_pose)[None, ...]
        self.init_full_pose = init_full_pose
        init_global_pose = init_full_pose[:,0,:]
        self.register_parameter('global_pose', torch.nn.Parameter(init_global_pose.to(self.device)))
        self.register_parameter('jaw_pose', torch.nn.Parameter(torch.zeros_like(init_global_pose).to(self.device)+0.00001))
        parameters = [{'params': [self.beta, self.exp, self.tex, self.cam, self.global_pose, self.jaw_pose], 'lr': 1e-3}]
        self.optimizer = torch.optim.Adam(params=parameters)  

    def optimize(self, batch, iters=2000,
                    vis_step=200,
                    batch_size=1,
                    lmk_only=True,
                    use_mask = False,
                    use_rendering = False,
                    use_normal = False):
        ''' optimize the pose and shape parameters of the model, using lmk loss only
        # global: 0, neck: 12, head: 15, leftarm: 16, rightarm: 17, jaw: 22, lefteye: 23, righteye: 24
        '''
        ####----- setup optimizer 
        self.configure_optimizers(batch)
        vispath = args.path+'/geometry'
        os.makedirs(vispath, exist_ok=True)
        image = batch['image']
        for iter in tqdm(range(iters)):
            batch['beta'] = self.beta.expand(batch_size, -1)
            batch['exp'] = self.exp.expand(batch_size, -1)
            batch['tex'] = self.tex.expand(batch_size, -1)
            batch['cam'] = self.cam.expand(batch_size, -1)
            full_pose = self.init_full_pose.clone()
            full_pose[:,0] = self.global_pose
            full_pose[:,22] = self.jaw_pose
            full_pose = rotation_converter.batch_axis2matrix(full_pose.reshape(-1,3)).reshape(batch_size, 55, 3, 3)
            batch['full_pose'] = full_pose
            opdict = self.forward_model(batch, returnMask=use_mask, returnRendering=use_rendering, returnNormal=use_normal)

            losses = {}
            if 'iris' in batch.keys():
                pred_iris = opdict['trans_verts'][:, self.idx_iris, :2]
                pred_lmk = torch.cat([opdict['pred_lmk'], pred_iris], dim=1)
                gt_lmk = torch.cat([batch['lmk'][:, :], batch['iris'][:, :]], dim=1)
            else:
                pred_lmk = opdict['pred_lmk']
                gt_lmk = batch['lmk']
            # lmk loss, use confidence, if confidence is 0, then ignore this points (e.g. especially for iris points)
            losses['lmk'] = lossfunc.batch_kp_2d_l1_loss(gt_lmk, pred_lmk)*self.loss_cfg.lmk
            
            # eye distance loss from DECA
            if self.cfg.loss.eyed > 0.:
                losses['eyed'] = lossfunc.eyed_loss(pred_lmk[:,:68,:2], gt_lmk[:,:68,:2])*self.cfg.loss.eyed
            if self.cfg.loss.lipd > 0.:
                losses['lipd'] = lossfunc.lipd_loss(pred_lmk[:,:68,:2], gt_lmk[:,:68,:2])*self.cfg.loss.lipd
            if use_mask:
                losses['mesh_inside_mask'] = (torch.relu(opdict['mesh_mask'] - batch['mask'])).abs().mean()*self.loss_cfg.inside_mask
                mesh_mask = opdict['mesh_mask']
                non_skin_mask = 1-mesh_mask.detach()
                hair_only = non_skin_mask*batch['hair_mask']
                mesh_mask = mesh_mask + hair_only
                opdict['mesh_mask'] = mesh_mask
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.loss_cfg.mesh_mask
            if use_rendering:
                losses['image'] = (batch['face_mask'] *(batch['image'] - opdict['image'])).abs().mean() * self.loss_cfg.image
                losses['image_albedo'] = (batch['face_mask'] *
                                        (batch['image'] - opdict['albedo_image'])).abs().mean() * self.loss_cfg.albedo
            losses['reg_shape'] = (torch.sum(batch['beta']**2) / 2).mean() * 1e-5
            losses['reg_exp'] = (torch.sum(batch['exp']**2) / 2).mean() * 1e-5
            losses['reg_tex'] = (torch.sum(batch['tex']**2) / 2).mean() * 5e-5
            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            if vis_step < 1000 and iter % vis_step == 0:
                loss_info = f"Iter: {iter}/{iters}: "
                for k, v in losses.items():
                    loss_info = loss_info + f'{k}: {v:.6f}, '
                print(loss_info)
                visdict = {
                    'inputs': image,
                    'lmk_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=True),
                    'lmk_pred': util.tensor_vis_landmarks(image, pred_lmk, isScale=True)
                }
                # render shape
                faces = self.smplx.faces_tensor
                shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                        faces=faces.expand(batch_size, -1, -1),
                                        image_size=image.shape[-1],
                                        background=image)
                visdict['shape'] = shape_image
                for key in opdict.keys():
                    if 'image' in key:
                        visdict[key] = opdict[key]
                util.visualize_grid(visdict, os.path.join(vispath, f'{iter:06}.png'), return_gird=False)
                
        visdict = {
                    'inputs': image,
                    'lmk_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=True),
                    'lmk_pred': util.tensor_vis_landmarks(image, pred_lmk, isScale=True)
                }
        # render shape
        faces = self.smplx.faces_tensor
        shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                faces=faces.expand(batch_size, -1, -1),
                                image_size=image.shape[-1],
                                background=image)
        visdict['shape'] = shape_image
        for key in opdict.keys():
            if 'image' in key:
                visdict[key] = opdict[key]
        util.visualize_grid(visdict, os.path.join(vispath, f'{iter:06}.png'), return_gird=False)
        ## save obj
        print("Save obj")
        # add uv from toplogy size
        obj_filename = model_cfg.topology_smplxtex_path
        _, uvcoords, faces, uvfaces = util.load_obj(obj_filename)
        batch['full_pose'][:, 22] = batch['full_pose'][:, 23] 
        batch['full_pose'][:, 0] = batch['full_pose'][:, 23] 
        verts, landmarks, joints = self.smplx(shape_params=batch['beta'],expression_params=batch['exp'],
                                              full_pose=batch['full_pose'])

        ## save beta
        print("Save beta")
        beta = batch['beta'].detach().cpu().numpy()
        exp = batch['exp'].detach().cpu().numpy()
        jaw = self.jaw_pose.detach().cpu().numpy()
        np.save(os.path.join(vispath, f'shape.npy'), beta)
        np.save(os.path.join(vispath, f'expression.npy'), exp)
        np.save(os.path.join(vispath, f'jaw_pose.npy'), jaw)
        util.write_obj(os.path.join(vispath, f'{iter:06}.obj'),
                    verts[0].detach().cpu().numpy(), 
                    faces.cpu().numpy(),
                    texture=image[0].detach().cpu().numpy(),
                    uvcoords=uvcoords.cpu().numpy(),
                    uvfaces=uvfaces.cpu().numpy())
        print(os.path.join(vispath, f'{iter:06}.obj'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath', type=str, 
                        default='/home/hzhang/DECA/image/high_quality/A woman crying.png')
    parser.add_argument('--geometry_text', type=str, default='a smiling woman')
    parser.add_argument('--geometry_neg_text', type=str, default='face out of frame, out of frame, ugly')
    parser.add_argument('--path', default='geometry', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    args = parser.parse_args()

    # generate image based on text prompt
    with torch.no_grad():
        for i in range(100):  # avoid stable diffusion not generating a proper face image for the first time, if the landmarks not totaly in image, stable diffusion will generate again
            from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
            model_id = "stabilityai/stable-diffusion-2"
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            prompt = args.geometry_text
            negative_prompt = args.geometry_neg_text
            image = pipe(prompt, negative_prompt=negative_prompt).images[0]
            # determine if the face is in the middle of the picture
            import face_alignment
            model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            out = model.get_landmarks(np.array(image))
            if out != None and np.min(out) >0 and np.max(out) <768:
                os.makedirs(args.path+'/geometry', exist_ok=True)
                vispath = args.path+'/geometry'
                image.save(vispath+ '/image.png')
                image_path = vispath+ '/image.png'
                break
    ## load dataset
    testdata = TestData(image_path, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
    optimizer = SMPLX_optimizer(args.device, light_type='SH')

    # batch for cluster running
    dataloader = torch.utils.data.DataLoader(testdata,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 drop_last=False)
        
    for batch in dataloader:
        util.move_dict_to_device(batch, args.device)
        optimizer.optimize(batch)
        print('out_optimize')
    print('outer_optimize')
# python smplx_detail_tracking_single.py  --batch_idx 500 --batch_num 1 --batch_size 1  --iters 2000