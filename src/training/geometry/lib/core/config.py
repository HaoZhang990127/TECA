from yacs.config import CfgNode as CN
import os

cfg = CN()
workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 

# ---------------------------------------------------------------------------- #
# Options for losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.reg_pose_l1 = 1e-4

# ---------------------------------------------------------------------------- #
# Options for Body model (SMPLX)
# ---------------------------------------------------------------------------- #
# cfg.data_dir = os.path.join(workdir, 'data')
cfg.data_dir = os.path.join('./data')
cfg.model = CN()
cfg.model.use_highres=True
cfg.model.add_inner_mouth=True
cfg.model.highres_path = os.path.join(cfg.data_dir, 'subdiv_level_1') 
cfg.model.inner_mouth_path = os.path.join(cfg.data_dir, 'head_template_mesh_mouth.obj') 
cfg.model.topology_path = os.path.join(cfg.data_dir, 'SMPL_X_template_FLAME_uv.obj') 
cfg.model.topology_smplxtex_path = os.path.join(cfg.data_dir, 'smplx_tex.obj')
cfg.model.topology_smplx_hand_path = os.path.join(cfg.data_dir, 'smplx_hand.obj')
cfg.model.smplx_model_path = os.path.join(cfg.data_dir, 'SMPLX_NEUTRAL_2020.npz')
cfg.model.face_mask_path = os.path.join(cfg.data_dir, 'uv_face_mask.png')
cfg.model.face_eye_mask_path = os.path.join(cfg.data_dir, 'uv_face_eye_mask.png')
cfg.model.tex_path = os.path.join(cfg.data_dir, 'FLAME_albedo_from_BFM.npz')
cfg.model.extra_joint_path = os.path.join(cfg.data_dir, 'smplx_extra_joints.yaml')
cfg.model.j14_regressor_path = os.path.join(cfg.data_dir, 'SMPLX_to_J14.pkl')
cfg.model.flame2smplx_cached_path = os.path.join(cfg.data_dir, 'flame2smplx_tex_1024.npy')
cfg.model.smplx_tex_path = os.path.join(cfg.data_dir, 'smplx_tex.png')
cfg.model.mano_ids_path = os.path.join(cfg.data_dir, 'MANO_SMPLX_vertex_ids.pkl')
cfg.model.flame_ids_path = os.path.join(cfg.data_dir, 'SMPL-X__FLAME_vertex_ids.npy')
cfg.model.flame_vertex_masks_path = os.path.join(cfg.data_dir, 'FLAME_masks.pkl')
cfg.model.fr_model_path = os.path.join(cfg.data_dir, 'resnet50_ft_weight.pkl')
cfg.model.flame_tex_path = os.path.join(cfg.data_dir, 'FLAME_texture.npz')
# use higher dimension for face tracking
cfg.model.uv_size = 512
cfg.model.n_shape = 300
cfg.model.n_tex = 100
cfg.model.n_exp = 100
cfg.model.n_body_cam = 3
cfg.model.n_head_cam = 3
cfg.model.n_hand_cam = 3
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uvtex_type = 'SMPLX' # FLAME or SMPLX
cfg.model.use_tex = True # whether to use flame texture model

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

