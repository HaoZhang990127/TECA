from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    # Grid size for rendering during painting
    train_grid_size: int = 1200
    # Grid size of evaluation
    eval_grid_size: int = 1024
    # Painting camera radius
    radius: float = 1.3
    # Set [0,overhead_range] as the overhead region
    overhead_range: float = 40
    # Define the front angle region
    front_range: float = 80
    # The front offset, use to rotate shape from code
    front_offset: float = 0.0
    # Number of views to use
    n_views: int = 10
    # Theta value for rendering during training
    base_theta: float = 80
    # Additional views to use before rotating around shape
    views_before: List[Tuple[float,float]] = field(default_factory=list)
    # Additional views to use after rotating around shape
    views_after: List[Tuple[float,float]] = field(default_factory=list)
    # Whether to alternate between the rotating views from the different sides
    alternate_views: bool = True

@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Geometry text prompt 
    geometry_text: str = 'a fat European woman with bob cut hairstyle, front view, ID photo, headshot, portrait, a virtual eigen photo, high resolution, high quality, high details'
    # Paint text prompt 
    paint_text: str = "a fat European woman wearing a long sleeveless floral print dress with bob cut hairstyle, {} view, a virtual eigen photo, high resolution, high quality, high details, uniform lighting, studio lighting"
    # NeRF training text prompt 
    nerf_text: str = "a woman with bob cut hairstyle, {} view, no watermark, a virtual eigen photo, high resolution, high quality, high details, uniform lighting, studio lighting"
    # Blip fine tunning text prompt, normally it is same with nerf_text, but you can also change color of asset in here by use blip_text
    blip_text: str = "a woman with bob cut hairstyle, {} view, no watermark, a virtual eigen photo, high resolution, high quality, high details, uniform lighting, studio lighting"
    # Geometry negative text prompt 
    geometry_neg_text: str = 'face out of frame, out of frame, ugly, tiling, poorly drawn face, out of frame, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face'
    # Paint negative text prompt
    paint_neg_text: str = 'grey image, grey color, teeth, glasses, earrings, necklace, dirty, tattoos, floating artifacts, watermark, blur, glasses, ugly, tiling, out of frame, disfigured, deformed, bad anatomy, watermark, signature, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face'
    # NeRF training negative text prompt 
    nerf_neg_text: str = 'face, beard, eyes, mouth, clothes, earrings, necklace, dirty, blur, floating artifacts, cartoon, watermark, low resolution, ugly, tiling, poorly drawn face, out of frame, disfigured, deformed, bad anatomy, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face'
    # Blip fine tunning negative text prompt
    blip_neg_text: str = 'watermark, blur, face, beard, eyes, mouth, clothes, earrings, necklace, low resolution, dirty, tattoos, floating artifacts, ugly, tiling, poorly drawn face'
    # Whether to align the geometry with text prompt
    geometry_init: bool = True
    # Whether use jaw pose, it will influence expression, if you need a big expression change like screaming, it should be set to true
    use_jaw_pose: bool = False
    # Append direction to text prompts
    append_direction: bool = True
    # Scale of mesh
    shape_scale: float = 4
    # Move mesh in x direction
    dx: float = 0.0
    # Move mesh in y direction
    dy: float = -1.2
    # Move mesh in z direction
    dz: float = 0.0
    # Texture image resolution
    texture_resolution: int = 1024
    # Texture mapping interpolation mode from texture image, options: 'nearest', 'bilinear', 'bicubic'
    texture_interpolation_mode: str= 'bilinear'
    # Guidance scale for diffusion in mesh painting process
    mesh_paint_guidance_scale: float = 7.5
    # Guidance scale for score distillation in nerf process, sds need a large guidance scale
    nerf_guidance_scale: int = 90
    # Use inpainting in relevant iterations
    use_inpainting: bool = True
    # The texture before editing
    reference_texture: Optional[Path] = None
    # The edited texture
    initial_texture: Optional[Path] = None
    # Whether to use background color or image
    use_background_color: bool = False
    # Background image to use
    background_img: str = 'textures/brick_wall.png'
    # Threshold for defining refine regions
    z_update_thr: float = 0.4
    # Clipseg mask threshold for 0/1 mask
    mask_thr: float = 0.6
    # Some more strict masking for projecting back
    strict_projection: bool = True
    # Define the proximal distance allowed for shape loss
    proximal_surface: float = 0.3
    # Text prompt for clipseg to choose the target region
    clipseg_text: str = 'hair'
    # Erode to eliminate sparse points
    maskerode: int = 2
    # Dilate to extend mask
    maskdilate: int = 1
    # Smplx topology path
    topology_path: str = './data/smplx.obj'
    # Smplx model path
    smplx_model_path: str = './data/SMPLX_NEUTRAL_lockedhead_exp.npz'
    # Smplx joint path
    extra_joint_path: str = './data/smplx_extra_joints.yaml'
    j14_regressor_path: str = './data/SMPLX_to_J14.pkl'
    # Dimension of smplx shape
    n_shape: int = 300
    # Dimension of smplx expression
    n_exp: int = 100
    # whether animate
    animate: bool = False
    
@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Seed for experiment
    seed: int = 0
    # Learning rate for painting
    paint_lr: float = 1e-2
    # Learning rate for nerf
    nerf_lr: float = 1e-3
    # Time step range for diffusion model
    min_timestep: float = 0.02
    # Time step range for diffusion model
    max_timestep: float = 0.98
    # For diffusion model init
    no_noise: bool = False
    # Whether calculate flip loss
    flip_loss: float = 0.0
    # Resume from checkpoint
    resume: bool = False
    # Load existing model
    ckpt: Optional[str] = None
    # Total iters
    iters: int = 3000
    # Loss scale for alpha entropy
    lambda_sparsity: float = 5e-4
    # Loss scale for mesh-guidance
    lambda_shape: float = 0 # 5e-6
    # Loss scale for blip
    lambda_blip_sim: float = 1
    # Loss scale for mask
    lambda_mask: float = 0.1
    # Start shading at this iteration
    start_shading_iter: Optional[int] = None
    # Use amp mixed precision training
    fp16: bool = True
    # Whether use sds loss in nerf training
    sds_loss: bool = True
    # Blip ckpt path
    blip_path: str = './src/training/BLIP/model_base.pth'
    # Blip model
    blip_vit: str = 'base'

@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str
    # Experiment root path
    exp_root_path: str
    # Experiment root
    exp_root: Path = Path('experiments/')
    # Experiment save path
    exp_path_save: Path = Path('experiments/')
    # Checkpoint save path
    ckpt_root: Path = Path('experiments/')
    # How many steps between save
    save_interval: int = 300
    # Run only for evaluation
    eval_only: bool = False
    # Number of views to sample for eval during training
    eval_size: int = 20
    # Number of views to sample for eval after training
    full_eval_size: int = 50
    # Export a mesh
    save_mesh: bool = False
    # Whether to show intermediate diffusion visualizations
    vis_diffusion_steps: bool = False
    # Whether to log intermediate images
    log_images: bool = True
    # Skip stable diffusion decoder visualization
    skip_rgb: bool = False
    # Number of max past checkpoints to keep
    max_keep_ckpts: int = 3

    @property
    def exp_dir(self) -> Path:
        return self.exp_path_save / f'{self.exp_name}'
    @property   
    def exp_path(self) -> Path:
        return self.exp_root / f'{self.exp_root_path}'
    @property
    def ckpt_dir(self) -> Path:
        return self.ckpt_root / self.exp_name

from src.models.nerf_utils import NeRFType
@dataclass
class NeRFRenderConfig:
    """ Parameters for the NeRF Renderer """
    # Number of steps sampled when rendering
    num_steps: int = 64
    # Number of upsampled steps per ray
    upsample_steps: int = 32
    # Batch size of rays at inference
    max_ray_batch: int = 4096
    # If positive, use a background model at sphere(bg_radius)
    bg_radius: float =  0.0
    # Render width for training
    train_w: int = 64
    # Render height for training
    train_h: int = 64
    # Render width for inference
    eval_w: int = 64
    # Render height for inference
    eval_h: int = 64
    # Assume the scene is bounded in box(-bound,bount)
    bound: float = 2
    # Minimum near distance for camera
    min_near: float = 0.1
    # Training camera radius range
    radius_range: Tuple[float, float] = (1.5, 2)
    # Training camera fovy range
    fovy_range: Tuple[float, float] = (40, 70)
    # Set [0,angle_overhead] as the overhead region
    angle_overhead: float = 30
    # Define the front angle region
    angle_front: float = 70
    # Which NeRF backbone to use
    backbone: str = 'grid'
    # Define the nerf output type
    nerf_type: NeRFType = NeRFType['latent']
    # The offset of far value in query process
    far_offset: float = 0.0
    # Theta_range (40-100)
    theta_range: Tuple[float, float] = (40, 100)
    # Phi_range (0-360)
    phi_range: Tuple[float, float] = (0, 360)
    # Eval camera theta
    eval_theta: float = 80
    # Eval camera radius
    eval_radius: float = 1.3
    # Rendered mesh depth dim in hybrid tarining stage
    train_mesh_depth_dim: int = 64
    # Rendered mesh rgb dim in hybrid tarining stage
    train_mesh_rgb_dim: int = 512
    # Rendered mesh depth dim in hybrid eval stage
    eval_mesh_depth_dim: int = 64
    # Rendered mesh rgb dim in hybrid eval stage
    eval_mesh_rgb_dim: int = 512
    # Whether use mesh depth as index of ray and mesh interaction
    mesh_depth_as_index: bool = True
    # Weight for last point in nerf, the nerf last point use mesh color
    last_point_weight: float = 10
    # Whether use background augment
    bg_aug: bool = False
    # Whether perturb z values in nerf querying process
    perturb_z_vals: bool = False
    # Whether perturb sigma values in nerf training process
    perturb_sigma: bool = False
    # The image width in fine tunning stage
    fine_tunning_w: int = 256
    # The image high in fine tunning stage
    fine_tunning_h: int = 256
    # The number of near point for skinning
    k_neigh: int = 1
    # Eval for skinning transfer
    skinning_eval: bool = False
    # Whether to randomly jitter sampled camera
    jitter_pose: bool = False

@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    nerfrender: NeRFRenderConfig = field(default_factory=NeRFRenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)