import pyrallis
import os
import sys
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
from src.configs.train_config import TrainConfig
from src.training.trainer import TECA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from subprocess import run
import torch
import numpy as np

@pyrallis.wrap('./configs/a_fat_European_woman_with_bob_cut_hairstyle_refine.yaml')
def main(cfg: TrainConfig):
    # aligning smplx geometry and text prompt
    from src.models.nerf_utils import NeRFType
    if cfg.guide.geometry_init and not cfg.log.eval_only and cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
        geometry_text = cfg.guide.geometry_text
        geometry_neg_text = cfg.guide.geometry_neg_text
        path = cfg.log.exp_dir
        run(['python', 'src/training/geometry/run_geometry.py', '--geometry_text', geometry_text, '--geometry_neg_text', geometry_neg_text, '--path', path], check=True, env=os.environ.copy()) 
    
    # load smplx geometry parameters from run_geometry results, user can also use the shape, expression, and pose parameters defined by themself
    shape_path = os.path.join(cfg.log.exp_path, f'geometry/shape.npy')
    shape = torch.from_numpy(np.load(shape_path))  # user can define shape parameters in here
    expression_path = os.path.join(cfg.log.exp_path, f'geometry/expression.npy')
    expression = torch.from_numpy(np.load(expression_path)) # user can define expression parameters in here
    jaw_pose_path = os.path.join(cfg.log.exp_path, f'geometry/jaw_pose.npy')
    jaw_pose = torch.from_numpy(np.load(jaw_pose_path))  # user can define jaw pose in here, jaw pose affects expression
    pose = None
    trainer = TECA(cfg, shape, expression, jaw_pose, pose)
    if cfg.log.eval_only:
        trainer.full_eval(hybrid=True)
    else:
        # paint the texture of smplx
        from src.models.nerf_utils import NeRFType
        if cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
            trainer.mesh_paint()
        # train NeRF in hybrid representation
        trainer.nerf_hybrid()

if __name__ == '__main__':
    main()

# python -m scripts.run_teca --config_path=configs/a_fat_European_woman_with_bob_cut_hairstyle.yaml