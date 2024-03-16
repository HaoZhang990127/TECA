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
import imageio

@pyrallis.wrap('./configs/skinning_animation.yaml')
def main(cfg: TrainConfig):
    animate_all_frames = []
    shape_path = os.path.join(cfg.log.exp_path, f'geometry/shape.npy')
    shape_init = torch.from_numpy(np.load(shape_path))  
    expression_path = 'shapes/exp_animate.npy'
    expression_init = torch.from_numpy(np.load(expression_path)) 
    jaw_pose_path = 'shapes/jaw_pose_animate.npy'
    jaw_pose_init = torch.from_numpy(np.load(jaw_pose_path)) 
    pose_path = 'shapes/pose_animate.npy'
    pose_init = torch.from_numpy(np.load(pose_path))  
    n = expression_init.shape[0]
    for i in range(20):
        shape = shape_init
        expression = expression_init[i,...].unsqueeze(0)
        jaw_pose = jaw_pose_init[i,...].unsqueeze(0)
        pose = pose_init[i,...].unsqueeze(0)
        if 'trainer' in locals():
            del trainer
            torch.cuda.empty_cache()
            # to evaluate it step by step to avoid cuda out of memory
            import ipdb; ipdb.set_trace()

        with torch.no_grad():
            trainer = TECA(cfg, shape, expression, jaw_pose, pose)
            frame = trainer.full_eval(hybrid=True)

        frame_tensor = torch.from_numpy(frame[0]).cpu()
        animate_all_frames.append(frame_tensor)
    dump_vid = lambda video, name: imageio.mimsave(cfg.log.exp_dir / f"{cfg.log.exp_name}_{name}.mp4", video,
                                                            fps=10,
                                                            quality=8, macro_block_size=1)
    animate_all_frames = np.stack(animate_all_frames)
    dump_vid(animate_all_frames, 'animate_all_frames')

if __name__ == '__main__':
    main()

# python -m scripts.run_teca_animation --config_path=configs/skinning_animation.yaml