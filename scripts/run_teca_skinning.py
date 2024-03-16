import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import imageio
import cv2
import torch.nn.functional as F
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
from src.configs.train_config import TrainConfig
from subprocess import run
import pyrallis

def video_gen(folder_path1, folder_path2, output_name, skinning_h):
    def load_images_from_folder(folder):
        images = []
        transform = transforms.ToTensor()
        if folder != None:
            image_files = sorted(os.listdir(folder)) 
            for filename in image_files:
                if filename.endswith(('.png')):  # Add more file extensions if needed
                    img_path = os.path.join(folder, filename)
                    img = Image.open(img_path)
                    img_tensor = transform(img)
                    images.append(img_tensor)
            return images

    video_capture1 = cv2.VideoCapture(folder_path1)
    if not video_capture1.isOpened():
        print("can not open video file")
        exit()
    # transfer to tensor
    transform = transforms.ToTensor()
    frames1 = []
    # read frames
    while True:
        ret, frame = video_capture1.read()
        if not ret:
            break
        frame = frame[:,:,::-1]
        tensor_frame = transform(frame.copy())
        frames1.append(tensor_frame)
    images_tensor1 = torch.stack(frames1)
    images_tensor1 =  F.interpolate(images_tensor1, (skinning_h,3*skinning_h))
    images_tensor1_hybridimg = images_tensor1[..., 0:skinning_h]
    images_tensor1_mesh_txture_img = images_tensor1[..., skinning_h:2*skinning_h]
    images_tensor1_nerf_img = images_tensor1[..., 2*skinning_h:3*skinning_h]

    # load the images and convert them to tensors
    images2 = load_images_from_folder(folder_path2)
    images_tensor2 = torch.stack(images2)
    images_tensor2 =  F.interpolate(images_tensor2, (skinning_h,3*skinning_h))
    images_tensor2_hybridimg = images_tensor2[..., 0:skinning_h]
    images_tensor2_rgbimg = images_tensor2[..., skinning_h:2*skinning_h]
    images_tensor2_mask = images_tensor2[..., 2*skinning_h:3*skinning_h]
    skinning_img1 = images_tensor1_mesh_txture_img*(1-images_tensor2_mask) + images_tensor2_hybridimg*images_tensor2_mask
    dump_vid = lambda video, name: imageio.mimsave(str(folder_path2) + f"{name}.mp4", video,
                                                                fps=10,
                                                                quality=8, macro_block_size=1)
    dump_vid(skinning_img1.permute(0,2,3,1), output_name)

@pyrallis.wrap('./configs/a_fat_European_woman_with_bowl_cut_hairstyle_refine.yaml')
def main(cfg: TrainConfig):
    config_path = 'configs/skinning_try_on.yaml'
    run(['python', '-m' 'scripts.run_teca', '--config_path', config_path], check=True, env=os.environ.copy()) 
    # source avatar
    folder_path1 = "./experiments/a slim woman with dreadlocks hairstyle refine/vis/eval/a slim woman with dreadlocks hairstyle refine_step_01000_eval.mp4"
    # transfered assets
    folder_path2 = './experiments/a slim woman with dreadlocks hairstyle_skinning/results/skinning'
    output_name = 'video'
    video_gen(folder_path1=folder_path1, folder_path2=folder_path2, output_name=output_name, skinning_h=cfg.nerfrender.fine_tunning_h)

if __name__ == '__main__':
    main()

# python -m scripts.run_teca_skinning --config_path=configs/skinning_try_on.yaml