import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from src.configs.train_config import GuideConfig
import cv2
import numpy as np

class mask_loss(nn.Module):
    def __init__(self, cfg:GuideConfig, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device

    def clipseg_mask(self, prompts, image):
        inputs = processor(text=prompts, images=[image] * len([prompts]), padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        preds = outputs.logits
        return torch.sigmoid(preds)

    def forward(self, pred_rgb, weight_sum_except_last_point):
        mask_nerf = self.clipseg_mask(self.cfg.guide.clipseg_text, torchvision.transforms.functional.to_pil_image(pred_rgb[0])).to(device=self.device)
        from src.models.nerf_utils import NeRFType
        if self.cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
            mask_tar = F.interpolate(mask_nerf.unsqueeze(0).unsqueeze(0), (self.cfg.nerfrender.train_h, self.cfg.nerfrender.train_w), mode='bilinear', antialias=True)[0][0]
        if self.cfg.nerfrender.nerf_type == NeRFType['fine_tunning']:
            mask_tar = F.interpolate(mask_nerf.unsqueeze(0).unsqueeze(0), (self.cfg.nerfrender.fine_tunning_h, self.cfg.nerfrender.fine_tunning_w), mode='bilinear', antialias=True)[0][0]
        
        # when training in coarse stage use init mask value
        if self.cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
            mask_tar = mask_tar
        # when training in refine stage use 0/1 mask vlaue
        if self.cfg.nerfrender.nerf_type == NeRFType['fine_tunning']:
            mask_tar = (mask_tar>0.6).float()

        # erode eliminate some sparse points
        if self.cfg.guide.maskerode >0:
            mask_tar = torch.from_numpy(
                cv2.erode(mask_tar.detach().cpu().numpy(), np.ones((self.cfg.guide.maskerode, self.cfg.guide.maskerode), np.uint8))).to(
                mask_tar.device)
        # dilate extend mask
        if self.cfg.guide.maskdilate>0:
            mask_tar = torch.from_numpy(
                cv2.dilate(mask_tar.detach().cpu().numpy(), np.ones((self.cfg.guide.maskdilate, self.cfg.guide.maskdilate), np.uint8))).to(
                mask_tar.device)  

        from src.models.nerf_utils import NeRFType
        if self.cfg.nerfrender.nerf_type != NeRFType['fine_tunning']:
            mask_pre = weight_sum_except_last_point.reshape(self.cfg.nerfrender.train_h, self.cfg.nerfrender.train_w)
        if self.cfg.nerfrender.nerf_type == NeRFType['fine_tunning']:
            mask_pre = weight_sum_except_last_point.reshape(self.cfg.nerfrender.fine_tunning_h, self.cfg.nerfrender.fine_tunning_w)
        mse_loss = torch.nn.MSELoss()
        clipseg_loss = mse_loss(mask_pre,mask_tar)
        return clipseg_loss