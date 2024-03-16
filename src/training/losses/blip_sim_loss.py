from torchvision import transforms
from src.configs.train_config import TrainConfig
import torch
from torch import nn

class blip_sim_loss(nn.Module):
    def __init__(self, cfg:TrainConfig, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.view_dirs = ['front', 'left side', 'back', 'right side', 'overhead', 'bottom']
        self.blip_model, self.blip_text_z, self.blip_text_z_neg, self.blip_text_str = self.cal_blip_text_embeddings()

    # both blip and clip work well
    def cal_blip_text_embeddings(self):
        from ..BLIP.models.blip_itm import blip_itm, BLIP_ITM
        image_size = 224
        path = self.cfg.optim.blip_path
        vit  = self.cfg.optim.blip_vit
        model = blip_itm(pretrained=path, image_size=image_size, vit=vit)
        model.eval()
        model = model.to(device=self.device)
        ref_text = self.cfg.guide.blip_text
        text_z = []
        blip_text_str = []
        for d in self.view_dirs:
            text = ref_text.format(d)
            blip_text_str.append(text)
            with torch.no_grad():
                text_output = model.get_text_feature(text, device=self.device)
            text_z.append(text_output)  
        text_neg = self.cfg.guide.blip_neg_text
        text_z_neg = model.get_text_feature(text_neg, device=self.device).detach()
        return model, text_z, text_z_neg, blip_text_str

    def forward(self, pred_rgb, dirs):
        image_size = 224
        transform = transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ])   
        augmented_rgb = transform(pred_rgb)
        text_feature = self.blip_text_z[dirs]
        img_feature = self.blip_model.get_img_feature(augmented_rgb)
        cos_sim = img_feature @ text_feature.t()
        cos_sim_neg = img_feature @ self.blip_text_z_neg.t()
        loss = cos_sim_neg[0][0]-cos_sim[0][0]
        return loss