import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import config

CLIP_MEAN=(0.48145466,0.4578275,0.40821073)
CLIP_STD=(0.26862953,0.26130258,0.27577711)
class CLIP(nn.Module):
    def __init__(self, class_names):
        super().__init__()
        self.model,_,_=open_clip.create_model_and_transforms(
            config.CLIP_MODEL,pretrained=config.CLIP_PRETRAINED
        )
        self.model.eval()
        tokenizer=open_clip.get_tokenizer(config.CLIP_MODEL)
        self.register_buffer("mean",torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer("std",torch.tensor(CLIP_STD).view(1,3,1,1))
        prompts=tokenizer([f"a photo of a {c}" for c in class_names])
        with torch.no_grad():
            text_features=self.model.encode_text(prompts)
            self.register_buffer("text_features", F.normalize(text_features, dim=1))

    def forward(self, x):
        x=F.interpolate(x, size=(224,224), mode="bilinear",align_corners=False)
        x=(x-self.mean)/self.std
        img_features=F.normalize(self.model.encode_image(x), dim=1)
        return 100 * img_features @ self.text_features.T