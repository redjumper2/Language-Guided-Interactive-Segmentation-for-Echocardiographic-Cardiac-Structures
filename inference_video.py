# Cell 1

VIDEO_PATH = "insert_video_path_here"

MODEL_PATH_ENDO = "insert_model_path_here"
MODEL_PATH_EPI  = "insert_model_path_here"
MODEL_PATH_LA   = "insert_model_path_here"


# Cell 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from segment_anything import sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cell 3

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, lang_dim):
        super().__init__()
        self.lang_proj = nn.Linear(lang_dim, dim)
        self.W_Q = nn.Conv2d(dim, dim, 1)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x, lang_tokens):
        B, C, H, W = x.shape
        lang = self.lang_proj(lang_tokens)

        q = self.W_Q(x).flatten(2).permute(0, 2, 1)
        k = self.W_K(lang)
        v = self.W_V(lang)

        attn = torch.softmax(q @ k.transpose(-2, -1) / (C ** 0.5), dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return self.norm(x + out)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj1 = nn.Conv2d(768, 256, 1)
        self.proj2 = nn.Conv2d(768, 128, 1)
        self.proj3 = nn.Conv2d(768, 64, 1)

        self.ca1 = CrossAttentionBlock(256, 768)
        self.ca2 = CrossAttentionBlock(128, 768)
        self.ca3 = CrossAttentionBlock(64, 768)

        self.up1 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, f1, f2, f3, lang_tokens):
        x1 = self.ca1(self.proj1(f1), lang_tokens)
        x2 = self.ca2(self.proj2(f2), lang_tokens)
        x3 = self.ca3(self.proj3(f3), lang_tokens)

        x = self.up1(x1)

        x2 = F.interpolate(x2, size=x.shape[-2:], mode='bilinear')
        x = x + x2

        x = self.up2(x)

        x3 = F.interpolate(x3, size=x.shape[-2:], mode='bilinear')
        x = x + x3

        x = self.up3(x)
        x = self.up4(x)

        return self.final(x)


class EchoVLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )
        self.lang_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )

        sam = sam_model_registry["vit_b"](checkpoint="insert_sam_checkpoint_path_here")
        self.image_encoder = sam.image_encoder

        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.decoder = Decoder()
        self.intermediates = {}

        self._register_hooks()

    def _register_hooks(self):
        def hook(name):
            def fn(module, inp, out):
                self.intermediates[name] = out
            return fn

        self.image_encoder.blocks[3].register_forward_hook(hook("f1"))
        self.image_encoder.blocks[7].register_forward_hook(hook("f2"))
        self.image_encoder.blocks[11].register_forward_hook(hook("f3"))

    def encode_text(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        return self.lang_encoder(**tokens).last_hidden_state

    def forward(self, image, prompt):
        lang = self.encode_text(prompt)
        _ = self.image_encoder(image)

        f1 = self.intermediates["f1"].permute(0,3,1,2)
        f2 = self.intermediates["f2"].permute(0,3,1,2)
        f3 = self.intermediates["f3"].permute(0,3,1,2)

        out = self.decoder(f1, f2, f3, lang)
        out = F.interpolate(out, size=(1024,1024), mode="bilinear")

        return out


# Cell 4

def load_model(path):
    model = EchoVLM().to(device)
    ckpt = torch.load(path, map_location=device)

    model.decoder.load_state_dict(ckpt["decoder"], strict=False)
    model.lang_encoder.load_state_dict(ckpt["bert_model"], strict=False)

    model.eval()
    return model

model_endo = load_model(MODEL_PATH_ENDO)
model_epi  = load_model(MODEL_PATH_EPI)
model_la   = load_model(MODEL_PATH_LA)


# Cell 5

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
cap.release()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (1024,1024))


# Cell 6

img = frame.astype(np.float32) / 255.0
img = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

with torch.no_grad():
    m1 = torch.sigmoid(model_endo(img, "outline the inner boundary of the left ventricle"))
    m2 = torch.sigmoid(model_epi(img, "segment the outer boundary of the left ventricle"))
    m3 = torch.sigmoid(model_la(img, "segment the left atrium"))

m1 = (m1 > 0.5).cpu().numpy().squeeze()
m2 = (m2 > 0.5).cpu().numpy().squeeze()
m3 = (m3 > 0.5).cpu().numpy().squeeze()


# Cell 7

overlay = frame.copy()
overlay[m1>0] = [255,0,0]
overlay[m2>0] = [0,255,0]
overlay[m3>0] = [0,0,255]

combined = np.zeros_like(frame)
combined[m1>0] = [255,0,0]
combined[m2>0] = [0,255,0]
combined[m3>0] = [0,0,255]

fig, ax = plt.subplots(2,2, figsize=(10,10))

ax[0,0].axis("off")
ax[0,0].text(0.1,0.5,"Prompts:\nEndo\nEpi\nLA")

ax[0,1].imshow(frame)
ax[0,1].set_title("Original")

ax[1,0].imshow(combined)
ax[1,0].set_title("Segmentation")

ax[1,1].imshow(overlay)
ax[1,1].set_title("Overlay")

for a in ax.flatten():
    a.axis("off")

plt.tight_layout()
plt.savefig("final.png", dpi=300)
plt.show()
