"""
train_lv_epi.py — EchoVLM LV Epicardium Training Script
Language-Guided Interactive Segmentation for Echocardiographic Cardiac Structures

Trains a model to segment the Left Ventricular Epicardium (outer muscle boundary)
from echocardiographic images using natural language prompts.

Usage:
    python train_lv_epi.py --camus_root /path/to/database_nifti
                           --sam_checkpoint /path/to/sam_vit_b_01ec64.pth
                           --checkpoint_dir ./weights

SAM ViT-B weights:
    Download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Requirements:
    pip install torch torchvision transformers segment-anything
    pip install nibabel albumentations opencv-python scipy tqdm pandas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import random
import time
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from segment_anything import sam_model_registry
import albumentations as A
import nibabel as nib


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="EchoVLM — LV Epicardium Segmentation Training"
    )
    parser.add_argument("--camus_root",     type=str, default="data/CAMUS_public/database_nifti",
                        help="Path to CAMUS database_nifti folder")
    parser.add_argument("--sam_checkpoint", type=str, default="weights/sam_vit_b_01ec64.pth",
                        help="Path to SAM ViT-B weights (.pth)")
    parser.add_argument("--checkpoint_dir", type=str, default="weights",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--batch_size",     type=int, default=2)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--unfreeze_epoch", type=int, default=5,
                        help="Epoch to unfreeze BiomedBERT for fine-tuning")
    parser.add_argument("--backup_every",   type=int, default=5,
                        help="Save a backup checkpoint every N epochs")
    parser.add_argument("--num_workers",    type=int, default=0,
                        help="DataLoader workers (use 0 on Windows)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def load_nii(path):
    return nib.load(path).get_fdata().astype(np.float32)

def preprocess_echo(image_array, target_size=(1024, 1024)):
    if image_array.max() > 0:
        image_array = image_array / image_array.max()
    image_resized = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
    return np.stack([image_resized] * 3, axis=-1).astype(np.float32)

def preprocess_mask(mask_array, structure, target_size=(1024, 1024)):
    struct_label = {"lv_endo": 1, "lv_epi": 2, "la": 3}
    binary_mask  = (mask_array == struct_label[structure]).astype(np.float32)
    return cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)


# ---------------------------------------------------------------------------
# Prompt bank — three linguistic categories (Section III of paper)
# ---------------------------------------------------------------------------
PROMPT_BANK = {
    "lv_endo": {
        "clinical":     "segment the left ventricular endocardium",
        "abbreviation": "LV cavity",
        "descriptive":  "outline the inner boundary of the left ventricle"
    },
    "lv_epi": {
        "clinical":     "segment the left ventricular epicardium",
        "abbreviation": "LV wall outer boundary",
        "descriptive":  "delineate the outer muscle boundary of the left ventricle"
    },
    "la": {
        "clinical":     "segment the left atrium",
        "abbreviation": "LA chamber",
        "descriptive":  "outline the upper left heart chamber"
    }
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CAMUSDataset(Dataset):
    """
    CAMUS echocardiography dataset loader.

    Expects the standard CAMUS NIfTI folder structure:
        database_nifti/
            patient0001/
                patient0001_2CH_ED.nii.gz
                patient0001_2CH_ED_gt.nii.gz
                ...

    Split:
        train — patients 1–400
        val   — patients 401–450
        test  — patients 451–500
    """
    def __init__(self, data_root, split="train", structure="lv_epi"):
        self.data_root = data_root
        self.structure = structure
        all_patients   = sorted([
            p for p in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, p)) and p.startswith("patient")
        ])
        if split == "train":
            self.patients = all_patients[:400]
        elif split == "val":
            self.patients = all_patients[400:450]
        else:
            self.patients = all_patients[450:]

        self.samples = []
        for p in self.patients:
            for view in ["2CH", "4CH"]:
                for tp in ["ED", "ES"]:
                    # Support both .nii.gz and .nii
                    img_path  = os.path.join(data_root, p, f"{p}_{view}_{tp}.nii.gz")
                    mask_path = os.path.join(data_root, p, f"{p}_{view}_{tp}_gt.nii.gz")
                    if not os.path.exists(img_path):
                        img_path  = os.path.join(data_root, p, f"{p}_{view}_{tp}.nii")
                        mask_path = os.path.join(data_root, p, f"{p}_{view}_{tp}_gt.nii")
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img_raw  = load_nii(img_path).squeeze()
        mask_raw = load_nii(mask_path).squeeze()
        image    = preprocess_echo(img_raw)
        mask     = preprocess_mask(mask_raw, self.structure)
        category = random.choice(["clinical", "abbreviation", "descriptive"])
        prompt   = PROMPT_BANK[self.structure][category]
        return (
            torch.from_numpy(image).permute(2, 0, 1).float(),
            torch.from_numpy(mask).unsqueeze(0).float(),
            prompt
        )


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, p=0.5),
    A.Resize(1024, 1024),
], additional_targets={"mask": "mask"})

def apply_augmentation(image_tensor, mask_tensor, transform):
    image_np    = image_tensor.permute(1, 2, 0).cpu().numpy()
    mask_np     = mask_tensor.squeeze(0).cpu().numpy()
    image_uint8 = (image_np * 255).astype(np.uint8)
    mask_uint8  = (mask_np  * 255).astype(np.uint8)
    augmented   = transform(image=image_uint8, mask=mask_uint8)
    image_out   = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float() / 255.0
    mask_out    = torch.from_numpy(augmented["mask"]).unsqueeze(0).float() / 255.0
    return image_out, (mask_out > 0.5).float()


# ---------------------------------------------------------------------------
# Language encoder
# ---------------------------------------------------------------------------
def encode_prompt(prompt_list, tokenizer, bert_model, device):
    """Encode text prompts into full BiomedBERT token sequences [B, N, 768]."""
    tokens = tokenizer(prompt_list, return_tensors="pt",
                       padding=True, truncation=True, max_length=64)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state   # [B, N, 768] — full token sequence, NOT pooled


# ---------------------------------------------------------------------------
# SAM intermediate feature extraction — F1/F2/F3 from blocks 3, 7, 11
# ---------------------------------------------------------------------------
def extract_multiscale_features(image_encoder, x):
    x = image_encoder.patch_embed(x)
    if image_encoder.pos_embed is not None:
        x = x + image_encoder.pos_embed
    features = []
    for i, blk in enumerate(image_encoder.blocks):
        x = blk(x)
        if i in (3, 7, 11):
            features.append(x.permute(0, 3, 1, 2))
    x = image_encoder.neck(x.permute(0, 3, 1, 2))
    return features[0], features[1], features[2], x


# ---------------------------------------------------------------------------
# Cross-Attention Fusion — Equation 3 from paper
# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, lang_dim=768):
        super().__init__()
        self.dv        = visual_dim
        self.lang_proj = nn.Linear(lang_dim, visual_dim)
        self.W_Q       = nn.Linear(visual_dim, visual_dim)
        self.W_K       = nn.Linear(visual_dim, visual_dim)
        self.W_V       = nn.Linear(visual_dim, visual_dim)
        self.norm      = nn.LayerNorm(visual_dim)

    def forward(self, F_i, T):
        T_hat = self.lang_proj(T)
        Q     = self.W_Q(F_i)
        K     = self.W_K(T_hat)
        V     = self.W_V(T_hat)
        attn  = torch.bmm(Q, K.transpose(1, 2)) / (self.dv ** 0.5)
        attn  = F.softmax(attn, dim=-1)
        A_i   = torch.bmm(attn, V)
        return self.norm(F_i + A_i)


# ---------------------------------------------------------------------------
# Multi-Stage Fusion Decoder — Section IV-B of paper
# ---------------------------------------------------------------------------
class MultiStageFusionDecoder(nn.Module):
    def __init__(self, lang_dim=768):
        super().__init__()
        self.proj1 = nn.Conv2d(768, 256, 1)
        self.proj2 = nn.Conv2d(768, 128, 1)
        self.proj3 = nn.Conv2d(768, 64,  1)

        self.ca1 = CrossAttentionFusion(visual_dim=256, lang_dim=lang_dim)
        self.ca2 = CrossAttentionFusion(visual_dim=128, lang_dim=lang_dim)
        self.ca3 = CrossAttentionFusion(visual_dim=64,  lang_dim=lang_dim)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, 2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, 2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, F1, F2, F3, F_neck, T):
        F1 = self.proj1(F1)
        F2 = self.proj2(F2)
        F3 = self.proj3(F3)

        B, C, H, W = F1.shape
        F1_att = self.ca1(F1.flatten(2).permute(0,2,1), T).permute(0,2,1).view(B,C,H,W)
        x = self.up1(F_neck + F1_att)

        B, C, H, W = F2.shape
        F2_att = self.ca2(F2.flatten(2).permute(0,2,1), T).permute(0,2,1).view(B,C,H,W)
        F2_up  = F.interpolate(F2_att, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, F2_up], dim=1))

        B, C, H, W = F3.shape
        F3_att = self.ca3(F3.flatten(2).permute(0,2,1), T).permute(0,2,1).view(B,C,H,W)
        F3_up  = F.interpolate(F3_att, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, F3_up], dim=1))

        x = self.up4(x)
        return self.final(x)


# ---------------------------------------------------------------------------
# Loss — Equation 4: L = L_BCE + lambda * (1 - DSC)
# ---------------------------------------------------------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred        = torch.sigmoid(pred)
    pred_flat   = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1 - dice.mean()

def composite_loss(pred_logits, target, lambda_dice=1.0):
    bce  = F.binary_cross_entropy_with_logits(pred_logits, target.float())
    dice = dice_loss(pred_logits, target)
    return bce + lambda_dice * dice, bce.item(), dice.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure = "lv_epi"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "train_log_lv_epi.txt")

    print("--- EchoVLM LV Epicardium Training ---")
    print(f"Device:  {device}")
    print(f"Epochs:  {args.epochs}")

    # BiomedBERT
    print("\nLoading BiomedBERT...")
    tokenizer  = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    bert_model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        use_safetensors=False,
    ).to(device)
    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.eval()

    # SAM ViT-B
    print("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    image_encoder = sam.image_encoder.to(device)
    for param in image_encoder.parameters():
        param.requires_grad = False
    image_encoder.eval()

    # Decoder
    decoder = MultiStageFusionDecoder(lang_dim=768).to(device)

    # Datasets
    print("Building datasets...")
    train_set    = CAMUSDataset(args.camus_root, split="train", structure=structure)
    val_set      = CAMUSDataset(args.camus_root, split="val",   structure=structure)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_set)} samples | Val: {len(val_set)} samples")

    # Optimizer + scheduler
    ACCUMULATION_STEPS = 4
    optimizer = torch.optim.AdamW([
        {"params": decoder.parameters(), "lr": args.lr, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_dsc = 0.0
    print("\nStarting LV Epicardium training...\n")

    for epoch in range(args.epochs):

        if epoch == args.unfreeze_epoch:
            for param in bert_model.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                "params": bert_model.parameters(),
                "lr": 1e-5, "weight_decay": 1e-4
            })
            print(f"Epoch {epoch+1}: BiomedBERT unfrozen.")

        # ── Train ─────────────────────────────────────────────────────
        image_encoder.eval()
        decoder.train()
        bert_model.train() if epoch >= args.unfreeze_epoch else bert_model.eval()

        epoch_loss, epoch_bce, epoch_dice = 0.0, 0.0, 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (images, masks, prompts) in enumerate(train_loader):
            images = images.to(device)
            masks  = masks.to(device)

            aug_images, aug_masks = [], []
            for i in range(images.shape[0]):
                img_aug, mask_aug = apply_augmentation(images[i], masks[i], train_transform)
                aug_images.append(img_aug)
                aug_masks.append(mask_aug)
            images = torch.stack(aug_images).to(device)
            masks  = torch.stack(aug_masks).to(device)

            use_amp = device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                T = encode_prompt(list(prompts), tokenizer, bert_model, device)
                with torch.no_grad():
                    F1, F2, F3, F_neck = extract_multiscale_features(image_encoder, images)
                pred_logits             = decoder(F1, F2, F3, F_neck, T)
                loss, bce_val, dice_val = composite_loss(pred_logits, masks)
                loss                    = loss / ACCUMULATION_STEPS

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUMULATION_STEPS
            epoch_bce  += bce_val
            epoch_dice += dice_val

            if step % 100 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {step}/{len(train_loader)} | "
                      f"Loss: {loss.item()*ACCUMULATION_STEPS:.4f} | "
                      f"BCE: {bce_val:.4f} | Dice: {dice_val:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        avg_bce  = epoch_bce  / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────
        image_encoder.eval()
        decoder.eval()
        bert_model.eval()

        val_dscs = []
        with torch.no_grad():
            for images, masks, prompts in val_loader:
                images = images.to(device)
                masks  = masks.to(device)
                use_amp = device.type == "cuda"
                with torch.amp.autocast("cuda", enabled=use_amp):
                    T                  = encode_prompt(list(prompts), tokenizer, bert_model, device)
                    F1, F2, F3, F_neck = extract_multiscale_features(image_encoder, images)
                    pred_logits        = decoder(F1, F2, F3, F_neck, T)
                pred         = (torch.sigmoid(pred_logits) > 0.5).float()
                gt           = masks.float().to(device)
                intersection = (pred * gt).sum()
                dsc          = (2 * intersection) / (pred.sum() + gt.sum() + 1e-6)
                val_dscs.append(dsc.item())

        val_dsc = sum(val_dscs) / len(val_dscs)
        elapsed = time.time() - t0

        log_line = (f"Epoch {epoch+1}/{args.epochs} | "
                    f"Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | Dice: {avg_dice:.4f} | "
                    f"Val DSC: {val_dsc:.4f} | Time: {elapsed:.1f}s\n")
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line)

        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            save_path = os.path.join(args.checkpoint_dir, "best_model_lv_epi.pth")
            torch.save({
                "epoch":        epoch,
                "decoder":      decoder.state_dict(),
                "bert_model":   bert_model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "best_val_dsc": best_val_dsc,
                "structure":    structure,
            }, save_path)
            print(f"  ✓ New best DSC {best_val_dsc:.4f} — saved to {save_path}")

        if (epoch + 1) % args.backup_every == 0:
            backup_path = os.path.join(args.checkpoint_dir,
                                       f"checkpoint_lv_epi_epoch{epoch+1}.pth")
            torch.save({
                "epoch":        epoch,
                "decoder":      decoder.state_dict(),
                "bert_model":   bert_model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_dsc":      val_dsc,
                "structure":    structure,
            }, backup_path)
            print(f"  ✓ Backup saved: {backup_path}")

    print(f"\nLV Epicardium training complete. Best Val DSC: {best_val_dsc:.4f}")

    print("\nVerifying best_model_lv_epi.pth...")
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_model_lv_epi.pth"),
                      map_location="cpu")
    print(f"  Structure:    {ckpt['structure']}")
    print(f"  Best Val DSC: {ckpt['best_val_dsc']:.4f}")
    print("  Checkpoint verified.")


if __name__ == "__main__":
    main()
