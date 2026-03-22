# Language-Guided Interactive Segmentation for Echocardiographic Cardiac Structures

This repository contains the implementation for EchoVLM, a vision-language 
segmentation framework designed specifically for echocardiographic cardiac 
structure delineation. Rather than relying on geometric prompts such as point 
clicks or bounding boxes, EchoVLM accepts natural language descriptions of the 
target anatomy and produces binary segmentation masks without any spatial prompt 
placement.

## Architecture
- **Language Encoder:** BiomedBERT pretrained on PubMed abstracts and clinical notes
- **Visual Backbone:** SAM ViT-B image encoder with intermediate feature extraction 
  at blocks 3, 7, and 11
- **Fusion:** Multi-stage cross-attention between language tokens and visual feature 
  maps at three decoder resolutions (coarse → fine boundary)
- **Decoder:** Lightweight convolutional upsampling to 1024×1024 output

## Results on CAMUS
| Structure | Val DSC |
|-----------|---------|
| LV Endocardium | 0.8838 |
| Left Atrium | 0.8292 |
| LV Epicardium | TBD |

## Dataset
Trained and evaluated on the CAMUS echocardiography benchmark (500 patients, 
2CH and 4CH views, ED and ES timepoints).

## Prompt Categories
The model is evaluated across three linguistic prompt categories:
- Clinical terminology
- Anatomical abbreviations  
- Descriptive expressions

## Requirements
torch, transformers, segment-anything, nibabel, albumentations, opencv-python, scipy
