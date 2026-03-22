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
| Structure | Val DSC | Train Loss | BCE Loss | Dice Loss |
|-----------|---------|-----------|---------|---------|
| LV Endocardium | 0.8887 | 0.2073 | 0.0636 | 0.1436 |
| LV Epicardium | 0.8806 | 0.3091 | 0.1219 | 0.1772 |
| Left Atrium | 0.8292 | 0.3234 | 0.0644 | 0.2590 |

Peak single-sample DSC of 0.965 achieved on LV endocardium segmentation.

## Dataset
Trained and evaluated on the CAMUS echocardiography benchmark (500 patients, 
2CH and 4CH views, ED and ES timepoints). All images resized to 1024×1024 
and normalized to [0,1] prior to model input.

## Prompt Categories
The model is evaluated across three linguistic prompt categories:
- **Clinical terminology** — e.g. "segment the left ventricular endocardium"
- **Anatomical abbreviations** — e.g. "LV cavity"
- **Descriptive expressions** — e.g. "outline the inner boundary of the left ventricle"

## Project Structure
```
echo-vlm-project/
├── src/
│   ├── train.py       # Training script with multi-stage fusion decoder
│   ├── model.py       # EchoVLM architecture
│   └── dataset.py     # CAMUS NIfTI dataset loader
├── weights/           # Model checkpoints
├── data/              # CAMUS dataset
├── test_visualize.py  # Visualization script
└── test_inference.py  # Single forward pass test
```

## Requirements
```
pip install torch torchvision transformers segment-anything 
pip install nibabel albumentations opencv-python scipy tqdm
```

## Training
```bash
python -m src.train
```

## Authors
- Syed Hasib Akhter Faruqui — Sam Houston State University
- Suhaan Gopal — McNeil High School
- Arjun Mijar — McNeil High School
