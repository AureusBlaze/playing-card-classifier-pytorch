# 🃏 Playing Card Classifier with EfficientNet & PyTorch

A Deep Learning Computer Vision system built with **PyTorch** and **timm** to classify images of playing cards into **53 distinct categories** (52 standard deck cards + Joker). The model leverages transfer learning using a pre-trained **EfficientNet-B0** convolutional neural network backbone.

---

## 📌 Features

- **53-Class Fine-Grained Classification**: Accurately recognizes card ranks and suits across diverse lighting, angles, and camera backgrounds.
- **EfficientNet-B0 Transfer Learning**: Harnesses compound scaling for high accuracy while maintaining lightweight inference speed.
- **Pre-trained Checkpoint Included**: Comes with `simple_card_classifier.pt` (21.7 MB) ready for instant testing without retraining.
- **Standalone CLI Inference Tool**: Run `predict.py` on any image file to output card name and softmax confidence scores.
- **Interactive Jupyter Notebook**: Full training, validation, data augmentation, and loss visualization workflow in `PlayingCard-Classification.ipynb`.

---

## 🧠 Model Architecture

```
Input Image (3 × 128 × 128)
  │
  ├──► EfficientNet-B0 Feature Extractor (Pre-trained)
  │    (Mobile Inverted Bottleneck Convolutions / MBConv)
  │
  ├──► Feature Embedding (1,280 dimensions)
  │
  └──► Linear Head: nn.Linear(1280, 53)  ──► Softmax Card Probabilities
```

- **Backbone**: `timm.create_model('efficientnet_b0', pretrained=True)`
- **Classifier**: `nn.Linear(1280, num_classes=53)`
- **Loss Function**: `nn.CrossEntropyLoss`

---

## 📁 Repository Structure

```
playing-card-classifier-pytorch/
├── .gitignore                          # Excludes Python caches & large binary archives
├── README.md                           # Model documentation & setup instructions
├── requirements.txt                    # Dependencies (torch, torchvision, timm, pandas)
├── predict.py                          # Instant single-image CLI inference script
├── PlayingCard-Classification.ipynb    # End-to-end training & analysis notebook
├── simple_card_classifier.pt           # Trained PyTorch model checkpoint (21.7 MB)
└── Card Image Dataset-Classification/  # Organized card dataset
    ├── cards.csv                       # Class index and metadata mapping
    ├── test/                           # 265 test images across all 53 card classes
    ├── valid/                          # 265 validation images
    └── train/                          # Raw train split (downloadable via Kaggle)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/AureusBlaze/playing-card-classifier-pytorch.git
cd playing-card-classifier-pytorch
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Inference on Any Card Image
Predict an image directly with `predict.py`:

```bash
python predict.py "Card Image Dataset-Classification/test/ace of clubs/1.jpg"
```

*Sample Output:*
```
[*] Input Image : Card Image Dataset-Classification/test/ace of clubs/1.jpg
[*] Prediction  : Ace Of Clubs (Index: 0)
[*] Confidence  : 98.74%
```

### 4. Interactive Training Workflow
To inspect the training loop, loss graphs, and evaluation matrices:
```bash
jupyter notebook PlayingCard-Classification.ipynb
```

---

## 🛠️ Tech Stack
- **Python 3.x**
- **PyTorch (`torch`, `torch.nn`)**
- **timm** (PyTorch Image Models: EfficientNet-B0)
- **Torchvision** (Image transforms & data augmentation)
- **Pandas** (Metadata index handling)
- **Pillow** (Image loading)
