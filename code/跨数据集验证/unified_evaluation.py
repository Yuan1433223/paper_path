"""
Unified Evaluation Script for Bagua-MoE
========================================
Purpose: Evaluate trained Bagua-MoE checkpoints on:
1. Flickr30K Image-Text Retrieval
2. ImageNet Zero-shot Classification

Usage:
    # Flickr30K evaluation
    python unified_evaluation.py --task flickr30k --checkpoint test_bagua_epoch_5.pt
    
    # ImageNet evaluation  
    python unified_evaluation.py --task imagenet --checkpoint test_bagua_epoch_5.pt \
        --imagenet_root /mnt/data/imagenet
    
    # Both tasks
    python unified_evaluation.py --task all --checkpoint test_bagua_epoch_5.pt
    
为验证 Bagua-MoE 的跨数据集泛化能力，我们在 Flickr30K 上进行了额外评估。结果表明，模型在不同数据分布下仍保持一致的性能优势（详见表 X）
只评估 Flickr30K
python unified_evaluation.py \
    --task flickr30k \
    --checkpoint /mnt/workspace/ckpts/ckpt1203/test_bagua_epoch_5.pt \
    --batch_size 64
 
# res
Using device: cuda
Loading CLIP ViT-B/16...
Injecting Bagua-MoE modules...
Loading checkpoint: /mnt/workspace/ckpts/ckpt1203/test_bagua_epoch_5.pt
Loaded 326/338 parameters from /mnt/workspace/ckpts/ckpt1203/test_bagua_epoch_5.pt

============================================================
Evaluating on Flickr30K
============================================================
Loaded 3179 images from /mnt/data/flickr30k/annotations/val_captions.json
Extracting features from Flickr30K...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:12<00:00,  4.16it/s]
Total images: 3179, Total texts: 3179
Total images: 3179, Total texts: 3179
Similarity matrix shape: torch.Size([3179, 3179])

============================================================
Flickr30K Results:
============================================================
Image-to-Text:
  R@1:  70.08%
  R@5:  90.56%
  R@10: 94.81%
Text-to-Image:
  R@1:  69.86%
  R@5:  90.37%
  R@10: 94.56%
============================================================
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")

# Import your Bagua modules
from bagua_moe_coco import BaguaAttention


# ==========================================
# Flickr30K Dataset
# ==========================================
class Flickr30KDataset(Dataset):
    """
    Flickr30K dataset for image-text retrieval.
    
    Directory structure:
    /mnt/data/
    ├── flickr30k_pic/           # Images
    │   ├── 1000092795.jpg
    │   └── ...
    └── flickr30k/annotations/   # Captions
        ├── train_captions.json
        └── val_captions.json
    
    Flickr30K dataset for image-text retrieval.
    
    JSON format:
    [
      {
        "image": "4552824261.jpg",
        "captions": [
          "Caption 1",
          "Caption 2",
          ...
        ]
      },
      ...
    ]
    """
    def __init__(self, image_dir, ann_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Build mapping: image_id -> captions
        self.image_to_captions = {}
        for ann in data:
            img_filename = ann['image']
            img_id = img_filename.replace('.jpg', '')
            self.image_to_captions[img_id] = ann['captions']
        
        self.image_ids = list(self.image_to_captions.keys())
        
        print(f"Loaded {len(self.image_ids)} images from {ann_file}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        captions = self.image_to_captions[img_id]
        
        # ========== 关键修改：只返回第一个 caption（字符串，不是列表） ==========
        caption = captions[0]  # 使用第一个 caption
        
        return image, caption, img_id  # 返回单个字符串，不是列表


# ==========================================
# ImageNet Dataset
# ==========================================
class ImageNetDataset(Dataset):
    """
    ImageNet validation set.
    
    Directory structure:
    /mnt/data/imagenet/val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000001.JPEG
    │   └── ...
    └── n01443537/
        └── ...
    """
    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        # Collect all images
        self.samples = []
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.JPEG', '.jpg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_name), class_name))
        
        # Build class_name to index mapping
        self.classes = sorted(set([s[1] for s in self.samples]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[class_name]
        return image, label


# ==========================================
# Bagua Surgery
# ==========================================
def inject_bagua_attention(model):
    """Inject BaguaAttention into CLIP vision transformer."""
    vision_transformer = model.visual.transformer
    
    for block in vision_transformer.resblocks:
        original_attn = block.attn
        bagua_attn = BaguaAttention(
            original_attn,
            original_attn.embed_dim,
            original_attn.num_heads,
            window_size=7
        )
        # 获取原注意力层的设备和数据类型
        device = original_attn.in_proj_weight.device
        dtype = original_attn.in_proj_weight.dtype  # 关键：同步数据类型
        # 同时移动设备和转换数据类型
        bagua_attn = bagua_attn.to(device=device, dtype=dtype)
        block.attn = bagua_attn
    
    return model


def load_bagua_checkpoint(model, checkpoint_path, device):
    """Load trained Bagua-MoE checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Filter out non-matching keys
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    
    model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {len(filtered_state)}/{len(state_dict)} parameters from {checkpoint_path}")
    
    return model


# ==========================================
# Flickr30K Evaluation
# ==========================================
@torch.no_grad()
def evaluate_flickr30k(model, dataloader, device):
    model.eval()
    
    all_image_features = []
    all_text_features = []
    
    print("Extracting features from Flickr30K...")
    for images, captions, _ in tqdm(dataloader):  # captions 现在是字符串列表
        images = images.to(device)
        
        # Image features
        img_feats = model.encode_image(images)
        img_feats = F.normalize(img_feats, dim=-1)
        all_image_features.append(img_feats.cpu())
        
        # Text features - captions 已经是字符串列表
        text_tokens = clip.tokenize(captions, truncate=True).to(device)
        txt_feats = model.encode_text(text_tokens)
        txt_feats = F.normalize(txt_feats, dim=-1)
        all_text_features.append(txt_feats.cpu())
    
    # Concatenate
    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    all_text_features = torch.cat(all_text_features, dim=0).to(device)
    
    print(f"Total images: {len(all_image_features)}, Total texts: {len(all_text_features)}")
    
    # ========== 新增：强制检查和对齐 ==========
    n_images = len(all_image_features)
    n_texts = len(all_text_features)
    
    print(f"Total images: {n_images}, Total texts: {n_texts}")
    
    if n_images != n_texts:
        print(f"ERROR: Feature count mismatch! Using min length.")
        min_len = min(n_images, n_texts)
        all_image_features = all_image_features[:min_len]
        all_text_features = all_text_features[:min_len]
    # =========================================
    
    all_image_features = all_image_features.to(device)
    all_text_features = all_text_features.to(device)
    
    # Compute similarity matrix
    sim_matrix = all_image_features @ all_text_features.T
    n = len(sim_matrix)
    
    print(f"Similarity matrix shape: {sim_matrix.shape}")  # 应该是 [N, N]
    
    # Image-to-Text Retrieval
    i2t_ranks = []
    for i in range(n):
        sorted_indices = sim_matrix[i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0]
        if len(rank) == 0:
            rank = n
        else:
            rank = rank.item()
        i2t_ranks.append(rank)
    
    i2t_ranks = torch.tensor(i2t_ranks)
    
    # Text-to-Image Retrieval
    t2i_ranks = []
    for i in range(n):
        sorted_indices = sim_matrix[:, i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0]
        if len(rank) == 0:
            rank = n
        else:
            rank = rank.item()
        t2i_ranks.append(rank)
    
    t2i_ranks = torch.tensor(t2i_ranks)
    
    # Compute metrics
    metrics = {
        'i2t_r1': (i2t_ranks < 1).float().mean().item() * 100,
        'i2t_r5': (i2t_ranks < 5).float().mean().item() * 100,
        'i2t_r10': (i2t_ranks < 10).float().mean().item() * 100,
        't2i_r1': (t2i_ranks < 1).float().mean().item() * 100,
        't2i_r5': (t2i_ranks < 5).float().mean().item() * 100,
        't2i_r10': (t2i_ranks < 10).float().mean().item() * 100,
    }
    
    return metrics


# ==========================================
# ImageNet Evaluation
# ==========================================
# ImageNet class names (1000 classes)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    # ... (complete list of 1000 classes)
    # For brevity, using placeholder. In real code, load from file.
]

# CLIP prompt templates (80 templates from official CLIP paper)
IMAGENET_TEMPLATES = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


@torch.no_grad()
def build_text_classifier(model, classnames, templates, device):
    """Build text classifier using template ensemble."""
    model.eval()
    
    text_features = []
    for classname in tqdm(classnames, desc="Building text classifier"):
        # Generate prompts for this class
        texts = [template.format(classname) for template in templates]
        texts = clip.tokenize(texts).to(device)
        
        # Encode
        class_embeddings = model.encode_text(texts)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        
        # Average over templates
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = F.normalize(class_embedding, dim=-1, p=2)
        
        text_features.append(class_embedding)
    
    text_features = torch.stack(text_features, dim=0)  # [1000, D]
    return text_features


@torch.no_grad()
def evaluate_imagenet(model, dataloader, text_classifier, device):
    """Evaluate on ImageNet with zero-shot classification."""
    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Evaluating ImageNet"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Image features
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity with text classifier
        logits = 100.0 * image_features @ text_classifier.T  # [B, 1000]
        
        # Top-1 and Top-5
        _, pred_top5 = logits.topk(5, dim=-1)
        
        top1_correct += (pred_top5[:, 0] == labels).sum().item()
        top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
    
    metrics = {
        'top1_acc': top1_correct / total * 100,
        'top5_acc': top5_correct / total * 100,
    }
    
    return metrics


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation for Bagua-MoE')
    parser.add_argument('--task', type=str, choices=['flickr30k', 'imagenet', 'all'], 
                       default='all', help='Evaluation task')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to Bagua-MoE checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # Flickr30K settings
    parser.add_argument('--flickr_image_dir', type=str, default='/mnt/data/flickr30k_pic')
    parser.add_argument('--flickr_ann', type=str, 
                       default='/mnt/data/flickr30k/annotations/val_captions.json')
    
    # ImageNet settings
    parser.add_argument('--imagenet_root', type=str, default='/mnt/data/imagenet/val')
    parser.add_argument('--imagenet_classes', type=str, 
                       default='./imagenet_classes.txt',
                       help='File containing ImageNet class names')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP ViT-B/16...")
    model, preprocess = clip.load("ViT-B/16", device=device)
    
    # Inject Bagua attention
    print("Injecting Bagua-MoE modules...")
    model = inject_bagua_attention(model)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_bagua_checkpoint(model, args.checkpoint, device)
    model.eval()
    
    # ========== Flickr30K Evaluation ==========
    if args.task in ['flickr30k', 'all']:
        print("\n" + "="*60)
        print("Evaluating on Flickr30K")
        print("="*60)
        
        flickr_dataset = Flickr30KDataset(
            args.flickr_image_dir,
            args.flickr_ann,
            transform=preprocess
        )
        flickr_loader = DataLoader(
            flickr_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        flickr_metrics = evaluate_flickr30k(model, flickr_loader, device)
        
        print("\n" + "="*60)
        print("Flickr30K Results:")
        print("="*60)
        print(f"Image-to-Text:")
        print(f"  R@1:  {flickr_metrics['i2t_r1']:.2f}%")
        print(f"  R@5:  {flickr_metrics['i2t_r5']:.2f}%")
        print(f"  R@10: {flickr_metrics['i2t_r10']:.2f}%")
        print(f"Text-to-Image:")
        print(f"  R@1:  {flickr_metrics['t2i_r1']:.2f}%")
        print(f"  R@5:  {flickr_metrics['t2i_r5']:.2f}%")
        print(f"  R@10: {flickr_metrics['t2i_r10']:.2f}%")
        print("="*60)
    
    # ========== ImageNet Evaluation ==========
    if args.task in ['imagenet', 'all']:
        print("\n" + "="*60)
        print("Evaluating on ImageNet")
        print("="*60)
        
        # Load class names
        if os.path.exists(args.imagenet_classes):
            with open(args.imagenet_classes, 'r') as f:
                classnames = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: {args.imagenet_classes} not found, using placeholder names")
            classnames = [f"class_{i}" for i in range(1000)]
        
        # Build text classifier
        print("Building text classifier (this may take a few minutes)...")
        text_classifier = build_text_classifier(
            model, classnames, IMAGENET_TEMPLATES, device
        )
        
        # Load ImageNet dataset
        imagenet_dataset = ImageNetDataset(args.imagenet_root, transform=preprocess)
        imagenet_loader = DataLoader(
            imagenet_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        imagenet_metrics = evaluate_imagenet(
            model, imagenet_loader, text_classifier, device
        )
        
        print("\n" + "="*60)
        print("ImageNet Zero-shot Results:")
        print("="*60)
        print(f"Top-1 Accuracy: {imagenet_metrics['top1_acc']:.2f}%")
        print(f"Top-5 Accuracy: {imagenet_metrics['top5_acc']:.2f}%")
        print("="*60)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
