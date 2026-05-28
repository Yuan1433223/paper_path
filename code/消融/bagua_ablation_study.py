"""
Bagua-MoE Ablation Study for Science Advances
==============================================
Purpose: Systematic ablation experiments to validate the necessity of each operator.

Ablation Configurations:
- full: All four operators enabled (control group)
- wo_qian: Without Qian (Global) operator
- wo_kun: Without Kun (Local) operator  
- wo_zhen: Without Zhen (High-freq) operator
- wo_xun: Without Xun (Low-freq) operator
- spatial_only: Only Qian + Kun (spatial axis)
- freq_only: Only Zhen + Xun (frequency axis)

Visualizations:
1. Router weight distribution heatmap (per-head, per-token)
2. Expert utilization bar chart (global average usage)
3. Router entropy statistics (quantify routing diversity)
4. Ablation results comparison bar chart

Usage:
    python bagua_ablation_study.py --config full --seed 42
    python bagua_ablation_study.py --config wo_qian --seed 42
    python bagua_ablation_study.py --run_all --seeds 42,43,44
"""

import os
import json
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")


# ==========================================
# Configuration
# ==========================================
ABLATION_CONFIGS = {
    "full": {"qian": True, "kun": True, "zhen": True, "xun": True},
    "wo_qian": {"qian": False, "kun": True, "zhen": True, "xun": True},
    "wo_kun": {"qian": True, "kun": False, "zhen": True, "xun": True},
    "wo_zhen": {"qian": True, "kun": True, "zhen": False, "xun": True},
    "wo_xun": {"qian": True, "kun": True, "zhen": True, "xun": False},
    "spatial_only": {"qian": True, "kun": True, "zhen": False, "xun": False},
    "freq_only": {"qian": False, "kun": False, "zhen": True, "xun": True},
}

OPERATOR_NAMES = ["Qian", "Kun", "Zhen", "Xun"]
OPERATOR_COLORS = ["#E74C3C", "#3498DB", "#F1C40F", "#2ECC71"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# Bagua Router with Ablation Support
# ==========================================
class BaguaRouter(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.router = nn.Linear(embed_dim, num_heads * 4)
        self.last_weights = None

    def forward(self, x):
        L, N, D = x.shape
        # 确保精度一致
        x = x.to(self.router.weight.dtype)
        logits = self.router(x)
        logits = logits.reshape(L, N, self.num_heads, 4)
        self.last_weights = F.softmax(logits, dim=-1)
        return self.last_weights


def build_2d_local_mask(seq_len, window_size, device, dtype=torch.float32):
    if seq_len - 1 < 4:
        return torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    side = int(math.sqrt(seq_len - 1))
    if side * side != seq_len - 1:
        return torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    coords = torch.stack(torch.meshgrid(
        torch.arange(side, device=device), 
        torch.arange(side, device=device), 
        indexing='ij'
    ), dim=-1).float()
    coords = coords.reshape(-1, 2)
    
    dist = torch.cdist(coords, coords)
    mask_patch = torch.zeros_like(dist, dtype=dtype)
    mask_patch[dist > window_size] = float('-inf')
    
    full_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    full_mask[1:, 1:] = mask_patch
    
    return full_mask


# ==========================================
# Bagua Attention with Ablation Support
# ==========================================
class BaguaAttentionAblation(nn.Module):
    """
    BaguaAttention with ablation configuration support.
    Allows disabling specific operators for controlled experiments.
    """
    def __init__(self, original_attn, embed_dim, num_heads, window_size=7, 
                 ablation_config=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.collect_stats = False  # 默认关闭
        
        # Ablation configuration
        self.ablation_config = ablation_config or ABLATION_CONFIGS["full"]
        self.use_qian = self.ablation_config.get("qian", True)
        self.use_kun = self.ablation_config.get("kun", True)
        self.use_zhen = self.ablation_config.get("zhen", True)
        self.use_xun = self.ablation_config.get("xun", True)
        
        # Count active operators for normalization
        self.active_operators = sum([self.use_qian, self.use_kun, self.use_zhen, self.use_xun])
        
        # Shared weights
        self.in_proj_weight = original_attn.in_proj_weight
        self.in_proj_bias = original_attn.in_proj_bias
        self.out_proj = original_attn.out_proj
        
        # Router
        self.router = BaguaRouter(embed_dim, num_heads)
        
        # Cache
        self.register_buffer("local_mask", None)
        
        # Statistics for visualization
        self.routing_stats = {
            "weights": [],
            "entropy": [],
        }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        x = query
        L, N, E = x.shape
        dtype = x.dtype
        
        # Router
        routing_weights = self.router(x.to(self.router.router.weight.dtype)) # 设别问题
        w = routing_weights.permute(1, 2, 0, 3)  # [N, H, L, 4]
        
        # Store for visualization # 爆内存
        # if self.collect_stats:
        #     self.routing_stats["weights"].append(routing_weights.detach().cpu())
            
        if self.collect_stats and len(self.routing_stats["weights"]) < 100:  # 只收集前100个batch
            self.routing_stats["weights"].append(routing_weights.detach().cpu())
        
        # QKV projection
        qkv = F.linear(x, self.in_proj_weight.to(dtype), 
                      self.in_proj_bias.to(dtype) if self.in_proj_bias is not None else None)
        qkv = qkv.reshape(L, N, 3, self.num_heads, self.head_dim).permute(2, 1, 3, 0, 4)
        q, k, v_qkv = qkv[0], qkv[1], qkv[2]
        
        # Base attention scores
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask.to(dtype)
        
        # ========== Four Operators with Ablation ==========
        outputs = []
        weights = []
        
        # Qian (Global)
        if self.use_qian:
            probs_qian = F.softmax(attn_logits, dim=-1)
            out_qian = probs_qian @ v_qkv
            outputs.append(out_qian)
            weights.append(w[..., 0:1])
        
        # Kun (Local)
        if self.use_kun:
            if self.local_mask is None or self.local_mask.shape[0] != L:
                self.local_mask = build_2d_local_mask(L, self.window_size, x.device, dtype)
            probs_kun = F.softmax(attn_logits + self.local_mask, dim=-1)
            out_kun = probs_kun @ v_qkv
            outputs.append(out_kun)
            weights.append(w[..., 1:2])
        
        # Zhen (High-freq)
        if self.use_zhen:
            probs_zhen = F.softmax(attn_logits, dim=-1)
            v_smooth = v_qkv.mean(dim=-2, keepdim=True)
            out_zhen = probs_zhen @ (v_qkv - v_smooth)
            outputs.append(out_zhen)
            weights.append(w[..., 2:3])
        
        # Xun (Low-freq)
        if self.use_xun:
            v_smooth = v_qkv.mean(dim=-2, keepdim=True)
            out_xun = v_smooth.expand_as(v_qkv)
            outputs.append(out_xun)
            weights.append(w[..., 3:4])
        
        # Aggregate
        if len(outputs) == 0:
            # Fallback: standard attention
            probs = F.softmax(attn_logits, dim=-1)
            output = probs @ v_qkv
        else:
            # Normalize weights among active operators
            weights_cat = torch.cat(weights, dim=-1)  # [N, H, L, num_active]
            weights_norm = weights_cat / (weights_cat.sum(dim=-1, keepdim=True) + 1e-8)
            
            output = torch.zeros_like(outputs[0])
            for i, out in enumerate(outputs):
                output = output + weights_norm[..., i:i+1] * out
        
        # Reshape
        output = output.permute(2, 0, 1, 3).reshape(L, N, E)
        final_output = self.out_proj(output.to(self.out_proj.weight.dtype))
        
        return final_output, None

    def get_routing_statistics(self):
        """Compute routing statistics for visualization."""
        if not self.routing_stats["weights"]:
            return None
        
        # Aggregate all recorded weights
        all_weights = torch.cat(self.routing_stats["weights"], dim=1)  # [L, N_total, H, 4]
        
        # Expert utilization (global average)
        utilization = all_weights.mean(dim=(0, 1, 2)).numpy()  # [4]
        
        # Entropy per head
        avg_probs = all_weights.mean(dim=(0, 1))  # [H, 4]
        entropy_per_head = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=-1).numpy()  # [H]
        
        return {
            "utilization": utilization,
            "entropy_per_head": entropy_per_head,
            "avg_entropy": entropy_per_head.mean(),
            "raw_weights": all_weights,
        }

    def clear_stats(self):
        self.routing_stats = {"weights": [], "entropy": []}


# ==========================================
# Dataset
# ==========================================
class COCORetrievalDataset(Dataset):
    def __init__(self, image_dir, ann_file, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        self.id_to_file = {img['id']: img['file_name'] for img in data['images']}
        
        self.image_captions = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_captions:
                self.image_captions[img_id] = []
            self.image_captions[img_id].append(ann['caption'])
        
        self.image_ids = list(self.image_captions.keys())
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, self.id_to_file[img_id])
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        captions = self.image_captions[img_id]
        caption = random.choice(captions)
        
        return image, caption, img_id


# ==========================================
# Surgery Function
# ==========================================
def perform_ablation_surgery(model, ablation_config):
    """Inject BaguaAttentionAblation into CLIP vision transformer."""
    vision_transformer = model.visual.transformer
    
    for block in vision_transformer.resblocks:
        original_attn = block.attn
        bagua_attn = BaguaAttentionAblation(
            original_attn,
            original_attn.embed_dim,
            original_attn.num_heads,
            window_size=7,
            ablation_config=ablation_config
        )
        block.attn = bagua_attn
        
    # ========== 新增：移动到正确设备和精度 ==========
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype  # 通常是 float16
    
    for block in vision_transformer.resblocks:
        block.attn.router = block.attn.router.to(device).to(dtype)
    # ================================================
    
    return model


def get_trainable_params(model):
    """Get trainable parameters (Router weights)."""
    params = []
    param_count = 0
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    for block in model.visual.transformer.resblocks:
        if hasattr(block.attn, 'router'):
            for param in block.attn.router.parameters():
                param.requires_grad = True
                params.append(param)
                param_count += param.numel()
    
    return params, param_count


# ==========================================
# Training and Evaluation
# ==========================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, captions, _ in pbar:
        images = images.to(device)
        texts = clip.tokenize(captions, truncate=True).to(device)
        
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        labels = torch.arange(len(images), device=device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    
    all_image_features = []
    all_text_features = []
    
    for images, captions, _ in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        texts = clip.tokenize(captions, truncate=True).to(device)
        
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())
    
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # Normalize
    all_image_features = F.normalize(all_image_features, dim=-1)
    all_text_features = F.normalize(all_text_features, dim=-1)
    
    # Similarity
    sim_i2t = all_image_features @ all_text_features.T
    sim_t2i = sim_i2t.T
    
    n = sim_i2t.shape[0]
    
    # I2T Retrieval
    i2t_ranks = []
    for i in range(n):
        rank = (sim_i2t[i].argsort(descending=True) == i).nonzero().item()
        i2t_ranks.append(rank)
    i2t_ranks = torch.tensor(i2t_ranks)
    
    # T2I Retrieval
    t2i_ranks = []
    for i in range(n):
        rank = (sim_t2i[i].argsort(descending=True) == i).nonzero().item()
        t2i_ranks.append(rank)
    t2i_ranks = torch.tensor(t2i_ranks)
    
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
# Visualization Functions
# ==========================================
def collect_routing_stats(model):
    """Collect routing statistics from all BaguaAttention layers."""
    all_stats = []
    for block in model.visual.transformer.resblocks:
        if hasattr(block.attn, 'get_routing_statistics'):
            stats = block.attn.get_routing_statistics()
            if stats is not None:
                all_stats.append(stats)
    return all_stats


def clear_routing_stats(model):
    """Clear routing statistics from all layers."""
    for block in model.visual.transformer.resblocks:
        if hasattr(block.attn, 'clear_stats'):
            block.attn.clear_stats()


def visualize_expert_utilization(all_stats, config_name, save_dir):
    """Generate expert utilization bar chart."""
    if not all_stats:
        return
    
    # Average utilization across all layers
    avg_utilization = np.mean([s["utilization"] for s in all_stats], axis=0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(OPERATOR_NAMES, avg_utilization, color=OPERATOR_COLORS, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_utilization):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title(f'Expert Utilization Distribution\nConfiguration: {config_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Operator (Four Images)', fontsize=12)
    plt.ylabel('Average Selection Probability', fontsize=12)
    plt.ylim(0, max(avg_utilization) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'expert_utilization_{config_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_utilization


def visualize_router_entropy(all_stats, config_name, save_dir):
    """Generate router entropy statistics visualization."""
    if not all_stats:
        return
    
    # Entropy per layer
    layer_entropies = [s["avg_entropy"] for s in all_stats]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Layer-wise entropy
    axes[0].bar(range(len(layer_entropies)), layer_entropies, color='steelblue', edgecolor='black')
    axes[0].axhline(y=np.log(4), color='red', linestyle='--', label='Max Entropy (uniform)')
    axes[0].set_title(f'Router Entropy per Layer\nConfig: {config_name}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Layer Index', fontsize=11)
    axes[0].set_ylabel('Average Entropy', fontsize=11)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Head-wise entropy (from first layer as example)
    head_entropy = all_stats[0]["entropy_per_head"]
    axes[1].bar(range(len(head_entropy)), head_entropy, color='coral', edgecolor='black')
    axes[1].axhline(y=np.log(4), color='red', linestyle='--', label='Max Entropy')
    axes[1].set_title(f'Router Entropy per Head (Layer 0)\nConfig: {config_name}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Head Index', fontsize=11)
    axes[1].set_ylabel('Entropy', fontsize=11)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'router_entropy_{config_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "mean_entropy": np.mean(layer_entropies),
        "std_entropy": np.std(layer_entropies),
        "max_possible": np.log(4),
    }


def visualize_token_heatmap(all_stats, config_name, save_dir):
    """Generate token-level routing heatmap."""
    if not all_stats or all_stats[0]["raw_weights"] is None:
        return
    
    # Use first layer, first sample
    weights = all_stats[0]["raw_weights"][:, 0, :, :]  # [L, H, 4]
    L, H, _ = weights.shape
    
    # Determine dominant operator per token per head
    dominant = weights.argmax(dim=-1).numpy()  # [L, H]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, (ax, op_name, color) in enumerate(zip(axes.flat, OPERATOR_NAMES, OPERATOR_COLORS)):
        # Probability of selecting this operator
        prob_map = weights[:, :, idx].numpy()  # [L, H]
        
        im = ax.imshow(prob_map.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'{op_name} Selection Probability', fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Index', fontsize=10)
        ax.set_ylabel('Head Index', fontsize=10)
        plt.colorbar(im, ax=ax, label='Probability')
    
    plt.suptitle(f'Token-wise Router Selection Heatmap\nConfiguration: {config_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'token_heatmap_{config_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(results, save_dir):
    """Generate ablation study comparison bar chart."""
    configs = list(results.keys())
    i2t_r1 = [results[c]["i2t_r1"] for c in configs]
    t2i_r1 = [results[c]["t2i_r1"] for c in configs]
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, i2t_r1, width, label='I2T R@1', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, t2i_r1, width, label='T2I R@1', color='#E74C3C', edgecolor='black')
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Ablation Configuration', fontsize=12)
    ax.set_ylabel('Recall@1 (%)', fontsize=12)
    ax.set_title('Bagua-MoE Ablation Study Results\nMS-COCO Retrieval', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Mark the full model
    ax.axhline(y=results["full"]["i2t_r1"], color='#3498DB', linestyle='--', alpha=0.5)
    ax.axhline(y=results["full"]["t2i_r1"], color='#E74C3C', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_latex_table(results, save_dir):
    """Generate LaTeX table for paper."""
    configs = list(results.keys())
    
    table = r"""
\begin{table}[h]
\centering
\caption{Ablation Study Results on MS-COCO Retrieval}
\label{tab:ablation}
\begin{tabular}{l|ccc|ccc}
\toprule
\multirow{2}{*}{Configuration} & \multicolumn{3}{c|}{Image-to-Text} & \multicolumn{3}{c}{Text-to-Image} \\
 & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 \\
\midrule
"""
    
    for config in configs:
        r = results[config]
        row = f"{config} & {r['i2t_r1']:.2f} & {r['i2t_r5']:.2f} & {r['i2t_r10']:.2f} & {r['t2i_r1']:.2f} & {r['t2i_r5']:.2f} & {r['t2i_r10']:.2f} \\\\\n"
        table += row
    
    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(save_dir, 'ablation_table.tex'), 'w') as f:
        f.write(table)
    
    print(f"LaTeX table saved to {os.path.join(save_dir, 'ablation_table.tex')}")


# ==========================================
# Main
# ==========================================
def run_single_ablation(config_name, seed, cfg):
    """Run a single ablation experiment."""
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Config: {config_name} | Seed: {seed}")
    print(f"Active operators: {ABLATION_CONFIGS[config_name]}")
    print(f"{'='*60}")
    
    # Load CLIP
    model, preprocess = clip.load("ViT-B/16", device=device)
    
    # Surgery
    ablation_config = ABLATION_CONFIGS[config_name]
    model = perform_ablation_surgery(model, ablation_config)
    
    # Trainable params
    trainable_params, param_count = get_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {param_count:,} ({param_count/total_params*100:.4f}%)")
    
    # Data 数据集注意路径 
    train_dataset = COCORetrievalDataset(
        os.path.join(cfg.data_root, 'train2017/train2017'),
        os.path.join(cfg.data_root, 'annotations/annotations/captions_train2017.json'),
        transform=preprocess,
        max_samples=cfg.train_samples
    )
    val_dataset = COCORetrievalDataset(
        os.path.join(cfg.data_root, 'val2017/val2017'),
        os.path.join(cfg.data_root, 'annotations/annotations/captions_val2017.json'),
        transform=preprocess,
        max_samples=cfg.val_samples
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=1e-5)
    
    # 训练时关闭统计收集
    for block in model.visual.transformer.resblocks:
        if hasattr(block.attn, 'collect_stats'):
            block.attn.collect_stats = False
    
    # Training
    for epoch in range(cfg.epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{cfg.epochs}: avg_loss = {avg_loss:.4f}")
    
    # Collect routing stats before final evaluation
    clear_routing_stats(model)
    
    # 评估时开启统计收集
    for block in model.visual.transformer.resblocks:
        if hasattr(block.attn, 'collect_stats'):
            block.attn.collect_stats = True
    
    # Final evaluation (also collects stats)
    metrics = evaluate(model, val_loader, device)
    
    # Collect stats
    all_stats = collect_routing_stats(model)
    
    print(f"\n{'='*40}")
    print(f"Results [{config_name}]:")
    print(f"  I2T R@1: {metrics['i2t_r1']:.2f}%")
    print(f"  I2T R@5: {metrics['i2t_r5']:.2f}%")
    print(f"  I2T R@10: {metrics['i2t_r10']:.2f}%")
    print(f"  T2I R@1: {metrics['t2i_r1']:.2f}%")
    print(f"  T2I R@5: {metrics['t2i_r5']:.2f}%")
    print(f"  T2I R@10: {metrics['t2i_r10']:.2f}%")
    print(f"{'='*40}\n")
    
    return metrics, all_stats


def main():
    parser = argparse.ArgumentParser(description='Bagua-MoE Ablation Study')
    parser.add_argument('--config', type=str, choices=list(ABLATION_CONFIGS.keys()), default='full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--val_samples', type=int, default=2000)
    parser.add_argument('--data_root', type=str, default='/mnt/data/coco2017')
    parser.add_argument('--output_dir', type=str, default='./ablation_results')
    parser.add_argument('--run_all', action='store_true', help='Run all ablation configs')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated seeds for run_all')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.run_all:
        # Run all configurations
        seeds = [int(s) for s in args.seeds.split(',')]
        all_results = {config: {} for config in ABLATION_CONFIGS.keys()}
        
        for config_name in ABLATION_CONFIGS.keys():
            config_results = []
            for seed in seeds:
                metrics, stats = run_single_ablation(config_name, seed, args)
                config_results.append(metrics)
                
                # Visualizations (only for first seed)
                if seed == seeds[0] and stats:
                    visualize_expert_utilization(stats, config_name, args.output_dir)
                    visualize_router_entropy(stats, config_name, args.output_dir)
                    visualize_token_heatmap(stats, config_name, args.output_dir)
            
            # Average results
            avg_results = {}
            for key in config_results[0].keys():
                values = [r[key] for r in config_results]
                avg_results[key] = np.mean(values)
                avg_results[f"{key}_std"] = np.std(values)
            
            all_results[config_name] = avg_results
        
        # Save all results
        with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate comparison chart
        plot_ablation_comparison(all_results, args.output_dir)
        
        # Generate LaTeX table
        generate_latex_table(all_results, args.output_dir)
        
        print(f"\nAll results saved to {args.output_dir}")
        
    else:
        # Run single configuration
        metrics, stats = run_single_ablation(args.config, args.seed, args)
        
        # Visualizations
        if stats:
            visualize_expert_utilization(stats, args.config, args.output_dir)
            visualize_router_entropy(stats, args.config, args.output_dir)
            visualize_token_heatmap(stats, args.config, args.output_dir)
        
        # Save results
        with open(os.path.join(args.output_dir, f'results_{args.config}_seed{args.seed}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
