"""
Standard MoE Baseline Implementation
对比传统4-expert MoE (独立权重) vs Bagua-MoE (权重共享+拓扑)
exp_a_standard_moe.py、verify_standard_moe.py、exp_a_compare_results.py配合使用


[Step 1/5] Load CLIP ViT-B/16

[Step 2/5] Perform Standard MoE Surgery
✓ Layer 6: Replaced with StandardMoE (4 experts)
✓ Layer 7: Replaced with StandardMoE (4 experts)
✓ Layer 8: Replaced with StandardMoE (4 experts)
✓ Layer 9: Replaced with StandardMoE (4 experts)
✓ Layer 10: Replaced with StandardMoE (4 experts)
✓ Layer 11: Replaced with StandardMoE (4 experts)

=== Standard MoE Surgery Complete ===
Total params: 181.53M
Trainable params: 181.53M (100.00%)
Trainable params: 52.78M

[Step 3/5] Load datasets
[Test Dataset] Loaded 50000 samples (max: 50000)
[Test Dataset] Loaded 5000 samples (max: 5000)

[Step 4/5] Init optimizer

[Step 5/5] Start training
Epoch 1/5: 100%|██████████████████████████████████████████████████████████████████████████| 1563/1563 [07:09<00:00,  3.64it/s, clip=0.482, balance=0.137]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:42<00:00,  3.73it/s]

Epoch 1/5
  Train - Loss: 1.3043, CLIP: 1.1802, Balance: 0.1241
  Val   - I2T R@1: 3.52%, R@5: 12.20%, R@10: 20.56%
  ✓ Saved best model (R@1=3.52%)
Epoch 2/5: 100%|██████████████████████████████████████████████████████████████████████████| 1563/1563 [07:08<00:00,  3.64it/s, clip=0.380, balance=0.129]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:41<00:00,  3.74it/s]

Epoch 2/5
  Train - Loss: 0.9780, CLIP: 0.8594, Balance: 0.1186
  Val   - I2T R@1: 4.16%, R@5: 13.70%, R@10: 21.28%
  ✓ Saved best model (R@1=4.16%)
Epoch 3/5: 100%|██████████████████████████████████████████████████████████████████████████| 1563/1563 [07:08<00:00,  3.65it/s, clip=0.753, balance=0.180]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:42<00:00,  3.73it/s]

Epoch 3/5
  Train - Loss: 0.9666, CLIP: 0.8237, Balance: 0.1429
  Val   - I2T R@1: 4.38%, R@5: 13.90%, R@10: 20.74%
  ✓ Saved best model (R@1=4.38%)
Epoch 4/5: 100%|██████████████████████████████████████████████████████████████████████████| 1563/1563 [07:08<00:00,  3.65it/s, clip=0.259, balance=0.102]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:42<00:00,  3.73it/s]

Epoch 4/5
  Train - Loss: 0.9938, CLIP: 0.8407, Balance: 0.1531
  Val   - I2T R@1: 3.34%, R@5: 11.52%, R@10: 18.98%
Epoch 5/5: 100%|██████████████████████████████████████████████████████████████████████████| 1563/1563 [07:07<00:00,  3.65it/s, clip=0.196, balance=0.102]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:41<00:00,  3.74it/s]

Epoch 5/5
  Train - Loss: 1.0758, CLIP: 0.8758, Balance: 0.2000
  Val   - I2T R@1: 2.94%, R@5: 12.30%, R@10: 19.38%

============================================================
Training Complete! Best R@1: 4.38%
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import clip
import random
import numpy as np
import os
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============ Part 1: 核心类定义 ============
class StandardMoEAttention(nn.Module):
    """
    传统MoE实现：4个expert各有独立的QKV权重
    参数量：4 × (768×768×3) ≈ 7.1M (相对ViT-B/16的86M为8.2%开销)
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_experts: int = 4,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 每个expert独立的QKV投影
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(embed_dim, embed_dim, bias=bias),
                'k_proj': nn.Linear(embed_dim, embed_dim, bias=bias),
                'v_proj': nn.Linear(embed_dim, embed_dim, bias=bias),
            }) for _ in range(num_experts)
        ])
        
        # 路由器：为每个token选择expert组合
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        
        # 输出投影（共享）
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # 负载均衡损失权重
        self.balance_loss_weight = 0.1
        self.routing_weights_history = []
    
    def _init_router_bias(self):
        """初始化路由器：从单expert逐渐分化"""
        nn.init.zeros_(self.router.weight)
        # 可选：warm-start bias (如需类似Bagua的"太极→两仪"演化)
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        标准scaled dot-product attention
        Args:
            q, k, v: [B, T, num_heads, head_dim]
            attn_mask: [T, T] or [B, num_heads, T, T]
        Returns:
            [B, T, num_heads, head_dim]
        """
        B, T, H, D = q.shape
        
        # [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores: [B, H, T, T]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_logits = attn_logits + attn_mask
        
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # [B, H, T, D]
        out = torch.matmul(attn_probs, v)
        # [B, T, H, D]
        out = out.transpose(1, 2).contiguous()
        
        return out
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T, B, D = query.shape
        x = query.permute(1, 0, 2)  # [B, T, D]
        x = x.to(self.router.weight.dtype)
        
        # 1. 路由权重
        routing_logits = self.router(x)
        routing_weights = F.softmax(routing_logits, dim=-1)  # [B, T, 4]
        
        # ★ 修复：记录路由权重用于balance loss计算
        self.routing_weights_history.append(
            routing_weights.detach().mean(dim=(0, 1))  # [4]
        )
        
        # 2. 每个expert计算attention
        expert_outputs = []
        for expert in self.experts:
            q = expert['q_proj'](x).view(B, T, self.num_heads, self.head_dim)
            k = expert['k_proj'](x).view(B, T, self.num_heads, self.head_dim)
            v = expert['v_proj'](x).view(B, T, self.num_heads, self.head_dim)
            
            out = self._attention(q, k, v, attn_mask)  # [B, T, H, D_head]
            out = out.view(B, T, D)
            expert_outputs.append(out)
        
        # 3. 加权聚合
        expert_stack = torch.stack(expert_outputs, dim=-1)  # [B, T, D, 4]
        output = (expert_stack * routing_weights.unsqueeze(2)).sum(dim=-1)
        output = self.out_proj(output)
        
        return output.permute(1, 0, 2), None
    
    def get_balance_loss(self) -> torch.Tensor:
        """计算负载均衡损失"""
        if len(self.routing_weights_history) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 平均routing权重
        avg_weights = torch.stack(self.routing_weights_history).mean(dim=0)  # [4]
        
        # 熵计算
        entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum()
        target_entropy = math.log(self.num_experts)
        
        balance_loss = self.balance_loss_weight * (target_entropy - entropy)
        
        # ★ 清空历史（避免累积）
        self.routing_weights_history = []
        
        return balance_loss


def perform_standard_moe_surgery(
    model: nn.Module,
    target_layers: list = None
) -> nn.Module:
    """
    将CLIP ViT-B/16的注意力层替换为Standard MoE
    
    Args:
        model: CLIP模型
        target_layers: 要替换的层索引，默认[6,7,8,9,10,11] (后6层)
    """
    if target_layers is None:
        target_layers = list(range(6, 12))
    
    visual_encoder = model.visual
    resblocks = visual_encoder.transformer.resblocks
    
    for layer_idx in target_layers:
        if layer_idx >= len(resblocks):
            continue
        
        block = resblocks[layer_idx]
        old_attn = block.attn
        
        # 提取原始配置
        embed_dim = old_attn.embed_dim
        num_heads = old_attn.num_heads
        
        # 创建Standard MoE替换
        new_attn = StandardMoEAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_experts=4,
            dropout=0.0,
            bias=True
        )
        
        # 设备和精度对齐
        device = old_attn.in_proj_weight.device
        dtype = old_attn.in_proj_weight.dtype
        new_attn = new_attn.to(device=device, dtype=dtype)
        
        # 替换
        block.attn = new_attn
        print(f"✓ Layer {layer_idx}: Replaced with StandardMoE (4 experts)")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Standard MoE Surgery Complete ===")
    print(f"Total params: {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    return model


def count_moe_parameters(model: nn.Module) -> dict:
    """统计Standard MoE的参数分布"""
    stats = {
        'router': 0,
        'experts_qkv': 0,
        'out_proj': 0,
        'total_moe': 0
    }
    
    for name, param in model.named_parameters():
        if 'router' in name:
            stats['router'] += param.numel()
        elif any(x in name for x in ['q_proj', 'k_proj', 'v_proj']):
            stats['experts_qkv'] += param.numel()
        elif 'out_proj' in name and 'attn' in name:
            stats['out_proj'] += param.numel()
    
    stats['total_moe'] = stats['router'] + stats['experts_qkv'] + stats['out_proj']
    
    return stats


# ============ Part 2: 复用您的辅助函数 ============
class COCOLightDataset(Dataset):
    """测试专用：仅加载前N个样本，快速验证数据流程"""
    def __init__(self, img_root, ann_file, transform=None, max_samples=5000, max_captions=5):
        self.img_root = img_root
        self.transform = transform
        self.max_captions = max_captions
        self.max_samples = max_samples  # 测试核心：限制总样本数
        
        # 加载标注
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # 构建img_id→filename和img_id→captions映射
        self.images = {img['id']: img['file_name'] for img in data['images']}
        self.img_to_caps = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_caps:
                self.img_to_caps[img_id] = []
            if len(self.img_to_caps[img_id]) < self.max_captions:
                self.img_to_caps[img_id].append(ann['caption'])
        
        # 填充空caption到固定长度
        for img_id in self.img_to_caps:
            while len(self.img_to_caps[img_id]) < self.max_captions:
                self.img_to_caps[img_id].append("")
        
        # 仅保留前max_samples个有效样本（测试核心）
        self.ids = [img_id for img_id in self.img_to_caps.keys() if img_id in self.images][:max_samples]
        print(f"[Test Dataset] Loaded {len(self.ids)} samples (max: {max_samples})")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        filename = self.images[img_id]
        captions = self.img_to_caps[img_id]  # 长度5
        
        # 加载图像（简化错误处理：跳过损坏图）
        try:
            img_path = os.path.join(self.img_root, filename)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Skip broken image {img_path}: {e}")
            # 用随机噪声图替代（避免中断流程）
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        return image, captions, img_id

def custom_collate_fn(batch, max_captions=5):
    """复用Baseline的Collate，保证数据格式一致"""
    images = []
    captions_batch = []
    img_ids = []
    for img, caps, img_id in batch:
        images.append(img)
        captions_batch.append(caps)
        img_ids.append(img_id)
    images = torch.stack(images, dim=0)
    return images, captions_batch, img_ids

def set_seed(seed=42):
    """固定Seed，测试可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
# ============ Part 3: 训练流程（main） ============
def main():
    class TestConfig:
        def __init__(self):
            self.root_dir = "/mnt/data/coco2017"
            self.train_img_root = os.path.join(self.root_dir, "train2017/train2017")
            self.val_img_root = os.path.join(self.root_dir, "val2017/val2017")
            self.train_ann_file = os.path.join(self.root_dir, "annotations/annotations/captions_train2017.json")
            self.val_ann_file = os.path.join(self.root_dir, "annotations/annotations/captions_val2017.json")
            self.model_name = "ViT-B/16"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.batch_size = 32
            self.num_workers = 2
            self.train_max_samples = 50000
            self.val_max_samples = 5000
            self.epochs = 5
            self.lr = 3e-4
            self.balance_loss_weight = 0.1  # 改为0.1，与Bagua一致
    
    cfg = TestConfig()
    set_seed(45)
    
    # 1. 加载模型
    print(f"\n[Step 1/5] Load CLIP {cfg.model_name}")
    model, preprocess = clip.load(cfg.model_name, device="cpu", jit=False)
    model = model.float()
    
    # 2. Standard MoE Surgery
    print("\n[Step 2/5] Perform Standard MoE Surgery")
    model = perform_standard_moe_surgery(model, target_layers=list(range(6, 12)))
    model = model.to(cfg.device)
    
    # 冻结非MoE参数
    for name, param in model.named_parameters():
        if 'attn' in name and any(x in name for x in ['router', 'experts', 'q_proj', 'k_proj', 'v_proj', 'out_proj']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.2f}M")
    
    # 3. 加载数据集
    print(f"\n[Step 3/5] Load datasets")
    train_dataset = COCOLightDataset(
        img_root=cfg.train_img_root,
        ann_file=cfg.train_ann_file,
        transform=preprocess,
        max_samples=cfg.train_max_samples
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: custom_collate_fn(x)
    )
    
    val_dataset = COCOLightDataset(
        img_root=cfg.val_img_root,
        ann_file=cfg.val_ann_file,
        transform=preprocess,
        max_samples=cfg.val_max_samples
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: custom_collate_fn(x)
    )
    
    # 4. 优化器
    print("\n[Step 4/5] Init optimizer")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=1e-5
    )
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. 训练循环
    print(f"\n[Step 5/5] Start training")
    best_r1 = 0.0
    history = []
    
    for epoch in range(cfg.epochs):
        # ===== 训练阶段 =====
        model.train()
        total_loss = 0.0
        total_clip_loss = 0.0
        total_balance_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, (images, captions_list, _) in enumerate(loop):
            images = images.to(cfg.device)
            
            # 每张图只用第一个caption（与Bagua一致）
            texts_raw = [caps[0] if caps[0].strip() else "" for caps in captions_list]
            texts = clip.tokenize(texts_raw, truncate=True).to(cfg.device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                # CLIP对比损失
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(len(images), device=cfg.device)
                
                loss_i = F.cross_entropy(logits_per_image, ground_truth)
                loss_t = F.cross_entropy(logits_per_text, ground_truth)
                clip_loss = (loss_i + loss_t) / 2
                
                # 负载均衡损失（修复版）
                balance_loss = torch.tensor(0.0, device=cfg.device)
                for module in model.modules():
                    if isinstance(module, StandardMoEAttention):
                        balance_loss += module.get_balance_loss()
                
                total_batch_loss = clip_loss + balance_loss
            
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += total_batch_loss.item()
            total_clip_loss += clip_loss.item()
            total_balance_loss += balance_loss.item()
            
            loop.set_postfix({
                'clip': f'{clip_loss.item():.3f}',
                'balance': f'{balance_loss.item():.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_clip = total_clip_loss / len(train_loader)
        avg_balance = total_balance_loss / len(train_loader)
        
        # ===== 验证阶段（修复版） =====
        model.eval()
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for images, captions_list, _ in tqdm(val_loader, desc="Validating"):
                images = images.to(cfg.device)
                
                # ★ 关键修复：只用第一个caption
                texts_raw = [caps[0] if caps[0].strip() else "" for caps in captions_list]
                texts = clip.tokenize(texts_raw, truncate=True).to(cfg.device)
                
                # 编码
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                
                # 归一化
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
        
        # 拼接所有特征
        image_features = torch.cat(all_image_features, dim=0)  # [N, 512]
        text_features = torch.cat(all_text_features, dim=0)    # [N, 512]
        
        # ★ 关键修复：相似度矩阵计算
        # 每张图对应1个caption（不是5个），所以是N×N矩阵
        similarity = image_features @ text_features.T  # [N, N]
        
        # I2T Recall@K
        ranks = torch.argsort(similarity, dim=1, descending=True)
        gt = torch.arange(len(image_features)).unsqueeze(1)  # [N, 1]
        
        r1 = (ranks[:, :1] == gt).any(dim=1).float().mean().item() * 100
        r5 = (ranks[:, :5] == gt).any(dim=1).float().mean().item() * 100
        r10 = (ranks[:, :10] == gt).any(dim=1).float().mean().item() * 100
        
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        print(f"  Train - Loss: {avg_loss:.4f}, CLIP: {avg_clip:.4f}, Balance: {avg_balance:.4f}")
        print(f"  Val   - I2T R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@10: {r10:.2f}%")
        
        # 保存最佳模型
        if r1 > best_r1:
            best_r1 = r1
            torch.save(model.state_dict(), f'standard_moe_best.pth')
            print(f"  ✓ Saved best model (R@1={r1:.2f}%)")
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'clip_loss': avg_clip,
            'balance_loss': avg_balance,
            'i2t_r1': r1,
            'i2t_r5': r5,
            'i2t_r10': r10
        })
        
        # ★ 新增：每个epoch后清空所有模块的routing history
        for module in model.modules():
            if isinstance(module, StandardMoEAttention):
                module.routing_weights_history = []
    
    # 保存训练历史
    with open('standard_moe_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best R@1: {best_r1:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
