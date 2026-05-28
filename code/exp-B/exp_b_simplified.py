"""
Exp-B Simplified: 通过Surgery直接修改算子
不继承，不重写forward，直接替换内部计算逻辑

root@dsw-660270-6c4b9c4ff4-52c54:/mnt/workspace# python exp_b_simplified.py 

============================================================
Training: structured
============================================================
[Surgery] Replaced 12 Attention blocks with BaguaAttention
✓ Layer 0: Patched to mode 'structured'
✓ Layer 1: Patched to mode 'structured'
✓ Layer 2: Patched to mode 'structured'
✓ Layer 3: Patched to mode 'structured'
✓ Layer 4: Patched to mode 'structured'
✓ Layer 5: Patched to mode 'structured'
✓ Layer 6: Patched to mode 'structured'
✓ Layer 7: Patched to mode 'structured'
✓ Layer 8: Patched to mode 'structured'
✓ Layer 9: Patched to mode 'structured'
✓ Layer 10: Patched to mode 'structured'
✓ Layer 11: Patched to mode 'structured'
[Test Dataset] Loaded 50000 samples (max: 50000)
[Test Dataset] Loaded 5000 samples (max: 5000)
Epoch 1: 100%|█████████████████████████████████████████████████████████████| 1563/1563 [06:33<00:00,  3.98it/s]
Val: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:39<00:00,  3.94it/s]
Epoch 1: R@1=37.64%
Epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:33<00:00,  3.98it/s]
Val: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:38<00:00,  4.05it/s]
Epoch 2: R@1=38.28%
Epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:24<00:00,  4.07it/s]
Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:38<00:00,  4.05it/s]
Epoch 3: R@1=38.28%

structured: Best R@1 = 38.28%


============================================================
Training: random_masks
============================================================
[Surgery] Replaced 12 Attention blocks with BaguaAttention
✓ Layer 0: Patched to mode 'random_masks'
✓ Layer 1: Patched to mode 'random_masks'
✓ Layer 2: Patched to mode 'random_masks'
✓ Layer 3: Patched to mode 'random_masks'
✓ Layer 4: Patched to mode 'random_masks'
✓ Layer 5: Patched to mode 'random_masks'
✓ Layer 6: Patched to mode 'random_masks'
✓ Layer 7: Patched to mode 'random_masks'
✓ Layer 8: Patched to mode 'random_masks'
✓ Layer 9: Patched to mode 'random_masks'
✓ Layer 10: Patched to mode 'random_masks'
✓ Layer 11: Patched to mode 'random_masks'
[Test Dataset] Loaded 50000 samples (max: 50000)
[Test Dataset] Loaded 5000 samples (max: 5000)
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:38<00:00,  3.92it/s]
Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:38<00:00,  4.09it/s]
Epoch 1: R@1=0.72%
Epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:37<00:00,  3.93it/s]
Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:40<00:00,  3.88it/s]
Epoch 2: R@1=0.88%
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:39<00:00,  3.91it/s]
Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:39<00:00,  3.97it/s]
Epoch 3: R@1=0.94%

random_masks: Best R@1 = 0.94%

============================================================
Training: all_identity
============================================================
[Surgery] Replaced 12 Attention blocks with BaguaAttention
✓ Layer 0: Patched to mode 'all_identity'
✓ Layer 1: Patched to mode 'all_identity'
✓ Layer 2: Patched to mode 'all_identity'
✓ Layer 3: Patched to mode 'all_identity'
✓ Layer 4: Patched to mode 'all_identity'
✓ Layer 5: Patched to mode 'all_identity'
✓ Layer 6: Patched to mode 'all_identity'
✓ Layer 7: Patched to mode 'all_identity'
✓ Layer 8: Patched to mode 'all_identity'
✓ Layer 9: Patched to mode 'all_identity'
✓ Layer 10: Patched to mode 'all_identity'
✓ Layer 11: Patched to mode 'all_identity'
[Test Dataset] Loaded 50000 samples (max: 50000)
[Test Dataset] Loaded 5000 samples (max: 5000)
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:22<00:00,  4.09it/s]
Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:37<00:00,  4.18it/s]
Epoch 1: R@1=0.12%
Epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:09<00:00,  4.23it/s]
Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:37<00:00,  4.19it/s]
Epoch 2: R@1=0.12%
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [06:03<00:00,  4.30it/s]
Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:37<00:00,  4.22it/s]
Epoch 3: R@1=0.12%

all_identity: Best R@1 = 0.12%


============================================================
Exp-B Summary:
============================================================
all_identity        : 0.12%
structured          : 38.28%
random_masks        : 0.94%


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

from train_bagua_test_pipeline import COCOLightDataset, custom_collate_fn, set_seed
from bagua_moe_coco import BaguaAttention, BaguaRouter, build_2d_local_mask


# ============ 关键：Monkey Patch算子实现 ============

def create_random_mask_operator(seed, threshold):
    """生成随机阈值的mask算子"""
    def operator(attn_logits, L, device, dtype):
        pos = torch.arange(L, device=device)
        distance = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        mask = (distance > threshold).float() * (-1e9)
        return mask.unsqueeze(0).unsqueeze(0).to(dtype)
    return operator


def patch_bagua_forward_for_mode(attn_module, mode):
    """
    直接替换BaguaAttention实例的forward方法
    通过闭包保留原始forward的引用
    """
    original_forward = attn_module.forward
    
    if mode == 'structured':
        # 不做任何修改，使用原始forward
        return
    
    elif mode == 'random_masks':
        # 生成4个随机阈值
        random.seed(42)
        thresholds = [random.randint(3, 15) for _ in range(4)]
        mask_ops = [create_random_mask_operator(42+i, th) for i, th in enumerate(thresholds)]
        
        def random_masks_forward(query, key, value, attn_mask=None, **kwargs):
            """用4个随机mask替换四象"""
            L, N, E = query.shape
            x = query
            dtype = x.dtype
            
            # 路由 + QKV（完全相同）
            routing_weights = attn_module.router(x)
            w = routing_weights.permute(1, 2, 0, 3)
            w_list = [w[..., i:i+1] for i in range(4)]
            
            qkv = F.linear(x, attn_module.in_proj_weight.to(dtype), 
                          attn_module.in_proj_bias.to(dtype) if attn_module.in_proj_bias is not None else None)
            qkv = qkv.reshape(L, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 1, 3, 0, 4)
            q, k, v_qkv = qkv[0], qkv[1], qkv[2]
            
            attn_logits = (q @ k.transpose(-2, -1)) * attn_module.scale
            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask.to(dtype)
            
            # ★ 用4个随机mask替换四象
            operator_outputs = []
            for mask_op in mask_ops:
                mask = mask_op(attn_logits, L, x.device, dtype)
                probs = F.softmax(attn_logits + mask, dim=-1)
                operator_outputs.append(probs @ v_qkv)
            
            # 聚合
            operator_stack = torch.stack(operator_outputs, dim=-1)
            w_combined = torch.stack(w_list, dim=-1)
            output = (operator_stack * w_combined).sum(dim=-1)
            
            output = output.permute(2, 0, 1, 3).reshape(L, N, E)
            return attn_module.out_proj(output.to(attn_module.out_proj.weight.dtype)), None
        
        attn_module.forward = random_masks_forward
    
    elif mode == 'all_identity':
        def identity_forward(query, key, value, attn_mask=None, **kwargs):
            """4个恒等算子（无多样性）"""
            L, N, E = query.shape
            x = query
            dtype = x.dtype

            # 路由 + QKV
            routing_weights = attn_module.router(x)
            # ★ 修复：也需要记录routing history（如果有balance loss）
            if hasattr(attn_module, 'routing_weights_history'):
                attn_module.routing_weights_history.append(routing_weights.detach().mean(dim=(0,1)))

            w = routing_weights.permute(1, 2, 0, 3)

            qkv = F.linear(x, attn_module.in_proj_weight.to(dtype), 
                          attn_module.in_proj_bias.to(dtype) if attn_module.in_proj_bias is not None else None)
            qkv = qkv.reshape(L, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 1, 3, 0, 4)
            q, k, v_qkv = qkv[0], qkv[1], qkv[2]

            attn_logits = (q @ k.transpose(-2, -1)) * attn_module.scale
            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask.to(dtype)

            # ★ 修复：确保梯度流动
            probs = F.softmax(attn_logits, dim=-1)
            base_out = probs @ v_qkv

            # ★ 关键修复：虽然4个输出相同，但要保留路由权重的梯度
            # 方法：用权重聚合，即使权重不同结果也相同
            w_sum = w.sum(dim=-1, keepdim=True)  # [N, H, L, 1]
            output = base_out * (w_sum / 4.0)  # 平均后仍是base_out，但保留梯度

            output = output.permute(2, 0, 1, 3).reshape(L, N, E)
            return attn_module.out_proj(output.to(attn_module.out_proj.weight.dtype)), None

        attn_module.forward = identity_forward


# ============ Surgery ============

def perform_mode_surgery(model, mode, target_layers=None):
    """对指定层的BaguaAttention进行mode配置"""
    if target_layers is None:
        # ★ 改为所有层（因为perform_bagua_surgery已经替换了所有层）
        target_layers = list(range(len(model.visual.transformer.resblocks)))
    
    for layer_idx in target_layers:
        block = model.visual.transformer.resblocks[layer_idx]
        attn = block.attn
        
        # 确保是BaguaAttention实例
        if not isinstance(attn, BaguaAttention):
            continue
        
        # 直接patch forward方法
        patch_bagua_forward_for_mode(attn, mode)
        print(f"✓ Layer {layer_idx}: Patched to mode '{mode}'")
    
    return model


# ============ 训练函数（极简版） ============

def train_mode(mode, cfg):
    """训练单个mode"""
    print(f"\n{'='*60}\nTraining: {mode}\n{'='*60}")
    
    set_seed(45)
    
    # 加载模型
    model, preprocess = clip.load("ViT-B/16", device="cpu", jit=False)
    model = model.float()
    
    # ★ 修复1：不传target_layers参数
    from train_bagua_test_pipeline import perform_bagua_surgery
    model = perform_bagua_surgery(model)  # 所有层都替换成BaguaAttention
    
    # ★ 修复2：再patch成指定mode（默认会patch所有层）
    model = perform_mode_surgery(model, mode)
    model = model.to(cfg.device)
    
    # 冻结参数
    for name, param in model.named_parameters():
        if 'attn' in name and 'router' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 数据 + 优化器（复用您的代码）
    train_dataset = COCOLightDataset(cfg.train_img_root, cfg.train_ann_file, preprocess, cfg.train_max_samples)
    val_dataset = COCOLightDataset(cfg.val_img_root, cfg.val_ann_file, preprocess, cfg.val_max_samples)
    
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练循环（简化版）
    best_r1 = 0
    for epoch in range(cfg.epochs):
        model.train()
        for images, captions_list, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(cfg.device)
            texts = clip.tokenize([caps[0] for caps in captions_list], truncate=True).to(cfg.device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                logits_img, logits_txt = model(images, texts)
                gt = torch.arange(len(images), device=cfg.device)
                loss = (F.cross_entropy(logits_img, gt) + F.cross_entropy(logits_txt, gt)) / 2
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 验证
        model.eval()
        all_img_feat, all_txt_feat = [], []
        with torch.no_grad():
            for images, captions_list, _ in tqdm(val_loader, desc="Val"):
                images = images.to(cfg.device)
                texts = clip.tokenize([caps[0] for caps in captions_list], truncate=True).to(cfg.device)
                
                img_feat = F.normalize(model.encode_image(images), dim=-1)
                txt_feat = F.normalize(model.encode_text(texts), dim=-1)
                
                all_img_feat.append(img_feat.cpu())
                all_txt_feat.append(txt_feat.cpu())
        
        img_feat = torch.cat(all_img_feat)
        txt_feat = torch.cat(all_txt_feat)
        sim = img_feat @ txt_feat.T
        ranks = torch.argsort(sim, dim=1, descending=True)
        gt = torch.arange(len(img_feat)).unsqueeze(1)
        
        r1 = (ranks[:, :1] == gt).any(dim=1).float().mean().item() * 100
        print(f"Epoch {epoch+1}: R@1={r1:.2f}%")
        
        if r1 > best_r1:
            best_r1 = r1
    
    return best_r1


# ============ 主程序 ============

def main():
    class Config:
        def __init__(self):
            self.root_dir = "/mnt/data/coco2017"
            self.train_img_root = os.path.join(self.root_dir, "train2017/train2017")
            self.val_img_root = os.path.join(self.root_dir, "val2017/val2017")
            self.train_ann_file = os.path.join(self.root_dir, "annotations/annotations/captions_train2017.json")
            self.val_ann_file = os.path.join(self.root_dir, "annotations/annotations/captions_val2017.json")
            self.device = "cuda"
            self.batch_size = 32
            self.num_workers = 2
            self.train_max_samples = 50000
            self.val_max_samples = 5000
            self.epochs = 3  # ← 先测试3个epoch
            self.lr = 3e-4
    
    cfg = Config()
    
    modes = ['all_identity'] # modes = ['structured', 'random_masks', 'all_identity']
    results = {}
    
    for mode in modes:
        best_r1 = train_mode(mode, cfg)
        results[mode] = best_r1
        print(f"\n{mode}: Best R@1 = {best_r1:.2f}%\n")
        
    # 手动添加前两个的结果
    results['structured'] = 38.28
    results['random_masks'] = 0.94
    
    # 打印汇总
    print(f"\n{'='*60}\nExp-B Summary:\n{'='*60}")
    for mode, r1 in results.items():
        print(f"{mode:20s}: {r1:.2f}%")


if __name__ == "__main__":
    main()
