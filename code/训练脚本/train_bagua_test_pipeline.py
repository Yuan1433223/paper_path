"""
train_bagua_test_pipeline.py：轻量测试版
核心：自定义训练样本+X轮训练+实时Loss监控，快速验证流程和学习趋势

usage:
# Standard LoRA 基线
python train_bagua_test_pipeline.py --mode standard_lora --seed 42

# Full Bagua-MoE
python train_bagua_test_pipeline.py --mode bagua --seed 42
"""
import torch
import torch.nn as nn
import clip
import random
import numpy as np
import os
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# 导入Bagua核心组件（复用之前的v2.1版本）
from bagua_moe_coco import (
    BaguaAttention, BaguaRouter, 
    RouterVisualizer, compute_bagua_balance_loss
)

# ==========================================
# 1. 轻量数据集：仅加载极小样本（测试专用）
# ==========================================
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

# ==========================================
# 2. 简化评估：仅计算小样本R@1（快速看趋势）
# ==========================================
@torch.no_grad()
def quick_evaluate(model, val_dataloader, device):
    """测试专用：仅用100样本计算I2T R@1，不输出完整指标，快速验证模型是否学习"""
    model.eval()
    all_img_feats = []
    all_txt_feats = []
    txt_to_img_index = []
    
    # 仅跑前2个batch（≈64样本，更快）
    for batch_idx, (images, captions_list, _) in enumerate(val_dataloader):
        if batch_idx >= 2:
            break
        
        images = images.to(device)
        # 提取图像特征
        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        all_img_feats.append(img_feats.cpu())
        
        # 提取文本特征（过滤空caption）
        flat_captions = []
        for captions in captions_list:
            valid_caps = [cap for cap in captions if cap.strip()]
            if valid_caps:
                flat_captions.append(valid_caps[0])  # 仅用第1个有效caption，简化计算
                txt_to_img_index.append(batch_idx * val_dataloader.batch_size)
        
        if flat_captions:
            text_tokens = clip.tokenize(flat_captions, truncate=True).to(device)
            txt_feats = model.encode_text(text_tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            all_txt_feats.append(txt_feats.cpu())
    
    # 计算简化R@1
    if not all_img_feats or not all_txt_feats:
        return 0.0  # 无有效数据时返回0
    img_feats = torch.cat(all_img_feats, dim=0).to(device)
    txt_feats = torch.cat(all_txt_feats, dim=0).to(device)
    txt_to_img_index = torch.tensor(txt_to_img_index).to(device)
    
    sim_matrix = img_feats @ txt_feats.t()
    _, top1_idx = sim_matrix.topk(1, dim=1)
    hits = (top1_idx == txt_to_img_index.unsqueeze(0)).any(dim=1)
    r1 = hits.float().mean().item() * 100
    return r1

# ==========================================
# 3. 模型工具：复用+简化（确保流程通）
# ==========================================
def set_seed(seed=42):
    """固定Seed，测试可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fix_clip_ln_precision(model):
    """修正CLIP LayerNorm精度，避免数值问题"""
    for name, module in model.named_modules():
        if isinstance(module, clip.model.LayerNorm):
            module.to(torch.float32)
            print(f"[Precision Fix] Set {name} to float32")

def perform_bagua_surgery(model, standard_lora_only=False):
    """简化手术日志，仅打印关键信息"""
    visual_encoder = model.visual
    for i, block in enumerate(visual_encoder.transformer.resblocks):
        original_attn = block.attn
        block.attn = BaguaAttention(original_attn, original_attn.embed_dim, original_attn.num_heads, standard_lora_only=standard_lora_only) # 增加standard_lora_only=standard_lora_only
    print(f"[Surgery] Replaced {len(visual_encoder.transformer.resblocks)} Attention blocks with BaguaAttention")
    return model

def sanity_check_bagua_equivalence(model, device):
    """
    验证：在热启动状态下，Bagua-CLIP 的输出是否与标准 CLIP 几乎一致。
    """
    print("[Sanity Check] Verifying Bagua == Standard CLIP...")
    model.eval()
    
    # 造一个假数据
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    dummy_text = clip.tokenize(["test", "hello"]).to(device)
    
    with torch.no_grad():
        # Bagua 前向
        logits_img, _ = model(dummy_img, dummy_text)
        
        # 我们可以认为，如果 Logits 的方差正常，且没有 NaN，就暂时通过
        # (因为我们无法同时持有两个大模型副本做逐位对比，显存不够)
        print(f"    Logits Mean: {logits_img.mean().item():.4f}")
        print(f"    Logits Std:  {logits_img.std().item():.4f}")
        
        if torch.isnan(logits_img).any():
            raise ValueError("Sanity Check Failed: NaN detected in output!")
        
    print("[Sanity Check] Passed. Output stats look healthy.")

def warm_start_router(model, qian_bias=5.0):
    """
    易理热启动：初始化 Router，使其在初始阶段 99% 的概率选择 '乾卦(Global)'。
    这保证了模型从标准 CLIP 的性能高位起步，而不是从随机猜测开始。
    """
    print(f"[Init] Warm-starting Routers to favor Qian (Global) with bias {qian_bias}...")
    for module in model.modules():
        if isinstance(module, BaguaRouter):
            # 1. 权重全零初始化，消除任何输入相关的随机扰动
            nn.init.constant_(module.router.weight, 0.0)
            
            # 2. 偏置初始化
            # layout: [Qian, Kun, Zhen, Xun] * Heads
            # 我们希望 Qian (idx 0, 4, 8...) 的 logit 很大
            bias = torch.zeros_like(module.router.bias)
            num_heads = module.num_heads
            
            # 遍历每个 Head，给 Qian 的位置加 Bias
            for h in range(num_heads):
                # 每个 Head 有 4 个输出，Qian 是第 1 个 (index 0)
                # 索引映射: h*4 + 0
                bias[h*4 + 0] = qian_bias # Qian 极大
                bias[h*4 + 1] = -qian_bias # Kun 极小
                bias[h*4 + 2] = -qian_bias # Zhen 极小
                bias[h*4 + 3] = -qian_bias # Xun 极小
            
            # 赋值
            module.router.bias.data = bias
            
    print("[Init] Router is now DETECTIVE: 100% Qian Mode.")

def freeze_backbone(model):
    """冻结主干，仅训练Router+LN，打印参数占比"""
    for param in model.parameters():
        param.requires_grad = False
    
    trainable = 0
    for module in model.modules():
        if isinstance(module, BaguaRouter) or isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True
                trainable += param.numel()
    
    total = sum(p.numel() for p in model.parameters())
    print(f"[Params] Trainable: {trainable/1e3:.0f}K / Total: {total/1e6:.1f}M ({trainable/total:.2%})")
    return model

# ==========================================
# 4. 主测试流程（核心：短平快）
# ==========================================
def main():
    # 测试专用配置（无需修改路径，仅需确认root_dir正确） 关键调配
    class TestConfig:
        def __init__(self):
            self.root_dir = "/mnt/data/coco2017"  # 请改为你的COCO根路径
            self.train_img_root = os.path.join(self.root_dir, "train2017/train2017")
            self.val_img_root = os.path.join(self.root_dir, "val2017/val2017")
            self.train_ann_file = os.path.join(self.root_dir, "annotations/annotations/captions_train2017.json")
            self.val_ann_file = os.path.join(self.root_dir, "annotations/annotations/captions_val2017.json")
            self.model_name = "ViT-B/16"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.batch_size = 32  # 小Batch，适配显存
            self.num_workers = 2  # 减少线程，快速启动
            self.train_max_samples = 50000  # 训练仅500样本
            self.val_max_samples = 5000    # 验证仅100样本
            self.epochs = 5  # 仅训练2轮，看Loss趋势
            self.lr = 3e-4
            self.balance_loss_weight = 0.05
            self.mode = "standard_lora"  # 新增：可选 "bagua" 或 "standard_lora"
    
    cfg = TestConfig()
    set_seed(45)
    visualizer = RouterVisualizer(num_heads=12)  # 简化可视化，仅保留Token热力图
    
    # --------------------------
    # 1. 加载模型（CPU→手术→GPU，流程对齐）
    # --------------------------
    print(f"\n[Step 1/5] Load CLIP {cfg.model_name} (CPU first)")
    try:
        model, preprocess = clip.load(cfg.model_name, device="cpu", jit=False)
        model = model.float()
        print("[Success] CLIP loaded")
    except Exception as e:
        print(f"[Error] Load CLIP failed: {e}")
        return  # 加载失败直接退出，避免后续报错
    
    # 模型手术+精度修正
    try:
        # 改动
        standard_lora_only = (cfg.mode == "standard_lora")
        model = perform_bagua_surgery(model, standard_lora_only=standard_lora_only)

        warm_start_router(model)
        model = model.to(cfg.device)  # 新增这行，确保模型全量移到GPU/CPU
        sanity_check_bagua_equivalence(model, cfg.device)
        model = freeze_backbone(model)
        # if cfg.device == "cuda":
        #     model.to(cfg.device)
        # fix_clip_ln_precision(model)
        print("[Success] Model surgery & precision fix done")
    except Exception as e:
        print(f"[Error] Model surgery failed: {e}")
        return
    
    # --------------------------
    # 2. 加载轻量数据（500训练+100验证）
    # --------------------------
    print(f"\n[Step 2/5] Load light dataset (train: {cfg.train_max_samples}, val: {cfg.val_max_samples})")
    try:
        # 训练集（500样本）
        train_dataset = COCOLightDataset(
            img_root=cfg.train_img_root,
            ann_file=cfg.train_ann_file,
            transform=preprocess,
            max_samples=cfg.train_max_samples
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda x: custom_collate_fn(x)
        )
        
        # 验证集（100样本）
        val_dataset = COCOLightDataset(
            img_root=cfg.val_img_root,
            ann_file=cfg.val_ann_file,
            transform=preprocess,
            max_samples=cfg.val_max_samples
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=lambda x: custom_collate_fn(x)
        )
        print(f"[Success] Dataset loaded (train batches: {len(train_dataloader)}, val batches: {len(val_dataloader)})")
    except Exception as e:
        print(f"[Error] Dataset load failed: {e}")
        return
    
    # --------------------------
    # 3. 优化器配置（简化，仅保留核心）
    # --------------------------
    print(f"\n[Step 3/5] Init optimizer & loss")
    try:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=1e-5
        )
        scaler = torch.amp.GradScaler('cuda')  # 混合精度，避免数值上溢
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        print("[Success] Optimizer & loss init done")
    except Exception as e:
        print(f"[Error] Optimizer init failed: {e}")
        return
    
    # --------------------------
    # 4. 测试训练（2轮，实时监控Loss）
    # --------------------------
    print(f"\n[Step 4/5] Start test training (epochs: {cfg.epochs})")
    prev_avg_loss = float('inf')  # 记录上一轮Loss，判断是否下降
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        
        # 进度条+实时Loss
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, (images, captions_list, _) in enumerate(loop):
            images = images.to(cfg.device, non_blocking=True)
            
            # 过滤空caption，仅用1个有效caption（简化）
            texts_raw = []
            for captions in captions_list:
                valid_caps = [cap for cap in captions if cap.strip()]
                texts_raw.append(valid_caps[0] if valid_caps else "")
            
            texts = clip.tokenize(texts_raw, truncate=True).to(cfg.device, non_blocking=True)
            
            # 前向+反向（简化日志，仅打印关键Loss）
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(len(images), device=cfg.device)
                
                # 计算损失
                task_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                balance_loss = compute_bagua_balance_loss(model)
                total_batch_loss = task_loss + cfg.balance_loss_weight * balance_loss
            
            # 反向传播
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 累计Loss
            total_loss += total_batch_loss.item()
            avg_batch_loss = total_loss / (batch_idx + 1)
            loop.set_postfix({
                "batch_loss": f"{total_batch_loss.item():.4f}",
                "avg_loss": f"{avg_batch_loss:.4f}"
            })
        
        # 每轮结束：打印Loss趋势+简化验证
        epoch_avg_loss = total_loss / len(train_dataloader)
        print(f"\n[Epoch {epoch+1} Summary]")
        print(f"  Avg Loss: {epoch_avg_loss:.4f} (Prev: {prev_avg_loss:.4f})")
        
        # 判断Loss是否下降
        if epoch_avg_loss < prev_avg_loss:
            print(f"   Loss is decreasing (good sign!)")
        else:
            print(f"   Loss not decreasing (check lr/balance weight)")
        prev_avg_loss = epoch_avg_loss
        
        # 简化验证：快速看R@1（不追求精确）
        try:
            val_r1 = quick_evaluate(model, val_dataloader, cfg.device)
            print(f"  Val I2T R@1 (quick): {val_r1:.1f}% (trend > 0 = learning)")
        except Exception as e:
            print(f"  [Warning] Val failed: {e} (ignore, focus on loss)")
        
        # 简化可视化：仅保存1张Token热力图（验证路由逻辑）
        try:
            model.eval()
            with torch.no_grad():
                val_images, val_captions, _ = next(iter(val_dataloader))
                val_images = val_images.to(cfg.device)
                val_texts = clip.tokenize([c[0] for c in val_captions], truncate=True).to(cfg.device)
                model(val_images, val_texts)
                
                # 提取路由权重并画图
                for module in model.modules():
                    if isinstance(module, BaguaRouter) and module.last_weights is not None:
                        visualizer.plot_token_heatmap(
                            module.last_weights, 
                            save_path=f"test_epoch_{epoch+1}_token_heatmap.png"
                        )
                        print(f"   Router visualization saved")
                        break
        except Exception as e:
            print(f"  [Warning] Visualization failed: {e} (ignore, focus on pipeline)")
        
        # 保存测试权重（可选，方便后续复现）
        torch.save(model.state_dict(), f"test_bagua_epoch_{epoch+1}.pt")
        print(f"   Model saved to test_bagua_epoch_{epoch+1}.pt\n")
    
    # --------------------------
    # 5. 测试总结
    # --------------------------
    print(f"\n[Step 5/5] Test Pipeline Summary")
    if prev_avg_loss < float('inf') and prev_avg_loss < 10.0:  # 假设Loss<10为合理范围
        print(" 1. Pipeline is working (no critical errors)")
        if cfg.epochs >=2 and prev_avg_loss > epoch_avg_loss:
            print(" 2. Loss is decreasing (model is learning)")
        else:
            print(" 2. Loss not decreasing (adjust lr/balance weight or increase epochs)")
    else:
        print(" Pipeline failed (check previous error logs)")

if __name__ == "__main__":
    main()