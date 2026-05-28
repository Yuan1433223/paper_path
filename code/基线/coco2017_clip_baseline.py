"""
基线结果
# Terminal result:
Image Features: torch.Size([5000, 512])
Text Features:  torch.Size([25000, 512])
[Info] Computing similarity matrix...
[Info] Calculating Image-to-Text Recall...
[Info] Calculating Text-to-Image Recall...

========== COCO Zero-Shot Results ==========
Image-to-Text Retrieval:
  R@1:  51.74%
  R@5:  76.76%
  R@10: 84.28%
------------------------------
Text-to-Image Retrieval:
  R@1:  32.71%
  R@5:  57.76%
  R@10: 68.24%
============================================
 SUCCESS: R@1 > 50%. Baseline verified.
 

"""

# 1202 - 未加入严谨基线
import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import math


# ==================== 初期检验 ====================
class StandardAttentionWrapper(nn.Module):
    """
    一个标准的注意力包装器，用于“恒等手术”测试。
    它的行为必须与 nn.MultiheadAttention 完全一致。
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # 内部包含一个标准的多头注意力模块
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

    def forward(self, x, k=None, v=None, attn_mask=None, key_padding_mask=None, need_weights=False):
        # 直接调用内部的标准注意力模块，并保持返回值格式一致
        # CLIP ViT 的 attn_mask 是共享的，需要传入
        return self.attn(x, x, x, need_weights=need_weights, attn_mask=attn_mask)


# ==================== 配置管理 ====================
class Config:
    def __init__(self):
        # 路径配置 (根据你的实际路径修改)
        self.root_dir = "/mnt/data/coco2017"
        self.img_root = os.path.join(self.root_dir, "val2017/val2017")
        self.ann_file = os.path.join(self.root_dir, "annotations/annotations/captions_val2017.json")
        
        # 模型配置
        self.model_name = "ViT-B/16"  # 标准ViT Baseline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 推理配置
        self.batch_size = 128         # 24G显存可设128-256
        self.num_workers = 8
        self.debug = False             # 【关键】True=只跑100张图验证; False=跑全量5k
        self.debug_limit = 100        # 调试模式下的样本数
        self.max_captions_per_img = 5 # COCO默认每张图5个caption，统一填充到这个长度

        
# ==================== 自适应注意力归一化层 ====================
class AdaptiveAttentionNorm(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.norm = nn.LayerNorm(d_head)

    def forward(self, x):
        return self.norm(x.float()).to(x.dtype) * self.scale

    
# ==================== BaguaMHA ====================
class BaguaMHA(nn.Module):
    """
    Bagua-MHA: 最终决定版 (Final Definitive Version)
    - 完全隔离 Zero-Shot 和 Fine-tuning 执行路径
    """
    def __init__(self, d_model=768, num_heads=8, rank=8, dropout=0.1, window_size=7):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.window_size = window_size

        self.W_q_shared = nn.Linear(d_model, d_model)
        self.W_k_shared = nn.Linear(d_model, d_model)
        self.W_v_shared = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.lora_A_q = nn.Parameter(torch.randn(num_heads, d_model, rank) * 0.02)
        self.lora_B_q = nn.Parameter(torch.zeros(num_heads, rank, self.d_head))
        self.lora_A_k = nn.Parameter(torch.randn(num_heads, d_model, rank) * 0.02)
        self.lora_B_k = nn.Parameter(torch.zeros(num_heads, rank, self.d_head))
        self.lora_A_v = nn.Parameter(torch.randn(num_heads, d_model, rank) * 0.02)
        self.lora_B_v = nn.Parameter(torch.zeros(num_heads, rank, self.d_head))

        self.functional_gates = nn.Parameter(torch.full((num_heads,), -10.0))
        self.attn_norm = AdaptiveAttentionNorm(self.d_head)
        
        self.smooth_conv = nn.Conv1d(
            self.d_head, self.d_head, kernel_size=3, padding=1, groups=self.d_head, bias=False
        )
        with torch.no_grad():
            kernel = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3).repeat(self.d_head, 1, 1)
            self.smooth_conv.weight.copy_(kernel)

        self.arbiter_gate = nn.Sequential(
            nn.Linear(self.d_head * 7, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 7), nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)
        self.is_finetuning = False

    def _apply_lora(self, x, lora_A, lora_B):
        mid = torch.einsum('btd,hdr->bthr', x.float(), lora_A.float())
        out = torch.einsum('bthr,hrd->bthd', mid, lora_B.float())
        return out.to(x.dtype)

    def get_functional_priors(self, x):
        x_float = x.float()
        x_norm = x_float.norm(p=2, dim=-1, keepdim=True)
        x_min, x_max = x_norm.min(dim=1, keepdim=True)[0], x_norm.max(dim=1, keepdim=True)[0]
        x_norm_score = (x_norm - x_min) / (x_max - x_min + 1e-6)

        x_diff = x_float[:, 1:, :] - x_float[:, :-1, :] 
        grad_mag = x_diff.norm(p=2, dim=-1) 
        grad_mag = torch.cat([grad_mag, grad_mag[:, -1:]], dim=1).unsqueeze(-1)
        
        return x_norm_score.to(x.dtype), grad_mag.to(x.dtype)

    def forward(self, x, k=None, v=None, attn_mask=None, key_padding_mask=None, need_weights=False, print_debug=False):
        # [T, B, D] -> [B, T, D]
        x = x.permute(1, 0, 2)
        B, T, D = x.shape
        dtype = x.dtype

        # 基础投影 (所有路径共用)
        q_base = self.W_q_shared(x).view(B, T, self.num_heads, self.d_head)
        k_base = self.W_k_shared(x).view(B, T, self.num_heads, self.d_head)
        v_base = self.W_v_shared(x).view(B, T, self.num_heads, self.d_head)

        if not self.is_finetuning:
            # === Zero-Shot Path: Standard, Numerically Stable MHA ===
            q, k, v = q_base, k_base, v_base
            q, k = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            q_norm, k_norm = self.attn_norm(q), self.attn_norm(k)
            attn_scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))
            
            if attn_mask is not None: attn_scores += attn_mask

            attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype)
            out_heads = torch.matmul(attn_probs, v)
            
            output = self.out_proj(out_heads.permute(0, 2, 1, 3).reshape(B, T, D))

        else:
            # === Fine-tuning Path: Activate Bagua Functions ===
            q = q_base + self._apply_lora(x, self.lora_A_q, self.lora_B_q)
            k = k_base + self._apply_lora(x, self.lora_A_k, self.lora_B_k)
            v = v_base + self._apply_lora(x, self.lora_A_v, self.lora_B_v)
            
            gates = torch.sigmoid(self.functional_gates)

            # Vector-level modifications
            gate_h2, gate_h3 = gates[2], gates[3]
            q_h2, v_h3 = q[:, :, 2, :], v[:, :, 3, :]
            
            q_h2_diff = q_h2.clone().float()
            q_h2_diff[:, 1:, :] -= q_h2_diff[:, :-1, :]
            q[:, :, 2, :] = (1 - gate_h2) * q_h2 + gate_h2 * q_h2_diff.to(dtype)

            v_h3_smooth = self.smooth_conv(v_h3.permute(0, 2, 1).float()).permute(0, 2, 1)
            v[:, :, 3, :] = (1 - gate_h3) * v_h3 + gate_h3 * v_h3_smooth.to(dtype)
            
            q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            
            q_norm, k_norm = self.attn_norm(q), self.attn_norm(k)
            attn_scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))
            
            # Gated biases
            x_salience, x_gradient = self.get_functional_priors(x)
            salience_bias, boundary_bias = x_salience.transpose(1, 2), x_gradient.transpose(1, 2)
            idx = torch.arange(T, device=x.device)
            dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            local_mask = (dist > self.window_size).float() * -1e9

            attn_scores[:, 0] += gates[0] * torch.ones_like(attn_scores[:, 0])
            attn_scores[:, 1] += gates[1] * local_mask
            attn_scores[:, 4] += gates[4] * salience_bias
            attn_scores[:, 5] += gates[5] * (1.0 - salience_bias)
            attn_scores[:, 6] += gates[6] * boundary_bias

            if attn_mask is not None: attn_scores += attn_mask

            attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype)
            attn_probs = self.dropout(attn_probs)
            out_heads = torch.matmul(attn_probs, v)

            # Arbiter Aggregation
            gate_h7 = gates[7]
            head_stack = out_heads[:, :7].permute(0, 2, 1, 3)
            gate_input = head_stack.reshape(B, T, -1)
            gate_weights = self.arbiter_gate(gate_input.float()).to(dtype).unsqueeze(-1)
            head_7_out_gated = (head_stack * gate_weights).sum(dim=2)
            
            out_heads_permuted = out_heads.permute(0, 2, 1, 3)
            out_heads_permuted[:, :, 7] = (1 - gate_h7) * out_heads_permuted[:, :, 7] + gate_h7 * head_7_out_gated
            
            output = self.out_proj(out_heads_permuted.reshape(B, T, D))

        # Restore dimension and return
        output = output.permute(1, 0, 2)
        return output, None


# ==================== 数据集加载 ====================
class COCORetrievalDataset(Dataset):
    """
    COCO 2017 验证集加载器
    返回: image, text_list (一张图对应5句话，不足则填充空字符串), img_id
    """
    def __init__(self, img_root, ann_file, transform=None, debug=False, limit=100, max_captions=5):
        self.img_root = img_root
        self.transform = transform
        self.max_captions = max_captions  # 统一每张图的caption数量
        
        print(f"[Info] Loading annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # 构建 img_id -> filename 映射
        self.images = {img['id']: img['file_name'] for img in data['images']}
        
        # 构建 img_id -> captions 映射（不足max_captions则填充空字符串）
        self.img_to_caps = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            if img_id not in self.img_to_caps:
                self.img_to_caps[img_id] = []
            if len(self.img_to_caps[img_id]) < self.max_captions:
                self.img_to_caps[img_id].append(caption)
        
        # 对所有样本填充空字符串到max_captions长度
        for img_id in self.img_to_caps:
            while len(self.img_to_caps[img_id]) < self.max_captions:
                self.img_to_caps[img_id].append("")
            
        # 过滤有效样本 (必须有对应的图片文件)
        self.ids = [img_id for img_id in self.img_to_caps.keys() if img_id in self.images]
        
        # 调试模式：截取部分数据
        if debug:
            print(f"[Warn] Debug mode enabled! Limiting to first {limit} images.")
            self.ids = self.ids[:limit]
            
        print(f"[Info] Dataset loaded. Total images: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        filename = self.images[img_id]
        captions = self.img_to_caps[img_id]  # List[str], 长度固定为max_captions
        
        # 加载图像
        img_path = os.path.join(self.img_root, filename)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, captions, img_id

# ==================== 自定义Collate函数（关键修复） ====================
def custom_collate_fn(batch, max_captions=5):
    """
    处理batch中数据的拼接：
    - images: 正常拼接张量
    - captions: 转为二维列表（batch_size, max_captions）
    - img_ids: 拼接列表
    """
    images = []
    captions_batch = []
    img_ids = []
    
    for img, caps, img_id in batch:
        images.append(img)
        captions_batch.append(caps)  # 每个元素是长度为max_captions的列表
        img_ids.append(img_id)
    
    # 拼接图像张量（torch默认的stack逻辑）
    images = torch.stack(images, dim=0)
    
    return images, captions_batch, img_ids

# ==================== 评估核心逻辑 ====================
@torch.no_grad()
def extract_features(model, dataloader, device, max_captions=5):
    """
    提取所有图像和文本的特征
    输出:
        image_feats: [N_imgs, Dim]
        text_feats:  [N_imgs * max_captions, Dim]
        img_id_map:  记录每行文本属于哪张图
    """
    model.eval()
    all_img_feats = []
    all_txt_feats = []
    txt_to_img_index = [] 
    
    print("[Info] Extracting features...")
    for batch_idx, (images, captions_list_batch, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        
        # 1. 提取图像特征
        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        all_img_feats.append(img_feats.cpu())
        
        # 2. 提取文本特征（修复后的数据结构：captions_list_batch是(batch_size, max_captions)的列表）
        batch_size = images.shape[0]
        flat_captions = []
        
        for i in range(batch_size):
            img_captions = captions_list_batch[i]  # 长度为max_captions
            # 过滤空字符串（避免无意义的文本编码）
            img_captions = [cap for cap in img_captions if cap.strip()]
            flat_captions.extend(img_captions)
            # 记录文本对应的图片索引
            global_img_idx = batch_idx * dataloader.batch_size + i
            txt_to_img_index.extend([global_img_idx] * len(img_captions))

        # 跳过空文本（理论上不会出现，保险起见）
        if not flat_captions:
            continue
            
        # Tokenize
        text_tokens = clip.tokenize(flat_captions, truncate=True).to(device)
        txt_feats = model.encode_text(text_tokens)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        all_txt_feats.append(txt_feats.cpu())

    # 拼接
    all_img_feats = torch.cat(all_img_feats, dim=0)  # [N, D]
    all_txt_feats = torch.cat(all_txt_feats, dim=0)  # [M, D] (M = 有效文本数)
    txt_to_img_index = torch.tensor(txt_to_img_index)  # [M]

    return all_img_feats, all_txt_feats, txt_to_img_index

def compute_recall(img_feats, txt_feats, txt_to_img_index):
    """
    计算 Recall@K
    Args:
        img_feats: [N, D]
        txt_feats: [M, D] (M = 有效文本数)
        txt_to_img_index: [M] 记录第m个文本对应的图片索引(0~N-1)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_feats = img_feats.to(device)
    txt_feats = txt_feats.to(device)
    txt_to_img_index = txt_to_img_index.to(device)

    # 计算相似度矩阵: [N, M]
    print("[Info] Computing similarity matrix...")
    sim_matrix = img_feats @ txt_feats.t() 
    
    num_imgs = img_feats.shape[0]
    num_txts = txt_feats.shape[0]
    
    # ================= Image-to-Text Retrieval (I2T) =================
    print("[Info] Calculating Image-to-Text Recall...")
    scores_i2t = sim_matrix  # [N, M]
    gt_mask = (txt_to_img_index.unsqueeze(0) == torch.arange(num_imgs, device=device).unsqueeze(1))
    
    i2t_res = {}
    for k in [1, 5, 10]:
        _, topk_indices = scores_i2t.topk(k, dim=1)
        hits = torch.gather(gt_mask, 1, topk_indices)
        recall = hits.any(dim=1).float().mean().item()
        i2t_res[f'R@{k}'] = recall * 100

    # ================= Text-to-Image Retrieval (T2I) =================
    print("[Info] Calculating Text-to-Image Recall...")
    scores_t2i = sim_matrix.t()  # [M, N]
    t2i_res = {}
    gt_img_ids = txt_to_img_index  # [M]
    
    for k in [1, 5, 10]:
        _, topk_indices = scores_t2i.topk(k, dim=1)
        hits = (topk_indices == gt_img_ids.unsqueeze(1)).any(dim=1)
        recall = hits.float().mean().item()
        t2i_res[f'R@{k}'] = recall * 100

    return i2t_res, t2i_res

# ==================== 适配Bagua-MHA ====================
def surgery_clip_bagua(model, rank=8):
    print("[Surgery] Starting Bagua-MHA injection...")
    visual_encoder = model.visual
    device = next(model.parameters()).device  # 获取模型设备
    replaced_count = 0
    for i, block in enumerate(visual_encoder.transformer.resblocks):
        old_attn = block.attn
        bagua_attn = BaguaMHA(d_model=768, num_heads=8, rank=rank).to(device)
        
        in_proj, d = old_attn.in_proj_weight, 768
        with torch.no_grad():
            # 复制 Q/K/V 共享权重
            bagua_attn.W_q_shared.weight.copy_(in_proj[:d, :].to(device))
            bagua_attn.W_k_shared.weight.copy_(in_proj[d:2*d, :].to(device))  
            bagua_attn.W_v_shared.weight.copy_(in_proj[2*d:, :].to(device))  
            
            # 复制 Q/K/V 偏置
            bagua_attn.W_q_shared.bias.copy_(old_attn.in_proj_bias[:d].to(device))  
            bagua_attn.W_k_shared.bias.copy_(old_attn.in_proj_bias[d:2*d].to(device))  
            bagua_attn.W_v_shared.bias.copy_(old_attn.in_proj_bias[2*d:].to(device))  
            
            # 复制输出投影层权重和偏置
            bagua_attn.out_proj.weight.copy_(old_attn.out_proj.weight.to(device)) 
            bagua_attn.out_proj.bias.copy_(old_attn.out_proj.bias.to(device))  
        
        block.attn = bagua_attn
        replaced_count += 1
    
    print(f"[Surgery] Successfully replaced {replaced_count} Attention blocks.")    
    return model


def surgery_identity_test(model):
    """
    “恒等手术”测试函数：用一个功能完全相同的包装器替换原生Attention。
    """
    print("[Diagnostic] Starting Identity Surgery Test...")
    visual_encoder = model.visual
    replaced_count = 0
    for i, block in enumerate(visual_encoder.transformer.resblocks):
        old_attn = block.attn
        
        # 创建一个新的、干净的包装器
        d_model = old_attn.embed_dim
        nhead = old_attn.num_heads
        new_attn_wrapper = StandardAttentionWrapper(d_model, nhead)

        # 将旧的权重精确地复制到新的包装器中
        with torch.no_grad():
            new_attn_wrapper.attn.in_proj_weight.copy_(old_attn.in_proj_weight)
            new_attn_wrapper.attn.in_proj_bias.copy_(old_attn.in_proj_bias)
            new_attn_wrapper.attn.out_proj.weight.copy_(old_attn.out_proj.weight)
            new_attn_wrapper.attn.out_proj.bias.copy_(old_attn.out_proj.bias)

        # 替换
        block.attn = new_attn_wrapper
        replaced_count += 1
        
    print(f"[Diagnostic] Replaced {replaced_count} blocks with StandardAttentionWrapper.")
    return model


# ==================== 尊重 ====================
def fix_clip_ln_precision(model):
    """
    遍历模型，将所有 clip.model.LayerNorm 实例的精度设置回 float32。
    """
    for name, module in model.named_modules():
        if isinstance(module, clip.model.LayerNorm):
            module.to(torch.float32)
            print(f"Set {name} to float32")


# ==================== 主流程 ====================
def main():
    cfg = Config()
    
    print(f"========== Configuration ==========")
    print(f"Model: {cfg.model_name}")
    print(f"Device: {cfg.device}")
    print(f"Dataset: {cfg.img_root}")
    print(f"Debug Mode: {cfg.debug}")
    print(f"Max Captions per Image: {cfg.max_captions_per_img}")
    print(f"===================================")

    # 1. 加载模型到 CPU
    print(f"[Info] Loading CLIP {cfg.model_name}...")
    model, preprocess = clip.load(cfg.model_name, device="cpu")
    
    # 2. 在 CPU 上进行手术
    # model = surgery_clip_bagua(model, rank=8)
    model = surgery_identity_test(model)

    # 3. 移动到 GPU 并应用混合精度
    if cfg.device == "cuda":
        print("[Info] Moving model to GPU...")
        model.to(cfg.device)
    
    # ### 关键修复 ###
    # 然后，将所有 LayerNorm 层纠正回 float32
    print("[Info] Correcting LayerNorm precision to float32 for stability...")
    fix_clip_ln_precision(model)
    # ### 修复结束 ###
    
    model.eval()

    # 2. 加载数据（使用自定义collate_fn）
    dataset = COCORetrievalDataset(
        img_root=cfg.img_root,
        ann_file=cfg.ann_file,
        transform=preprocess,
        debug=cfg.debug,
        limit=cfg.debug_limit,
        max_captions=cfg.max_captions_per_img
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda x: custom_collate_fn(x, max_captions=cfg.max_captions_per_img)  # 关键：指定自定义collate
    )

    # 3. 提取特征
    img_feats, txt_feats, txt_to_img_index = extract_features(
        model, dataloader, cfg.device, max_captions=cfg.max_captions_per_img
    )
    
    print(f"[Info] Features extracted.")
    print(f"Image Features: {img_feats.shape}")
    print(f"Text Features:  {txt_feats.shape}")

    # 4. 计算指标
    i2t, t2i = compute_recall(img_feats, txt_feats, txt_to_img_index)

    # 5. 输出结果
    print("\n========== COCO Zero-Shot Results ==========")
    print("Image-to-Text Retrieval:")
    print(f"  R@1:  {i2t['R@1']:.2f}%")
    print(f"  R@5:  {i2t['R@5']:.2f}%")
    print(f"  R@10: {i2t['R@10']:.2f}%")
    print("-" * 30)
    print("Text-to-Image Retrieval:")
    print(f"  R@1:  {t2i['R@1']:.2f}%")
    print(f"  R@5:  {t2i['R@5']:.2f}%")
    print(f"  R@10: {t2i['R@10']:.2f}%")
    print("============================================")
    
    # 6. 自动判断 (仅在非Debug模式下参考)
    if not cfg.debug:
        if i2t['R@1'] > 50.0:
            print("\n SUCCESS: R@1 > 50%. Baseline verified.")
        else:
            print("\n WARNING: R@1 < 50%. Check data alignment or preprocessing.")

if __name__ == "__main__":
    main()