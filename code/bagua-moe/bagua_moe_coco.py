"""
bagua_moe_v2_1.py (核心架构：补全易理对齐+工程鲁棒性)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import clip
from typing import Optional, Tuple

# 使用时：Optional[int] 等价于 int | None（Python 3.10+ 还支持 | 语法）
# 元组直接用 tuple：tuple[int, str] 等价于 Tuple[int, str]

# ==========================================
# 1. 易理核心组件: Head-wise Router (The Li)
# 改进点：添加last_weights存储，支持负载均衡损失；补全设备兼容
# ==========================================
class BaguaRouter(nn.Module):
    """
    Head-wise Router: 实现"十二消息卦"的精细化控制
    Input:  [L, N, D] (Sequence, Batch, Dim)
    Output: [L, N, H, 4] (每个Token，在每个Head，选择4个象的概率)
    四象映射：0=乾(Qian/Global)、1=坤(Kun/Local)、2=震(Zhen/High-Freq)、3=巽(Xun/Low-Freq)
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.router = nn.Linear(embed_dim, num_heads * 4)
        self.last_weights = None  # 存储最新路由权重，用于损失计算和可视化

    def forward(self, x):
        L, N, D = x.shape
        logits = self.router(x)  # [L, N, H*4]
        logits = logits.reshape(L, N, self.num_heads, 4)  # [L, N, H, 4]
        self.last_weights = F.softmax(logits, dim=-1)  # 保存权重（不detach，支持梯度回溯）
        return self.last_weights

# ==========================================
# 2. 易理核心组件: 2D Mask Generator (The Spatial Law)
# 改进点：明确CLS Token的乾卦属性；添加数值安全检查；支持非正方形序列兼容
# ==========================================
def build_2d_local_mask(seq_len, window_size, device, dtype=torch.float32):
    """
    构建符合视觉物理规律的2D欧氏距离掩码，严格对齐易理：
    - CLS Token (索引0)：对应乾卦，全局视野，与所有Patch无距离限制
    - Patch Token (索引1~L-1)：对应坤卦局部逻辑，仅与窗口内Patch交互
    """
    # 处理非标准序列（如截断后的序列）：退化为全通掩码（不影响核心逻辑）
    if seq_len - 1 < 4:  # 至少需要4个Patch才构成2D结构
        return torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    side = int(math.sqrt(seq_len - 1))
    if side * side != seq_len - 1:  # 非正方形Patch排列，退化为全通
        return torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    # 生成Patch坐标（14x14→196个Patch）
    coords = torch.stack(torch.meshgrid(
        torch.arange(side, device=device), 
        torch.arange(side, device=device), 
        indexing='ij'
    ), dim=-1).float()
    coords = coords.reshape(-1, 2)  # [196, 2]
    
    # 计算Patch间欧氏距离矩阵
    dist = torch.cdist(coords, coords)  # [196, 196]
    
    # 构建局部掩码：距离>窗口阈值→-inf（屏蔽交互）
    mask_patch = torch.zeros_like(dist, dtype=dtype)
    mask_patch[dist > window_size] = float('-inf')
    
    # 拼接CLS Token的掩码（全通）
    full_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    full_mask[1:, 1:] = mask_patch  # Patch-Patch按局部掩码交互
    full_mask[0, :] = 0  # CLS→所有Token（乾卦全局视野）
    full_mask[:, 0] = 0  # 所有Token→CLS（乾卦为核心）
    
    return full_mask

# ==========================================
# 3. 易理核心组件: Bagua Attention (The Shu)
# 改进点：设备/ dtype对齐；补全易理注释；优化数值稳定性
# ==========================================
class BaguaAttention(nn.Module):
    """
    Weight-Sharing Mixture-of-Experts (Operators)
    核心逻辑：共享QKV投影权重（数），通过易理驱动的算子组合（理）实现动态注意力
    """
    def __init__(self, original_attn, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        # 共享权重（不变的基础：契合"不易"思想）
        self.in_proj_weight = original_attn.in_proj_weight
        self.in_proj_bias = original_attn.in_proj_bias
        self.out_proj = original_attn.out_proj
        
        # 动态路由（变化的逻辑：契合"变易"思想）
        self.router = BaguaRouter(embed_dim, num_heads)
        
        # 缓存2D局部掩码（避免重复计算）
        self.register_buffer("local_mask", None)

    def forward(
        self,
        query: torch.Tensor,  # 对应原代码的 x
        key: torch.Tensor,    # 自注意力：与 query 相同（CLIP传x）
        value: torch.Tensor,  # 自注意力：与 query 相同（CLIP传x）
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,  # 原生接口必要参数，保留默认值
        is_causal: bool = False,             # 原生接口必要参数，保留默认值
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 2. 自注意力场景：query=key=value，统一用 query 作为输入（原代码的 x）
        x = query
        L, N, E = x.shape
        dtype = x.dtype
        
        # 3. 原有核心逻辑完全不变（后续所有用到 x 的地方都已通过 query 赋值）
        routing_weights = self.router(x)  # 已存储到router.last_weights
        w = routing_weights.permute(1, 2, 0, 3)  # [N, H, L, 4]，适配广播
        
        # 拆分四象权重（单独提取，强化易理映射可读性）
        w_qian = w[..., 0:1]  # 乾卦：全局注意力权重
        w_kun = w[..., 1:2]   # 坤卦：局部注意力权重
        w_zhen = w[..., 2:3]  # 震卦：高频/边缘权重
        w_xun = w[..., 3:4]   # 巽卦：低频/背景权重
        
        # 2. QKV投影（共享权重，参数高效）
        qkv = F.linear(x, self.in_proj_weight.to(dtype), self.in_proj_bias.to(dtype) if self.in_proj_bias is not None else None)
        qkv = qkv.reshape(L, N, 3, self.num_heads, self.head_dim).permute(2, 1, 3, 0, 4)
        q, k, v_qkv = qkv[0], qkv[1], qkv[2]  # [N, H, L, D]（注意变量名避免与参数 value 冲突）
        
        # 3. 基础注意力分数（所有算子共享的基础）
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [N, H, L, L]
        
        # 叠加外部注意力掩码（如padding mask）
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask.to(dtype)
        
        # ==================================================
        # 4. 四象算子执行（The Four Images Operators）：理统御数的核心
        # ==================================================
        # A. 乾卦（Qian/Global）：标准Softmax，全局视野（契合"天行健"）
        probs_qian = F.softmax(attn_logits, dim=-1)
        
        # B. 坤卦（Kun/Local）：2D局部掩码+Softmax，局部聚焦（契合"地势坤"）
        if self.local_mask is None or self.local_mask.shape[0] != L:
            self.local_mask = build_2d_local_mask(L, self.window_size, x.device, dtype)
        probs_kun = F.softmax(attn_logits + self.local_mask, dim=-1)
        
        # C. 震卦（Zhen/High-Freq）：Value差分，提取边缘/动态特征（契合"震为动"）
        v_smooth = v_qkv.mean(dim=-2, keepdim=True)  # 全局背景（直流分量）→ 变量名改v_qkv，避免与参数 value 冲突
        out_zhen = probs_qian @ (v_qkv - v_smooth)    # 差异特征（交流分量）
        
        # D. 巽卦（Xun/Low-Freq）：全局平均背景，提取趋势/静态特征（契合"巽为顺"）
        out_xun = v_smooth.expand_as(v_qkv)  # 广播背景到序列长度
        
        # 四象输出计算（复用注意力概率，无额外计算开销）
        out_qian = probs_qian @ v_qkv  # 乾卦输出 → 用v_qkv
        out_kun = probs_kun @ v_qkv    # 坤卦输出 → 用v_qkv
        
        # ==================================================
        # 5. 动态聚合（Dynamic Aggregation）：四象归一（契合"简易"思想）
        # ==================================================
        output = (w_qian * out_qian + 
                  w_kun * out_kun + 
                  w_zhen * out_zhen + 
                  w_xun * out_xun)
        
        # 维度还原：[N, H, L, D] → [L, N, E]
        output = output.permute(2, 0, 1, 3).reshape(L, N, E)
        
        # ==================================================
        # 6. 处理注意力权重返回（对齐原生接口）
        # ==================================================
        attn_weights = None  # 暂时不需要返回权重，用None占位
        if need_weights:
            # 可选：后续若需可视化，可计算四象聚合后的注意力权重（如加权平均probs_qian/probs_kun）
            # 示例：attn_weights = (w_qian * probs_qian + w_kun * probs_kun).mean(dim=1)  # 简化计算
            pass
        
        # ==================================================
        # 7. 最终投影+返回值（必须返回 (output, attn_weights) 二元组）
        # ==================================================
        final_output = self.out_proj(output.to(self.out_proj.weight.dtype))
        return final_output, attn_weights  # 严格对齐原生接口的返回格式

# ==========================================
# 4. 可解释性组件: Router Recorder + Visualizer
# 改进点：补全可视化逻辑；支持Token/Head双维度分析；保存易理解的图表
# ==========================================
import matplotlib.pyplot as plt
import numpy as np

class RouterVisualizer:
    """
    路由可视化工具：将"十二消息卦"的选择转化为直观图表，实证易理有效性
    支持：1) Token级四象选择热力图 2) Head级四象占比饼图 3) 六十四卦组合分布
    """
    def __init__(self, num_heads=12, four_images=["Qian", "Kun", "Zhen", "Xun"]):
        self.num_heads = num_heads
        self.four_images = four_images
        self.colors = ["#FF0000", "#0000FF", "#FFFF00", "#00FF00"]  # 红(乾)、蓝(坤)、黄(震)、绿(巽)
    
    def _get_token_coords(self, seq_len):
        """获取Token的2D坐标（适配ViT Patch排列）"""
        if seq_len - 1 < 4:
            return np.arange(seq_len)  # 1D坐标 fallback
        side = int(math.sqrt(seq_len - 1))
        cls_coord = np.array([-1, -1])  # CLS Token单独坐标
        patch_coords = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)
        return np.vstack([cls_coord, patch_coords])
    
    def plot_token_heatmap(self, routing_weights, save_path="token_4images_heatmap.png"):
        """
        绘制Token级四象选择热力图（每个Token的四象概率分布）
        routing_weights: [L, N, H, 4] → 取第一个样本（N=0）、第一个Head（H=0）
        """
        routing_weights = routing_weights.cpu()  # 将路由权重复制到 CPU 内存
        L = routing_weights.shape[0]
        coords = self._get_token_coords(L)
        token_weights = routing_weights[:, 0, 0, :].numpy()  # [L, 4]
        
        plt.figure(figsize=(12, 10))
        for i, (img_name, color) in enumerate(zip(self.four_images, self.colors)):
            plt.subplot(2, 2, i+1)
            if len(coords[0]) == 2:  # 2D坐标（CLS+Patch）
                # 绘制Patch热力图
                patch_weights = token_weights[1:, i].reshape(int(math.sqrt(L-1)), int(math.sqrt(L-1)))
                plt.imshow(patch_weights, cmap="YlOrRd", vmin=0, vmax=1)
                plt.title(f"Token-wise {img_name} (Head 0)", fontsize=12)
                plt.colorbar(label="Routing Probability")
                plt.xticks([])
                plt.yticks([])
                # 标记CLS Token
                plt.scatter([-0.5], [-0.5], c=color, s=50, label="CLS Token")
                plt.legend()
            else:  # 1D坐标 fallback
                plt.bar(coords, token_weights[:, i], color=color, alpha=0.7)
                plt.title(f"Token-wise {img_name} (Head 0)", fontsize=12)
                plt.xlabel("Token Index")
                plt.ylabel("Routing Probability")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Token heatmap saved to {save_path}")
    
    def plot_head_distribution(self, routing_weights, save_path="head_4images_dist.png"):
        """
        绘制Head级四象占比饼图（每个Head的四象选择偏好，对应"十二消息卦"）
        """
        routing_weights = routing_weights.cpu()
        N = routing_weights.shape[1]
        head_weights = routing_weights[:, 0, :, :].mean(dim=0).numpy()  # [H, 4]：每个Head的平均概率
        
        plt.figure(figsize=(15, 10))
        for head_idx in range(self.num_heads):
            plt.subplot(3, 4, head_idx+1)
            plt.pie(head_weights[head_idx], labels=self.four_images, colors=self.colors, autopct="%1.1f%%", startangle=90)
            plt.title(f"Head {head_idx+1} (Message Trigram)", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Head distribution saved to {save_path}")
        
        
class RouterRecorder:
    """
    Hook 工具：用于在推理阶段记录 Router 的选择
    用于论文可视化分析 (Figure 3)
    """
    def __init__(self):
        self.records = []
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        # output 是 weights [L, N, H, 4]
        # 关键修正：立即 detach 并转到 CPU，防止显存泄漏或绘图报错
        self.records.append(output.detach().cpu())
        
    def attach(self, model):
        """将 Recorder 挂载到模型所有的 BaguaRouter 上"""
        self.records = []
        self.hooks = []
        print("[RouterRecorder] Attaching hooks to BaguaRouters...")
        count = 0
        for name, module in model.named_modules():
            # 这里使用字符串判断，避免循环引用问题，或者确保 BaguaRouter 已定义
            if "BaguaRouter" in str(type(module)):
                self.hooks.append(module.register_forward_hook(self.hook_fn))
                count += 1
        print(f"[RouterRecorder] Attached to {count} routers.")
                
    def detach(self):
        """移除 Hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
    def get_last_record(self):
        """
        返回最近一次 forward 的记录
        Returns: Tensor of shape [L, N, H, 4]
        """
        if not self.records: return None
        return self.records[-1]

# ==========================================
# 5. 训练辅助组件: 负载均衡损失（契合"阴阳调和"易理）
# ==========================================
def compute_bagua_balance_loss(model, epsilon=1e-8):
    """
    负载均衡损失：强制四象算子被均匀使用，避免路由崩溃（契合易经"阴阳平衡"）
    核心逻辑：计算每个Router的四象概率熵，熵越大表示分布越均匀，损失越小
    """
    total_entropy = 0.0
    router_count = 0
    
    # 遍历所有Router模块，计算熵损失
    for module in model.modules():
        if isinstance(module, BaguaRouter) and module.last_weights is not None:
            # routing_weights: [L, N, H, 4] → 全局平均四象概率
            global_expert_probs = module.last_weights.mean(dim=[0, 1, 2])  # [4]
            # 计算熵（熵越大，分布越均匀）
            entropy = -torch.sum(global_expert_probs * torch.log(global_expert_probs + epsilon))
            total_entropy += entropy
            router_count += 1
    
    if router_count == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    # 损失 = 最大熵 - 平均熵（确保熵趋向最大，即分布均匀）
    max_entropy = torch.log(torch.tensor(4.0, device=total_entropy.device))
    balance_loss = max_entropy - (total_entropy / router_count)
    return balance_loss


