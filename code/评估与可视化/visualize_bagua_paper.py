import torch
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV 用于图像叠加，如果没有请 pip install opencv-python
import os

# 导入我们的核心组件
from bagua_moe_coco import (
    BaguaAttention, BaguaRouter,  RouterRecorder,
    RouterVisualizer, compute_bagua_balance_loss
)
from train_bagua_test_pipeline import perform_bagua_surgery

# ==================== 配置 ====================
MODEL_NAME = "ViT-B/16"
CHECKPOINT_PATH = "test_bagua_epoch_5.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你想测试的图片路径 (建议找 3 张不同风格的图)
# 如果没有本地图片，脚本会尝试生成一张随机噪声图或者请你替换路径
TEST_IMAGE_PATH = "test_image.jpg" 

# ==================== 可视化工具函数 ====================
def overlay_heatmap(original_img_pil, heatmap_2d, color_map=cv2.COLORMAP_JET):
    """
    将 14x14 的热力图叠加到原始图片上
    """
    # 1. 转换原图
    img_np = np.array(original_img_pil)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_np.shape[:2]
    
    # 2. 归一化热力图并缩放
    heatmap_2d = heatmap_2d - heatmap_2d.min()
    if heatmap_2d.max() > 0:
        heatmap_2d = heatmap_2d / heatmap_2d.max()
    
    # 3. 放大到原图尺寸
    heatmap_resized = cv2.resize(heatmap_2d, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    
    # 4. 伪彩色
    heatmap_color = cv2.applyColorMap(heatmap_uint8, color_map)
    
    # 5. 叠加 (原图 0.6 + 热力图 0.4)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay

def visualize_single_image(model, image_path, recorder):
    print(f"[Analysis] Processing {image_path}...")
    
    # 1. 加载和预处理图片
    try:
        raw_image = Image.open(image_path).convert("RGB").resize((224, 224))
    except Exception as e:
        print(f"[Error] Cannot load image: {e}")
        print("Please set TEST_IMAGE_PATH to a valid jpg/png file.")
        return

    # 预处理
    _, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    image_input = preprocess(raw_image).unsqueeze(0).to(DEVICE)
    
    # 2. 前向推理 (Hook 会自动记录 Router 权重)
    with torch.no_grad():
        model.encode_image(image_input)
    
    # 3. 获取记录
    # records 里的 shape: [L, N, H, 4] -> [197, 1, 12, 4]
    last_record = recorder.get_last_record() 
    if last_record is None:
        print("[Error] No routing weights recorded.")
        return
        
    # 4. 数据处理
    # 去掉 Batch 维，对所有 Heads 取平均 (展示层的整体趋势)
    # [197, 12, 4] -> [197, 4]
    weights_avg_head = last_record.squeeze(1).mean(dim=1)
    
    # 拆分 CLS 和 Spatial
    cls_weights = weights_avg_head[0].numpy() # [4]
    spatial_weights = weights_avg_head[1:].numpy() # [196, 4]
    
    # Reshape Spatial to 14x14
    spatial_weights_2d = spatial_weights.reshape(14, 14, 4)
    
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    
    # 第一行：原图 + 4个叠加热力图
    # 第二行：CLS 分布柱状图 + 4个纯热力图
    
    # 定义卦象名称和颜色风格
    guas = [
        ("Qian (Global)", cv2.COLORMAP_AUTUMN), # 红/黄
        ("Kun (Local)", cv2.COLORMAP_WINTER),   # 蓝/绿
        ("Zhen (High-Freq)", cv2.COLORMAP_HOT), # 黑/红 (强调热点)
        ("Xun (Low-Freq)", cv2.COLORMAP_OCEAN)  # 蓝/白
    ]
    
    # --- 原图 ---
    axes[0, 0].imshow(raw_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    # --- CLS 分布 ---
    axes[1, 0].bar(["Qian", "Kun", "Zhen", "Xun"], cls_weights, color=['red', 'blue', 'orange', 'cyan'])
    axes[1, 0].set_title("CLS Token Choice")
    axes[1, 0].set_ylim(0, 1.0)
    
    # --- 绘制 4 个专家 ---
    for i in range(4):
        gua_name, colormap = guas[i]
        heatmap_data = spatial_weights_2d[:, :, i]
        
        # 1. 叠加图
        overlay = overlay_heatmap(raw_image, heatmap_data, colormap)
        axes[0, i+1].imshow(overlay)
        axes[0, i+1].set_title(f"{gua_name}\nOverlay")
        axes[0, i+1].axis("off")
        
        # 2. 纯热力图
        im = axes[1, i+1].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=1)
        axes[1, i+1].set_title(f"{gua_name}\nProbability")
        axes[1, i+1].axis("off")
        plt.colorbar(im, ax=axes[1, i+1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_name = f"bagua_analysis_{os.path.basename(image_path).split('.')[0]}.png"
    plt.savefig(save_name, dpi=300)
    print(f"[Success] Visualization saved to {save_name}")
    plt.show()

# ==================== 主程序 ====================
def main():
    print("========== Bagua-CLIP Visualization Studio ==========")
    
    # 1. 加载模型
    print(f"[1/3] Loading Model structure...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model = perform_bagua_surgery(model)
    model = model.float()
    
    # 2. 加载权重
    print(f"[2/3] Loading Checkpoint: {CHECKPOINT_PATH}...")
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        # 去除 module. 前缀 (如果之前是DDP训练)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        model = model.to(DEVICE)
    else:
        print("[Error] Checkpoint not found! Please run training first.")
        return

    # 3. 挂载记录器
    recorder = RouterRecorder()
    recorder.attach(model)
    
    # 4. 运行可视化
    print(f"[3/3] Visualizing {TEST_IMAGE_PATH}...")
    if os.path.exists(TEST_IMAGE_PATH):
        visualize_single_image(model, TEST_IMAGE_PATH, recorder)
    else:
        # 如果没有图片，从网上下载一张猫的图片做 Demo
        print("Test image not found. Downloading a demo image...")
        try:
            import requests
            from io import BytesIO
            url = "http://images.cocodataset.org/val2017/000000397133.jpg" # 两只猫
# http://images.cocodataset.org/val2017/000000015335.jpg 三个人 http://images.cocodataset.org/val2017/000000397133.jpg 一个人在厨房 "http://images.cocodataset.org/val2017/000000039769.jpg" # 两只猫
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save("demo_cats.jpg")
            visualize_single_image(model, "demo_cats.jpg", recorder)
        except:
            print("[Error] Failed to download demo image. Please place a .jpg file in this folder.")

if __name__ == "__main__":
    main()
