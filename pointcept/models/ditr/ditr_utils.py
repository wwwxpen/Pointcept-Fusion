import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import timm # 【修改】使用 timm 替代 torch.hub

try:
    import open3d as o3d
except ImportError:
    o3d = None

class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitl14', frozen=True):
        super().__init__()
        
        # 【修改】将 DINOv2 官方名称映射到 timm 名称
        # timm 的命名规则略有不同，.lvd142m 后缀代表官方权重
        name_map = {
            'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
            'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
            'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
            'dinov2_vitg14': 'vit_giant_patch14_dinov2.lvd142m',
        }
        timm_name = name_map.get(model_name, model_name)

        # 【请修改这里】设置为你的本地权重路径
        local_weight_path = "pointcept/models/ditr/dinov2_vitl14_pretrain.bin"
        
        print(f"[DITR] Loading DINOv2 from timm: {timm_name}")
        
        # 检查文件是否存在，避免路径填错
        if not os.path.exists(local_weight_path):
            # 如果本地没找到，尝试用之前的逻辑（这里是为了容错，你也可以直接报错）
            print(f"Warning: Local weight not found at {local_weight_path}, trying to download...")
            pretrained_flag = True
            ckpt_path = ""
        else:
            pretrained_flag = False
            ckpt_path = local_weight_path

        # dynamic_img_size=True 允许处理不同分辨率的图片输入
        self.dino = timm.create_model(
            timm_name, 
            pretrained=pretrained_flag,     # 关闭自动下载
            checkpoint_path=ckpt_path,      # 指定本地路径
            num_classes=0, 
            dynamic_img_size=True 
        )
        
        if frozen:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
            
    def forward(self, images):
        # 兼容处理 5D 和 4D 输入
        if images.dim() == 5:
            # images: [B, N_views, 3, H, W] -> Flatten -> [B*N, 3, H, W]
            b, n, c, h, w = images.shape
            x = images.view(b * n, c, h, w)
        elif images.dim() == 4:
            # [Batch*N_views, C, H, W] -> 维度被合并的情况
            # nuScenes 固定有 6 个相机
            total_n, c, h, w = images.shape
            n = 6 
            # 确保 total_n 能被 6 整除，否则说明数据有问题
            if total_n % n != 0:
                # 容错：如果不是 6 的倍数，可能 batch_size=1 且只有 1 个 view? 
                # 但根据 Dataset 代码我们总是 stack 6 张。
                # 假如出错，这里会抛出异常
                raise ValueError(f"Images shape {images.shape} implies flattened batch, but {total_n} is not divisible by 6 views.")
            
            b = total_n // n
            x = images # 已经是 [B*N, C, H, W] 了，不需要 view
        else:
            raise ValueError(f"Unexpected images shape: {images.shape}. Expected 4 or 5 dimensions.")
        
        with torch.no_grad():
            # 【修改】timm 的 forward_features 返回 [Batch, N_tokens, Dim]
            # 包含了 CLS token 和 Patch tokens
            out = self.dino.forward_features(x)
            
            # 这里的 out 包含了 CLS token (index 0)
            # 我们只需要 patch tokens (index 1:)
            # 另外 DINOv2 有些变体可能有 register tokens，但标准版通常只有 CLS
            # 标准 ViT: out[:, 1:, :]
            patch_features = out[:, 1:, :]
            
        dim = patch_features.shape[-1]
        # Vit Patch Size = 14
        p_h, p_w = h // 14, w // 14
        
        # Reshape 回空间维度 
        # [B*N, H*W, Dim] -> [B, N, H, W, Dim] -> [B, N, Dim, H, W]
        feature_map = patch_features.view(b, n, p_h, p_w, dim)
        feature_map = feature_map.permute(0, 1, 4, 2, 3) 
        return feature_map

class DITRInjector(nn.Module):
    def __init__(self, dino_model, debug=False, output_dir="debug_ditr"):
        super().__init__()
        self.dino = dino_model
        self.debug = debug
        self.output_dir = output_dir
        if debug:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[DITR] Debug mode enabled. Visualization will be saved to {output_dir}")

    def project_and_sample(self, points, batch_idx, imgs, intrinsics, extrinsics):
        # 1. 提取图片特征 (调用上面的 timm dino)
        dino_features = self.dino(imgs)
        B, N_views, Dim, PH, PW = dino_features.shape
        H, W = imgs.shape[-2], imgs.shape[-1]
        
        point_features = torch.zeros((points.shape[0], Dim), device=points.device, dtype=points.dtype)

        # 逐个 Batch 处理
        for b in range(B):
            b_mask = (batch_idx == b)
            if not b_mask.any(): continue
            
            b_points = points[b_mask]
            b_points_homo = torch.cat([b_points, torch.ones_like(b_points[:, :1])], dim=1)
            
            for v in range(N_views):
                # 投影 Lidar -> Camera
                pc_cam = (extrinsics[b, v] @ b_points_homo.T).T
                depth = pc_cam[:, 2]
                # 1. 深度检查 (Z-Check): 检查点是否在相机前方
                # 相机坐标系中，Z轴通常朝前。Z < 0 表示点在相机背后，我们设置 > 0.1 是为了防止除以接近0的数导致数值不稳定。
                valid_z = depth > 0.1
                
                # 投影 Camera -> Pixel
                pc_img = (intrinsics[b, v] @ pc_cam[:, :3].T).T
                u = pc_img[:, 0] / pc_img[:, 2]
                v_coord = pc_img[:, 1] / pc_img[:, 2]
                
                # 2. 图像边界检查 (FOV Check): 检查点是否落在成像平面内
                # u, v 是像素坐标。如果 u < 0 或 u >= W，说明点在相机水平视场角之外，同理 v < 0 或 v >= H 说明在垂直视场角之外。
                valid_u = (u >= 0) & (u < W)
                valid_v = (v_coord >= 0) & (v_coord < H)
                valid = valid_z & valid_u & valid_v
                
                if not valid.any(): continue
                
                # 计算 Patch 索引
                u_patch = (u[valid] / 14).long().clamp(0, PW - 1)
                v_patch = (v_coord[valid] / 14).long().clamp(0, PH - 1)
                
                # 采样
                feats_map = dino_features[b, v] 
                sampled_feats = feats_map[:, v_patch, u_patch].T 
                
                global_indices = torch.where(b_mask)[0][valid]
                point_features[global_indices] = sampled_feats
                
                # 可视化 (仅第一个 Batch 的第一个 View)
                if self.debug and b == 0 and v == 0:
                     self.viz_projection(imgs[b,v], u[valid], v_coord[valid], f"proj_batch{b}_view{v}.png")
                     self.viz_dino_pca(feats_map, f"dino_pca_batch{b}_view{v}.png")

        if self.debug:
            self.viz_pcd_features(points, point_features, "pcd_dino_features.ply")

        return point_features

    def viz_projection(self, img_tensor, u, v, fname):
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3,1,1)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        # 随机画一些点，防止过密
        if len(u) > 500:
            idx = torch.randperm(len(u))[:500]
            u, v = u[idx], v[idx]
        plt.scatter(u.cpu().numpy(), v.cpu().numpy(), s=2, c='red', alpha=0.8)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, fname), bbox_inches='tight', pad_inches=0)
        plt.close()

    def viz_dino_pca(self, feature_map, fname):
        c, h, w = feature_map.shape
        reshaped = feature_map.view(c, -1).T.float().cpu().numpy()
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca_f = pca.fit_transform(reshaped)
        pca_f = (pca_f - pca_f.min(0)) / (pca_f.max(0) - pca_f.min(0) + 1e-6)
        pca_img = pca_f.reshape(h, w, 3)
        plt.imsave(os.path.join(self.output_dir, fname), pca_img)

    def viz_pcd_features(self, points, features, fname):
        if o3d is None or features.sum() == 0: return
        
        if len(points) > 200000:
            mask = torch.randperm(len(points))[:200000]
            points = points[mask]
            features = features[mask]
            
        points_np = points.cpu().numpy()
        feats_np = features.float().cpu().numpy()
        
        from sklearn.decomposition import PCA
        valid_mask = np.abs(feats_np).sum(1) > 0
        if valid_mask.sum() < 10: return
        
        colors = np.zeros((len(points_np), 3))
        colors[~valid_mask] = [0.5, 0.5, 0.5] 
        
        pca = PCA(n_components=3)
        pca_c = pca.fit_transform(feats_np[valid_mask])
        pca_c = (pca_c - pca_c.min(0)) / (pca_c.max(0) - pca_c.min(0) + 1e-6)
        colors[valid_mask] = pca_c
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, fname), pcd)