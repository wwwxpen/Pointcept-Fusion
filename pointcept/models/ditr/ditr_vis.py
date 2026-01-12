import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

try:
    import open3d as o3d
except ImportError:
    o3d = None

class DITRVisualizer:
    def __init__(self, output_dir="vis_debug", active=False, switches=None):
        self.output_dir = output_dir
        self.active = active
        self.global_counter = 0
        self.current_frame_id = 0
        self.current_frame_dir = None
        self.camera_counter = -1
        
        self.switches = {
            "save_raw_pcd": True,
            "save_raw_img": True,
            "save_proj": True,
            "save_dino_map": True,
            "save_dino_pcd": True,
            "save_final_pcd": True
        }
        if switches:
            self.switches.update(switches)

        if self.active:
            os.makedirs(self.output_dir, exist_ok=True)
            # print(f"[DITR Visualizer] Enabled. Saving to {self.output_dir}")

    def process_frame(self, points, labels, image, dino_map, dino_points_feat, proj_u, proj_v, proj_depth, proj_labels):
        if not self.active: return

        # 如果没有frame文件夹，自动创建一个
        if self.current_frame_dir is None:
            self.start_new_frame()
        # 更新camera计数器
        self.camera_counter += 1
        camera_id = self.camera_counter

        frame_id = self.global_counter
        self.global_counter += 1
        prefix = f"frame_{self.current_frame_id:04d}_cam{camera_id:02d}"

        # 1. 保存原始点云 (改为 .pcd)
        if self.switches["save_raw_pcd"]:
            self._save_pcd_label(points, labels, os.path.join(self.current_frame_dir, f"{prefix}_raw_pcd_label.pcd"))

        # 2. 保存原始图片
        if self.switches["save_raw_img"]:
            self._save_image(image, os.path.join(self.current_frame_dir, f"{prefix}_raw_img.jpg"))

        # 3. 保存投影结果
        if self.switches["save_proj"]:
            self._save_projection_depth(image, proj_u, proj_v, proj_depth, 
                                       os.path.join(self.current_frame_dir, f"{prefix}_proj_depth.jpg"))
            self._save_projection_semantic(image, proj_u, proj_v, proj_labels, 
                                          os.path.join(self.current_frame_dir, f"{prefix}_proj_semantic.jpg"))

        # 4. 保存 DINO 特征图
        if self.switches["save_dino_map"]:
            self._save_dino_pca_map(dino_map, os.path.join(self.current_frame_dir, f"{prefix}_dino_map_pca.jpg"))

        # 5. 保存 DINO 赋值后的点云 (改为 .pcd)
        if self.switches["save_dino_pcd"]:
            self._save_pcd_feature_pca(points, dino_points_feat, 
                                      os.path.join(self.current_frame_dir, f"{prefix}_new_pcd_dino.pcd"))

    def start_new_frame(self, frame_id=None):
        """开始处理一个新frame（一个batch的场景）"""
        if not self.active:
            return
        
        if frame_id is None:
            self.current_frame_id = self.global_counter
            self.global_counter += 1
        else:
            self.current_frame_id = frame_id
        
        # 创建frame文件夹
        self.current_frame_dir = os.path.join(self.output_dir, f"frame_{self.current_frame_id:04d}")
        os.makedirs(self.current_frame_dir, exist_ok=True)
        
        # 重置camera计数器
        self.camera_counter = -1
        
        # print(f"[DITR Visualizer] Started frame {self.current_frame_id} at {self.current_frame_dir}")

    def save_final_point_features(self, points, features, batch_idx):
        """保存经过所有camera融合后的最终点云特征"""
        if not self.active or not self.switches["save_final_pcd"]:
            return
        
        if self.current_frame_dir is None:
            return
        
        # 在frame文件夹下保存最终点云
        filename = f"frame_{self.current_frame_id:04d}_batch{batch_idx}_final.pcd"
        filepath = os.path.join(self.current_frame_dir, filename)
        
        self._save_pcd_feature_pca(points, features, filepath)
        # print(f"[DITR Visualizer] Saved final point features for batch {batch_idx} to {filepath}")

    def reset_for_next_frame(self):
        """为下一个frame重置内部状态"""
        self.current_frame_dir = None

    def _save_pcd_label(self, points, labels, filepath):
        if o3d is None: return
        points_np = points.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        np.random.seed(42)
        max_label = labels_np.max() + 1
        colors_map = np.random.uniform(0, 1, size=(max_label, 3))
        
        colors = np.zeros_like(points_np)
        valid_mask = labels_np >= 0
        if valid_mask.any():
            colors[valid_mask] = colors_map[labels_np[valid_mask]]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Open3D 会根据后缀名 .pcd 自动保存为 PCD 格式
        o3d.io.write_point_cloud(filepath, pcd)

    def _save_image(self, img_tensor, filepath):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        plt.imsave(filepath, img)

    def _save_projection(self, img_tensor, u, v, filepath):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        
        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        
        if len(u_np) > 2000:
            idx = np.random.choice(len(u_np), 2000, replace=False)
            u_np = u_np[idx]
            v_np = v_np[idx]
            
        plt.scatter(u_np, v_np, s=1, c='red', alpha=0.5)
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _save_dino_pca_map(self, feat_map, filepath):
        c, h, w = feat_map.shape
        reshaped = feat_map.view(c, -1).T
        self._pca_and_save_img(reshaped, h, w, filepath)

    def _save_pcd_feature_pca(self, points, features, filepath):
        if o3d is None: return
        points_np = points.detach().cpu().numpy()
        feats_np = features.detach()
        
        mask = feats_np.abs().sum(dim=1) > 0
        if mask.sum() < 3: return
        
        valid_feats = feats_np[mask]
        
        try:
            valid_feats = valid_feats.cpu().float()
            _, _, V = torch.pca_lowrank(valid_feats, q=3)
            projected = torch.matmul(valid_feats, V[:, :3])
        except:
            return 

        p_min = projected.min(dim=0)[0]
        p_max = projected.max(dim=0)[0]
        colors_valid = (projected - p_min) / (p_max - p_min + 1e-6)
        colors_valid = colors_valid.cpu().numpy()
        
        colors = np.zeros((len(points_np), 3)) + 0.5
        colors[mask.cpu().numpy()] = colors_valid
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Open3D 会根据后缀名 .pcd 自动保存为 PCD 格式
        o3d.io.write_point_cloud(filepath, pcd)

    def _pca_and_save_img(self, feat_flat, h, w, filepath):
        try:
            feat_flat = feat_flat.cpu().float()
            _, _, V = torch.pca_lowrank(feat_flat, q=3)
            projected = torch.matmul(feat_flat, V[:, :3])
        except:
            return

        p_min = projected.min(dim=0)[0]
        p_max = projected.max(dim=0)[0]
        rgb = (projected - p_min) / (p_max - p_min + 1e-6)
        
        rgb_img = rgb.view(h, w, 3).cpu().numpy()
        plt.imsave(filepath, rgb_img)
    
    # 【按深度画图】
    #  近处（车头）可能是蓝色/青色，远处（背景）是黄色/红色
    def _save_projection_depth(self, img_tensor, u, v, depth, filepath):
        # 准备底图
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)

        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        d_np = depth.detach().cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        # 使用 scatter 的 c 参数传入深度，cmap='jet' (蓝-青-黄-红)
        # s=1~3 调节大小
        plt.scatter(u_np, v_np, c=d_np, cmap='jet', s=1.5, alpha=0.7)
        plt.colorbar(label='Depth (m)')
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

    # 【按语义画图】
    def _save_projection_semantic(self, img_tensor, u, v, labels, filepath):
        # 准备底图
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)

        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        l_np = labels.detach().cpu().numpy().astype(int)

        # 生成颜色 (保持和 PCD 一致的随机种子)
        np.random.seed(42)
        max_label = l_np.max() + 1
        # 防止 max_label 太小导致 index error
        max_label = max(max_label, 20) 
        colors_map = np.random.uniform(0, 1, size=(max_label, 3))
        
        # 映射颜色
        point_colors = colors_map[l_np]

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.scatter(u_np, v_np, c=point_colors, s=1.5, alpha=0.8)
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()