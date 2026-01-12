import torch
import torch.nn as nn
from functools import partial
import random

from pointcept.models.builder import MODELS
# 导入 PTv3
from ..point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point
from .ditr_utils import DINOFeatureExtractor, DITRInjector
from .ditr_vis import DITRVisualizer

try:
    from torch_scatter import scatter_max
except ImportError:
    pass

@MODELS.register_module("PDITR-PTv3")
class PDITR_PTv3(PointTransformerV3):
    def __init__(self, 
                 use_visual_modality=True, 
                 dino_backbone_name="dinov2_vitl14", 
                 dino_dim=1024,
                 img_size=(378, 672),
                 vis_switches=None,         # 从配置接收可视化开关
                 vis_active=True,           # 是否激活可视化
                 vis_output_dir="vis_ditr_output",  # 可选输出路径
                 **kwargs):
        super().__init__(**kwargs)
        self.use_visual_modality = use_visual_modality
        self.img_size = img_size
        
        if self.use_visual_modality:
            # 注意：这里我们使用的是 ditr_utils 里修改过(支持本地权重)的 DINOFeatureExtractor
            print(f"[PDITR] Initializing DINOv2: {dino_backbone_name}")
            self.dino_extractor = DINOFeatureExtractor(dino_backbone_name)
            self.injector = DITRInjector(self.dino_extractor, debug=False)
            
            # 如果配置里没有提供 vis_switches，就使用默认值
            default_vis_switches = {
                "save_raw_pcd": False, 
                "save_raw_img": False,
                "save_proj": False,
                "save_dino_map": False,
                "save_dino_pcd": False,
                "save_final_pcd": False
            }
            vis_cfg = vis_switches if vis_switches is not None else default_vis_switches
            # active=True 开启保存，active=False 关闭所有保存
            self.vis = DITRVisualizer(output_dir=vis_output_dir, active=vis_active, switches=vis_cfg)

            self.dino_projections = nn.ModuleList()
            
            # 动态获取每一层 Decoder 需要的维度
            for i in range(len(self.dec)):
                # self.dec[i] 是一个 Stage
                # self.dec[i][0] 是 Unpooling 模块
                up_layer = self.dec[i][0] 
                
                # Unpooling 里的 proj 是一个 PointSequential，里面包含 Linear, Norm, Act 等
                # 我们通过遍历 modules() 找到第一个 nn.Linear 层
                out_dim = None
                for m in up_layer.proj.modules():
                    if isinstance(m, nn.Linear):
                        out_dim = m.out_features
                        break
                
                if out_dim is None:
                    # 如果万一没找到，打印错误信息帮助调试
                    raise AttributeError(f"Could not find nn.Linear layer in decoder stage {i} projection: {up_layer.proj}")
                
                self.dino_projections.append(
                    nn.Sequential(
                        nn.Linear(dino_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.GELU()
                    )
                )

    def forward(self, data_dict):
        #  可视化
        if self.use_visual_modality and hasattr(self, 'vis') and self.vis is not None:
            # 使用batch_index或scene_id作为frame_id
            frame_id = None
            if "batch_index" in data_dict:
                frame_id = data_dict["batch_index"][0].item()
            elif "scene_id" in data_dict:
                frame_id = data_dict["scene_id"][0].item()
            self.vis.start_new_frame(frame_id=frame_id)

        # ... (forward 函数保持之前的 Max Pooling 版本不变) ...
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        
        # 1. 初始 DINO 特征提取
        dino_feat_current = None
        dino_feat_pyramid = []
        
        if self.use_visual_modality and "imgs" in data_dict:
            imgs = data_dict["imgs"]
            img_feats_maps = self.injector.dino(imgs)

            # 【修改】传入 segment (labels) 给 sample_from_maps
            # 注意：Point 类封装后，point.segment 或者 data_dict['segment'] 都可以
            # 但这里 point 已经被 sparsify 了，为了画原始点云，最好用 data_dict['segment'] 
            # 不过 data_dict['coord'] 和 point.coord 在 sparsify 后可能不一样
            # 为了对齐，我们使用 point.segment (如果存在)
            labels = point.segment if "segment" in point.keys() else None

            # 【直接使用 point.color 进行投影】
            # 因为它就是我们需要的 "Downsample 后的原始坐标"
            # 如果是测试集(可能没color)，回退到 coord
            proj_coord = point.color if "color" in point.keys() else point.coord
            
            dino_feat_current = self.sample_from_maps(
                proj_coord, 
                offset2batch(point.offset), 
                img_feats_maps, 
                data_dict["intrinsics"], 
                data_dict["extrinsics"],
                imgs,   # 【新增】传入原始图片用于可视化
                labels  # 【新增】传入 Label 用于可视化
            )
            dino_feat_pyramid.append(dino_feat_current)

        # 2. Encoder (同步 Max Pooling)
        # 必须手动遍历 Encoder 以插入 Pooling 逻辑
        # 注意：这里需要根据 v3m1_base 的结构，不要调用 self.enc(point)
        # 而是遍历 self.enc
        
        # Point Embedding
        point = self.embedding(point)
        
        # 遍历 Encoder Stages
        for s, enc_stage in enumerate(self.enc):
            if s > 0:
                # 获取 Pooling 层
                down_layer = enc_stage[0] 
                point = down_layer(point)
                
                # DINO Max Pooling
                if dino_feat_current is not None:
                    if hasattr(point, "pooling_inverse"):
                        cluster_idx = point.pooling_inverse
                        # 确保引入了 scatter_max
                        from torch_scatter import scatter_max
                        dino_feat_next, _ = scatter_max(
                            dino_feat_current, 
                            cluster_idx, 
                            dim=0
                        )
                        dino_feat_current = dino_feat_next
                        dino_feat_pyramid.append(dino_feat_current)
                
                # 运行该 Stage 剩余的 Block
                for i, block in enumerate(enc_stage):
                    if i == 0: continue # 跳过第一个 (down_layer)
                    point = block(point)
            else:
                # s=0 没有 downsample
                point = enc_stage(point)
                # dino_feat_pyramid[0] 已在最开始添加

        # 3. Decoder
        if not self.enc_mode:
            for i, dec_stage in enumerate(self.dec):
                unpool_layer = dec_stage[0]
                
                if self.use_visual_modality and len(dino_feat_pyramid) > 0:
                    # 计算对应的 Pyramid 层级
                    # i=0 (deepest) -> target=L3 (if total 5 levels)
                    # 倒数第 (i+2) 个特征
                    target_idx = -(i + 2)
                    
                    if abs(target_idx) <= len(dino_feat_pyramid):
                        dino_feat_target = dino_feat_pyramid[target_idx]
                        
                        # 投影
                        dino_feat_proj = self.dino_projections[i](dino_feat_target)
                        
                        # 手动 Unpooling
                        parent = point.pop("pooling_parent")
                        inverse = point.pooling_inverse
                        
                        # 特征融合: Up + Skip + DINO
                        parent = unpool_layer.proj_skip(parent) # Skip
                        parent.feat = parent.feat + dino_feat_proj # Inject DINO
                        
                        point_feat_up = unpool_layer.proj(point).feat # Up
                        parent.feat = parent.feat + point_feat_up[inverse]
                        
                        if unpool_layer.traceable:
                            parent["unpooling_parent"] = point
                        
                        point = parent
                    else:
                        # 容错
                        point = unpool_layer(point)
                else:
                    point = unpool_layer(point)
                
                for j, block in enumerate(dec_stage):
                    if j == 0: continue # 跳过第一个 (unpool_layer)
                    point = block(point)

        # 可视化
        if self.use_visual_modality and hasattr(self, 'vis') and self.vis is not None:
            self.vis.reset_for_next_frame()

        return point

    def sample_from_maps(self, points, batch, feature_maps, intrinsics, extrinsics, raw_imgs=None, raw_labels=None):
        B, V, Dim, PH, PW = feature_maps.shape
        H_img, W_img = self.img_size 
        point_features = torch.zeros((points.shape[0], Dim), device=points.device, dtype=points.dtype)
        
        # 检查并修正 intrinsics 和 extrinsics 的维度
        # 如果它们被 Flatten 成了 [B*V, 3/4, 3/4] (3维)，需要 Reshape 回 [B, V, ...]
        if intrinsics.dim() == 3: intrinsics = intrinsics.view(B, V, 3, 3)
        if extrinsics.dim() == 3: extrinsics = extrinsics.view(B, V, 4, 4)

        for b in range(B):
            b_mask = (batch == b)
            if not b_mask.any(): continue
            b_points = points[b_mask]

            # 准备 Label (如果提供了)
            b_labels = raw_labels[b_mask] if raw_labels is not None else torch.zeros(len(b_points))
            
            # 强制转换为 2D [N, 3]
            if b_points.dim() == 1:
                b_points = b_points.unsqueeze(0)
            
            b_points_f = b_points.float()
            b_points_homo = torch.cat([b_points_f, torch.ones_like(b_points_f[:, :1])], dim=1)
            
            # 保存所有点的候选特征, 最后随机选择, key: 全局索引, value: 特征列表
            candidates_dict = {}
            for v in range(V):
                # 此时 extrinsics[b, v] 保证是 [4, 4] 矩阵
                pc_cam = (extrinsics[b, v] @ b_points_homo.transpose(0, 1)).transpose(0, 1)
                
                depth = pc_cam[:, 2]
                mask_z = depth > 0.1
                
                pc_img = (intrinsics[b, v] @ pc_cam[:, :3].transpose(0, 1)).transpose(0, 1)
                u = pc_img[:, 0] / pc_img[:, 2]
                v_coord = pc_img[:, 1] / pc_img[:, 2]
                
                mask_u = (u >= 0) & (u < W_img)
                mask_v = (v_coord >= 0) & (v_coord < H_img)
                valid = mask_z & mask_u & mask_v

                # 为了可视化，我们需要先拿到当前 View 的所有特征分配情况
                # 创建一个临时的 feat 容器给可视化用
                current_view_point_feats = torch.zeros((len(b_points), Dim), device=points.device)
                
                if not valid.any(): continue
                
                u_p = (u[valid] / 14).long().clamp(0, PW - 1)
                v_p = (v_coord[valid] / 14).long().clamp(0, PH - 1)
                
                feats = feature_maps[b, v, :, v_p, u_p].transpose(0, 1)
                
                idx_global = torch.where(b_mask)[0][valid]
                # point_features[idx_global] = feats   # 这里改用随机
                # 收集点的特征信息到字典
                for i, global_idx in enumerate(idx_global):
                    idx_item = global_idx.item()
                    if idx_item not in candidates_dict:
                        candidates_dict[idx_item] = []
                    candidates_dict[idx_item].append(feats[i])
                
                # 赋值给临时 (用于可视化)
                current_view_point_feats[valid] = feats
                # 仅保存第一个 View (v=0) 以节省空间，或者全部保存
                if raw_imgs is not None:
                    if raw_imgs.dim() == 4:  # [B*N_views, C, H, W]
                        # 计算每个batch的起始索引
                        if raw_imgs.shape[0] == B * V:  # 期望的格式
                            image_idx = b * V + v
                            if image_idx < raw_imgs.shape[0]:
                                image = raw_imgs[image_idx]
                            else:
                                print(f"[WARNING] Image index {image_idx} out of bounds")
                                continue
                        else:
                            # 使用简单的回退：只用第一个视图
                            if v == 0:
                                image = raw_imgs[b * (raw_imgs.shape[0] // B)]
                            else:
                                continue  # 只可视化第一个视图
                    elif raw_imgs.dim() == 5:  # [B, N_views, C, H, W]
                        if v < raw_imgs.shape[1]:
                            image = raw_imgs[b, v]
                        else:
                            continue
                    else:
                        print(f"[WARNING] Unexpected image dim: {raw_imgs.dim()}")
                        continue


                    self.vis.process_frame(
                        points = b_points,
                        labels = b_labels,
                        image = image,
                        dino_map = feature_maps[b, v],
                        dino_points_feat = current_view_point_feats,
                        proj_u = u[valid],          # 投影点 U
                        proj_v = v_coord[valid],    # 投影点 V
                        proj_depth = depth[valid],  # 传入可视点的深度
                        proj_labels = b_labels[valid] # 传入可视点的语义标签
                    )
            # 点的图像特征随机选择
            for global_idx, feat_list in candidates_dict.items():
                if feat_list:
                    point_features[global_idx] = random.choice(feat_list)

            # 为当前batch绘制最终点云特征 ============
            if hasattr(self, 'vis') and self.vis is not None:
                # 提取当前batch的点云和特征
                b_points = points[b_mask]
                b_features = point_features[b_mask]
                
                # 只保存有特征的点
                mask = b_features.abs().sum(dim=1) > 0
                if mask.any():
                    valid_points = b_points[mask]
                    valid_features = b_features[mask]
                    
                    self.vis.save_final_point_features(
                        points=valid_points,
                        features=valid_features,
                        batch_idx=b
                    )

        return point_features
