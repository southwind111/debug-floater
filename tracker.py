import torch
import json
import os
from collections import defaultdict
import numpy as np
import open3d as o3d


class GaussianTracker:
    def __init__(self, model, out_dir="logs", log_interval=100):
        """
        model: GaussianModel实例,负责提供_ids, _hit_counts等信息
        """
        self.model = model
        self.log_interval = log_interval
        self.logs = defaultdict(list)
        self.out_dir = out_dir
        self.suspicious = defaultdict(list)
        os.makedirs(self.out_dir, exist_ok=True)
        self.iteration = 0
        self.active_ids = set()
        self.prune_weights = {"color_jitter":2.0,
                              "fast_motion":1.0,
                              "elongate":1.0,
                              "isolated":1.0,
                              "fake_floater":1.0,
                              "transparency":1.0,
                              "overscaled":1.0,
                              "sudden_rise":1.0,
                              "low_visibility":1.0}


        '''位置，颜色相关'''
        self.prev_rgb_dict = defaultdict(list)
        self.prev_pos_dict = defaultdict(list)
        self.z_history = defaultdict(list)  # 每个高斯ID维护一段Z历史
        self.z_window_size = 10
    def initialize(self):
        """初始化日志字典,基于当前模型ID"""
        for idx in self.model._ids.cpu().tolist():
            self.logs[idx] = []

    def update(self):
        """
        优化版 update 函数：
        - 每10轮记录一次完整日志，其余只更新状态（节省内存）
        - 使用 torch.no_grad() 避免显存膨胀
        """
        import math
        import numpy as np
        import torch
        from collections import defaultdict
        from scipy.spatial import cKDTree
        from utils.sh_utils import sh_to_rgb

        self.iteration += 1
        full_log = (self.iteration % 10 == 0)

        with torch.no_grad():
            ids_tensor = self.model._ids.detach()
            ids = ids_tensor.cpu().tolist()
            current_ids = set(ids)

            positions = self.model.get_xyz.detach().cpu()
            scales = self.model.get_scaling.detach().cpu()
            alphas = self.model.get_opacity.detach().cpu()
            hit_counts = self.model._hit_counts.detach().cpu()

            #计算相关颜色
            features_dc = self.model._features_dc.detach().transpose(1, 2)
            features_rest = self.model._features_rest.detach().transpose(1, 2)
            features_all = torch.cat([features_dc, features_rest], dim=2)

            dirs = torch.tensor([0.0, 0.0, 1.0], device=features_all.device).repeat(features_all.shape[0], 1)
            coeff_num = features_all.shape[-1]
            deg = int(math.sqrt(coeff_num)) - 1
            rgbs = sh_to_rgb(features_all, dirs, deg).detach().cpu()

        # 维护Z历史
        zs = positions[:, 2].numpy()
        for gid, z in zip(ids, zs):
            self.z_history[gid].append(z)
            if len(self.z_history[gid]) > self.z_window_size:
                self.z_history[gid].pop(0)

        # 发现新 ID
        new_ids = current_ids - self.active_ids
        for new_id in new_ids:
            self.logs[new_id] = []
            idx = (ids_tensor == new_id).nonzero(as_tuple=True)[0].item()
            record = self._make_log_record(new_id, idx, positions, scales, alphas, hit_counts, initial=True)
            self.logs[new_id].append(record)

        removed_ids = self.active_ids - current_ids
        for rid in removed_ids:
            self.logs[rid].append({"iteration": self.iteration, "status": "removed"})

        self.active_ids = current_ids

        gid_to_color_delta = {}
        gid_to_pos_delta = {}
        shape_ratios = {}
        neighbor_dists = {}
        pos_dict = {}

        for i, gid in enumerate(ids):
            rgb = rgbs[i]
            pos = positions[i]
            alpha = float(alphas[i])
            scale = scales[i]

            prev_rgb = self.prev_rgb_dict.get(gid, rgb)
            prev_pos = self.prev_pos_dict.get(gid, pos)

            color_delta = torch.norm(rgb - prev_rgb).item()
            pos_delta = torch.norm(pos - prev_pos).item()

            self.prev_rgb_dict[gid] = rgb
            self.prev_pos_dict[gid] = pos

            if not full_log:
                continue  # 非记录周期只做状态更新

            gid_to_color_delta[gid] = color_delta
            gid_to_pos_delta[gid] = pos_delta
            pos_dict[gid] = pos.tolist()

            volume = float(torch.prod(scale).item() + 1e-6)
            density = alpha / volume
            strength = alpha * density

            try:
                # sigma = self.model._scaling[i].detach().cpu().numpy()
                sigma = scale.tolist()
                vals = np.abs(sigma)
                shape_ratio = float(np.max(vals) / max(np.min(vals), 1e-6))
            except Exception as e:
                print(f"[Warning] shape_ratio fallback due to: {e}, scale={scale}")
                shape_ratio = 1.0
            shape_ratios[gid] = shape_ratio
            neighbor_dists[gid] = None

            record = {
                "iteration": self.iteration,
                "pos": pos.tolist(),
                "scale": scale.tolist(),
                "alpha": alpha,
                "color_delta": color_delta,
                "pos_delta": pos_delta,
                "strength": strength,
                "shape_ratio": shape_ratio,
                "prune_reason": self._get_prune_reason(gid, alpha, scale, zs[i], hit_counts[i], strength, self.iteration)
            }
            self.logs[gid].append(record)
            if len(self.logs[gid]) > 20:
                self.logs[gid].pop(0)

        # 点间距离计算
        if full_log and len(pos_dict) > 1:
            positions_arr = np.array(list(pos_dict.values()))
            gids = list(pos_dict.keys())
            tree = cKDTree(positions_arr)
            distances, _ = tree.query(positions_arr, k=min(9, len(positions_arr)))
            for idx, gid in enumerate(gids):
                neighbor_dists[gid] = float(np.mean(distances[idx][1:]))

        # 记录可疑点
        if full_log:
            self.suspicious = defaultdict(list)
            num_points = len(gid_to_color_delta)
            topk = max(1, int(0.01 * num_points))
            topk_color_ids = sorted(gid_to_color_delta, key=gid_to_color_delta.get, reverse=True)[:topk]
            topk_pos_ids = sorted(gid_to_pos_delta, key=gid_to_pos_delta.get, reverse=True)[:topk]
            topk_shape_ids = sorted(shape_ratios, key=shape_ratios.get, reverse=True)[:topk]
            topk_neighbor_ids = sorted(neighbor_dists, key=neighbor_dists.get, reverse=True)[:topk]
            for gid in self.active_ids:
                last_log = self.logs[gid][-1]
                if "prune_reason" not in last_log or not isinstance(last_log["prune_reason"], list):
                    last_log["prune_reason"] = []

                if gid in topk_color_ids:
                    last_log["prune_reason"].append("color_jitter")
                if gid in topk_pos_ids:
                    last_log["prune_reason"].append("fast_motion")
                if gid in topk_shape_ids:
                    last_log["prune_reason"].append("elongated")
                if gid in topk_neighbor_ids:
                    last_log["prune_reason"].append("isolated")

                #计算可疑得分
                reasons = last_log["prune_reason"]
                if reasons:
                    weights = [self.prune_weights.get(r, 1.0) for r in reasons]
                    suspicious_score = sum(weights)
                    last_log["suspicious_score"] = suspicious_score
                    if suspicious_score >= 1.0:
                        self.suspicious[gid].append(last_log)
                        if len(self.suspicious[gid]) > 3:
                            self.suspicious[gid].pop(0)

            self.save_logs()
    def _make_log_record(self, gid, idx, positions, scales, alphas, hit_counts, initial=False):
        return {
            "iteration": self.iteration,
            "pos": positions[idx].tolist(),
            "scale": scales[idx].tolist(),
            "alpha": float(alphas[idx]),
            "hit_count": int(hit_counts[idx]),
            "status": "initial" if initial else "active"
         }
    
    def _get_prune_reason(self, gid, alpha, scale, z, hit_count,strength, iteration):
        reasons = []
        if strength > 150 and hit_count < 3:
            reasons.append("fake_floater")
        if alpha < 0.01:
            reasons.append("transparency")
        if max(scale) > 3:  # 按需调整
            reasons.append("overscaled")
        z_values = self.z_history.get(gid, [])
        if len(z_values) >= 3:
            z_mean = np.mean(z_values)
            z_std = np.std(z_values) + 1e-6
            if z > z_mean + 3 * z_std:
                reasons.append("sudden_rise")
        recent_hit_counts = [log.get("hit_count", 0) for log in self.logs.get(gid, [])[-10:]]
        low_hits = sum(hc < 5 for hc in recent_hit_counts)
        if len(recent_hit_counts) == 10 and low_hits >= 8:
            reasons.append("low_visibility")
        
        return reasons
    def visualize_suspicious(self):
        """
        使用 Open3D 可视化当前所有 suspicious 高斯球。
        仅展示带有可疑记录的点，颜色根据异常类型简单区分。
        """
        if not self.suspicious:
            print("当前无可疑高斯球。")
            return

        vis_points = []
        vis_colors = []

        

        for gid, logs in self.suspicious.items():
            # 取最新一条记录
            record = logs[-1]
            pos = record.get("pos", None)
            if pos is None:
                continue

            # 默认颜色hong色
            color = [1.0, 0.0, 0.0]

            vis_points.append(pos)
            vis_colors.append(color)

        if len(vis_points) == 0:
            print("当前无可疑高斯球有效位置可视化。")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_points)
        pcd.colors = o3d.utility.Vector3dVector(vis_colors)

        # 打开窗口显示
        o3d.visualization.draw_geometries([pcd], window_name="Suspicious Gaussians Visualization")

    def save_logs(self):
        def _to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, (bool, bool)):
                return bool(obj)
            elif isinstance(obj, (int, float, str, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {str(k): _to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            else:
                return str(obj)  # fallback for unexpected types

        serializable_logs = {
            "logs": _to_serializable(self.logs),
            "suspicious": _to_serializable(self.suspicious)
        }
        path = os.path.join(self.out_dir, f"gaussian_logs_iter_{self.iteration}.json")
        with open(path, "w") as f:
            json.dump(serializable_logs, f, indent=2)

    def load_logs(self, path):
        with open(path, "r") as f:
            self.logs = json.load(f)
