import torch
import json
import os
from collections import defaultdict
import numpy as np


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

        self.z_history = defaultdict(list)  # 每个高斯ID维护一段Z历史
        self.z_window_size = 10
    def initialize(self):
        """初始化日志字典,基于当前模型ID"""
        for idx in self.model._ids.cpu().tolist():
            self.logs[idx] = []

    def update(self):
        """
        每次调用，记录当前所有高斯球参数和状态
        """
        self.iteration += 1
        current_ids = set(self.model._ids.cpu().tolist())
        positions = self.model.get_xyz.detach().cpu()
        scales = self.model.get_scaling.detach().cpu()
        alphas = self.model.get_opacity.detach().cpu()
        hit_counts = self.model._hit_counts.cpu()

        # === 维护全局 z 值分布 ===
        ids = self.model._ids.cpu().tolist()
        zs = positions[:, 2].cpu().numpy()

        # 为每个ID记录Z值历史
        for gid, z in zip(ids, zs):
            self.z_history[gid].append(z)
            if len(self.z_history[gid]) > self.z_window_size:
                self.z_history[gid].pop(0)
        # =========================

        # 动态发现新增ID，初始化日志
        new_ids = current_ids - self.active_ids
        for new_id in new_ids:
            self.logs[new_id] = []

            #初始化时刻状态日志
            idx = (self.model._ids == new_id).nonzero(as_tuple=True)[0].item()
            record = self._make_log_record(new_id, idx, positions, scales, alphas, hit_counts, initial=True)
            self.logs[new_id].append(record)

        # 判断剔除ID（之前存在但不在模型里）
        removed_ids = self.active_ids - current_ids

        for rid in removed_ids:
            #标记为已剔除，保存一条状态
            self.logs[rid].append({
                "iteration": self.iteration,
                "status": "removed"
            })
        
        # 更新当前活跃ID集
        self.active_ids = current_ids
        '''计算状态相关'''
        # 计算是否在表面示例（可根据实际需求调整）
        on_surface = (alphas.squeeze() > 0.05)  # 简单阈值举例
        # 判断膨胀、突然上升、低命中等逻辑，可以用自定义函数实现
        is_expanded = self._check_expanded(scales)
        sudden_rise = self._check_sudden_rise(scales)
        low_hit = self._check_low_hits(hit_counts,self.iteration)
        # 计算剔除原因示例
        # prune_reasons = self._get_prune_reasons(alphas, scales)

        # 遍历所有当前活跃ID，记录状态
        for i, gid in enumerate(self.model._ids.cpu().tolist()):
            record = {
                "iteration": self.iteration,
                "pos": positions[i].tolist(),
                "scale": scales[i].tolist(),
                "alpha": float(alphas[i]),
                # "hit_count": int(hit_counts[i]),
                # "on_surface": bool(on_surface[i]),
                # "is_expanded": bool(is_expanded[i]),
                # "sudden_rise": bool(sudden_rise[i]),
                # "long_term_low_hit": bool(low_hit[i]),
                "prune_reason": self._get_prune_reason(gid, alphas[i], scales[i], z, hit_counts[i], self.iteration)
            }
            self.logs[gid].append(record)
        # print(f"[Tracker] Iter {self.iteration} - Num Gaussians: {len(self.model._ids)}")
        # print(f"Positions shape: {positions.shape}")
        # print(f"Scales shape: {scales.shape}")
        # print(f"Alphas shape: {alphas.shape}")
        # print(f"Hit counts shape: {hit_counts.shape}")

        # 定时保存日志
        if self.iteration % self.log_interval == 0:
            self.suspicious = defaultdict(list)
            for gid in self.active_ids:
                last_log = self.logs[gid][-1]
                
                # 根据是否有可疑的原因
                if last_log.get("prune_reason"):
                    self.suspicious[gid].append(last_log)
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
    def _check_expanded(self, scales):
        # 示例：如果scale任一维超过阈值，则认为膨胀
        threshold = 1.5  # 举例阈值
        return (scales > threshold).any(dim=1)

    def _check_sudden_rise(self, scales):
        # 需要保存历史scale比较，简化起见这里返回全False
        return torch.zeros(scales.size(0), dtype=torch.bool)

    def _check_low_hits(self, hit_counts: torch.Tensor, iteration: int):
        # 返回一个布尔Tensor，表示哪些高斯命中数低
        return (hit_counts < 5) & (iteration > 100)
    def _get_prune_reason(self, gid, alpha, scale, z, hit_count, iteration):
        reasons = []
        if alpha < 0.01:
            reasons.append("transparency")
        if max(scale) > 0.1:  # 按需调整
            reasons.append("overscaled")
        z_values = self.z_history.get(gid, [])
        if len(z_values) >= 3:
            z_mean = np.mean(z_values)
            z_std = np.std(z_values) + 1e-6
            if z > z_mean + 3 * z_std:
                return "sudden_rise"
        if hit_count < 5 and iteration > 100:
            reasons.append("low_visibility")
        
        return reasons

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
