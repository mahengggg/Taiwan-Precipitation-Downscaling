# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 01:56:53 2025

@author: user
"""

# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import glob
import re
from datetime import datetime

class RainOnlyUpscaleDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths=None, grap_paths=None, qpe_dir=None,
                 in_shape=(16, 16), out_shape=(320, 320),
                 normalize_rawdata_mean=torch.Tensor([0.1]),
                 normalize_rawdata_std=torch.Tensor([1.0]),
                 normalize_residual_mean=torch.Tensor([0.0]),
                 normalize_residual_std=torch.Tensor([1.0]),
                 is_netcdf=False):

        # 1. 優先使用傳入的路徑列表 (Prioritize input path list)
        self.actual_input_paths = data_paths if data_paths is not None else grap_paths
        self.qpe_dir = qpe_dir if qpe_dir is not None else "/8THDD6/Zrnie/QPE_data/12hr_00125deg_GT"

        if self.actual_input_paths is None:
            raise ValueError("Error: No input paths provided!")

        self.lowres_list, self.coarse_list, self.targets_list, self.fine_list = [], [], [], []
        self.year_list, self.doy_list, self.hour_list, self.time_list = [], [], [], []

        # 座標設定
        self.lat = np.linspace(21.5125,25.5 , out_shape[0])
        self.lon = np.linspace(119.0125,123, out_shape[1])

        loaded_count = 0
        for grap_path in self.actual_input_paths:
            fname = os.path.basename(grap_path)
            # 匹配時間戳 (Match timestamp)
            match = re.search(r"QPE_GFS_(\d{10})_f12.npy", fname)
            if not match: continue
            
            time_str = match.group(1)
            qpe_path = os.path.join(self.qpe_dir, f"QPE_GFS_GT_{time_str}_f12.npy")
            
            if os.path.exists(qpe_path):
                try:
                    dt = datetime.strptime(time_str, "%Y%m%d%H")
                    grap = torch.from_numpy(np.load(grap_path).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    qpe = torch.from_numpy(np.load(qpe_path).astype(np.float32)).unsqueeze(0).unsqueeze(0)

                    coarse = F.interpolate(grap, size=out_shape, mode='bilinear', align_corners=False)
                    residual = qpe - coarse

                    self.lowres_list.append(grap)
                    self.coarse_list.append(coarse)
                    self.targets_list.append(residual)
                    self.fine_list.append(qpe)
                    self.time_list.append(str(dt))
                    self.year_list.append(torch.tensor([(dt.year - 2014.0) / 100.0]))
                    self.doy_list.append(torch.tensor([((dt - datetime(dt.year, 1, 1)).days) / 365.0]))
                    self.hour_list.append(torch.tensor([dt.hour / 24.0]))
                    loaded_count += 1
                except: continue

        print(f"✅ Successfully loaded paired data: {loaded_count} samples")
        if loaded_count == 0:
            raise RuntimeError(f"Check your paths! QPE dir: {self.qpe_dir}")

        # 合併張量 (Concatenate tensors)
        self.lowres = torch.cat(self.lowres_list, dim=0)
        self.coarse = torch.cat(self.coarse_list, dim=0)
        self.fine = torch.cat(self.fine_list, dim=0)
        self.targets = torch.cat(self.targets_list, dim=0)
        self.year, self.doy, self.hour = torch.cat(self.year_list), torch.cat(self.doy_list), torch.cat(self.hour_list)

        # 正規化 
        self.norm_in = torchvision.transforms.Normalize(normalize_rawdata_mean, normalize_rawdata_std)
        self.norm_tar = torchvision.transforms.Normalize(normalize_residual_mean, normalize_residual_std)
        self.inputs = self.norm_in(self.coarse)
        self.targets = self.norm_tar(self.targets)
        
        # 關鍵反正規化函數 (The key inverse function)
        self.inv_std = normalize_residual_std.view(1, 1, 1)
        self.inv_mean = normalize_residual_mean.view(1, 1, 1)

    def residual_to_fine_image(self, residual, coarse):
        # 修正 Train 腳本 130 行的報錯
        # 將 residual 從 CPU 移回跟 coarse 一致的裝置
        res_denorm = residual * self.inv_std + self.inv_mean
        return coarse + res_denorm

    def __len__(self): return self.inputs.shape[0]

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx], "targets": self.targets[idx],
            "fine": self.fine[idx], "coarse": self.coarse[idx], "coarse_low": self.lowres[idx],
            "year": self.year[idx], "doy": self.doy[idx], "hour": self.hour[idx],
            "time": self.time_list[idx]
        }

def get_dataset_and_stats(train_ratio=0.9):
    grap_dir = "/8THDD6/Zrnie/QPE_data/12hr_025deg"
    qpe_dir = "/8THDD6/Zrnie/QPE_data/12hr_00125deg_GT"
    grap_files = sorted(glob.glob(os.path.join(grap_dir, "QPE_GFS_*_f12.npy")))
    
    valid_grap_paths, valid_times = [], []
    for gp in grap_files:
        ts = re.search(r"QPE_GFS_(\d{10})_f12.npy", os.path.basename(gp)).group(1)
        if os.path.exists(os.path.join(qpe_dir, f"QPE_GFS_GT_{ts}_f12.npy")):
            valid_grap_paths.append(gp)
            valid_times.append(datetime.strptime(ts, "%Y%m%d%H"))

    split = int(len(valid_grap_paths) * train_ratio)
    temp_ds = RainOnlyUpscaleDataset(data_paths=valid_grap_paths, qpe_dir=qpe_dir)
    
    return (valid_grap_paths[:split], valid_grap_paths[split:], 
            temp_ds.coarse.mean().item(), temp_ds.coarse.std().item(), 
            temp_ds.targets.mean().item(), temp_ds.targets.std().item(), 
            valid_times[split:])

# 初始化
train_files, test_files, mean_raw, std_raw, mean_residual, std_residual, test_times = get_dataset_and_stats()
