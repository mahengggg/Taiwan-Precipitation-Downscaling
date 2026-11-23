# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import os
import re
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from torch.utils.data import Subset, DataLoader
from tqdm.auto import tqdm

from Network import EDMPrecond
from datasetTaiwan import (
    RainOnlyUpscaleDataset,
    get_dataset_and_stats
)


# ============================================================
# 1. 全域設定
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

EXTERNAL_GRAP_DIR = (
    r"C:\Users\user\Desktop\GRAP_12h_Accumulated"
)

MODEL_WEIGHT_PATH = "./Model/45.pt"
SHAPEFILE_PATH = "D:/onlytw_shp/taiwan.shp"
OUTPUT_DIR = "./rainfall_comparism_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W = 320, 320

_LAND_MASK_CACHE = None
_TAIWAN_FEATURE_CACHE = None

RANDOM_SEED = 42

# 完全沿用第一個專案的 EDM 採樣參數
NUM_STEPS = 40
SIGMA_MIN = 0.002
SIGMA_MAX = 80
RHO = 7

S_CHURN = 40
S_MIN = 0
S_MAX = float("inf")
S_NOISE = 1


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 降雨色階
# 與第一個專案完全相同
# ============================================================

nws_precip_colors = [
    '#ffffff',
    '#98ffff',
    '#00ceff',
    '#009aff',
    '#006af7',
    '#2e9c00',
    '#2bff00',
    '#fefe08',
    '#ffcb00',
    '#ff9c00',
    '#fe0005',
    '#c90200',
    '#9d0000',
    '#9a009d',
    '#cf00d7',
    '#ff00f7',
    '#fdcafe'
]

clevels = [
    0, 1, 2, 6, 10, 15, 20,
    30, 40, 50, 70, 90, 110,
    130, 150, 200, 300, 600
]

custom_cmap = mcolors.ListedColormap(
    nws_precip_colors
)

custom_norm = mcolors.BoundaryNorm(
    clevels,
    custom_cmap.N
)


# ============================================================
# 3. 台灣 Shapefile 與 Land Mask
# ============================================================

def _get_taiwan_feature():

    global _TAIWAN_FEATURE_CACHE

    if _TAIWAN_FEATURE_CACHE is not None:
        return _TAIWAN_FEATURE_CACHE

    shapes = list(
        shpreader.Reader(SHAPEFILE_PATH).geometries()
    )

    _TAIWAN_FEATURE_CACHE = ShapelyFeature(
        shapes,
        ccrs.PlateCarree(),
        edgecolor="black",
        facecolor="none",
        linewidth=0.5
    )

    return _TAIWAN_FEATURE_CACHE


def _get_land_mask(lon2d, lat2d):

    global _LAND_MASK_CACHE

    if _LAND_MASK_CACHE is not None:
        if _LAND_MASK_CACHE.shape == lon2d.shape:
            return _LAND_MASK_CACHE

    from shapely.ops import unary_union
    from shapely.geometry import Point
    from shapely.prepared import prep

    shapes = list(
        shpreader.Reader(SHAPEFILE_PATH).geometries()
    )

    land_geom = prep(unary_union(shapes))

    flat_points = np.column_stack([
        lon2d.ravel(),
        lat2d.ravel()
    ])

    mask = np.array([
        land_geom.contains(Point(x, y))
        for x, y in flat_points
    ]).reshape(lon2d.shape)

    _LAND_MASK_CACHE = mask

    return mask


# ============================================================
# 4. 外部 GraphCast 讀取
# ============================================================

def format_time_for_filename(raw_time_str):

    digits = "".join(
        re.findall(r"\d+", str(raw_time_str))
    )

    return digits[:10]


def get_external_grap(raw_time_str):

    formatted_time = format_time_for_filename(
        raw_time_str
    )

    filename = (
        f"QPE_GFS_{formatted_time}_f12_0.25deg.npy"
    )

    file_path = os.path.join(
        EXTERNAL_GRAP_DIR,
        filename
    )

    if not os.path.exists(file_path):

        print(
            f"⚠️ 找不到 GraphCast: {file_path}"
        )

        return None

    data = np.load(file_path)

    data = np.asarray(
        data,
        dtype=np.float32
    )

    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(
            f"GraphCast shape 錯誤: {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"GraphCast 包含 NaN 或 Inf: {file_path}"
        )

    print(
        f"✅ GraphCast Loaded: {data.shape}"
    )

    # [B, C, H, W]
    return (
        torch.from_numpy(data)
        .unsqueeze(0)
        .unsqueeze(0)
    )


# ============================================================
# 5. GraphCast 前處理
# 完全對應 datasetTaiwan.py
# ============================================================

def prepare_graphcast_input(
    external_grap,
    mean_raw,
    std_raw
):

    # Dataset 中使用：
    # F.interpolate(grap, size=out_shape,
    #               mode='bilinear',
    #               align_corners=False)

    coarse = F.interpolate(
        external_grap,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    # 對應：
    # self.inputs = self.norm_in(self.coarse)

    mean = torch.as_tensor(
        mean_raw,
        dtype=coarse.dtype,
        device=coarse.device
    ).reshape(1, 1, 1, 1)

    std = torch.as_tensor(
        std_raw,
        dtype=coarse.dtype,
        device=coarse.device
    ).reshape(1, 1, 1, 1)

    if torch.any(std == 0):
        raise ValueError("std_raw 不可為 0")

    images_input = (
        coarse - mean
    ) / std

    return coarse, images_input


# ============================================================
# 6. QM Trend
# 與第一個專案使用相同公式
# ============================================================

def build_live_train_trend(train_dataset):

    print(
        f"🚀 準備處理 {len(train_dataset)} "
        f"張訓練影像計算 QM Trend..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    sum_fine = torch.zeros((H, W))
    sum_coarse = torch.zeros((H, W))

    pbar = tqdm(
        train_loader,
        desc="Calculating QM Trend"
    )

    with torch.no_grad():

        for batch in pbar:

            fine = batch["fine"].cpu()
            coarse = batch["coarse"].cpu()

            sum_fine += (
                fine.sum(dim=0).squeeze()
            )

            sum_coarse += (
                coarse.sum(dim=0).squeeze()
            )

    trend_map = (
        sum_fine / (sum_coarse + 1e-6)
    ).numpy()

    return np.clip(
        trend_map,
        0,
        15
    )


# ============================================================
# 7. EDM Sampling
# 與第一個專案的 Heun + Churn 完全相同
# ============================================================

@torch.no_grad()
def sample_model_EDS(
    input_batch,
    external_grap,
    model,
    device,
    dataset,
    mean_raw,
    std_raw,
    num_steps=NUM_STEPS,
    sigma_min=SIGMA_MIN,
    sigma_max=SIGMA_MAX,
    rho=RHO,
    S_churn=S_CHURN,
    S_min=S_MIN,
    S_max=S_MAX,
    S_noise=S_NOISE
):

    # --------------------------------------------------------
    # A. 唯一不同的資料來源
    # --------------------------------------------------------

    coarse, images_input = prepare_graphcast_input(
        external_grap,
        mean_raw,
        std_raw
    )

    images_input = images_input.to(device)

    # coarse 必須維持實際降雨量
    # 因為 residual_to_fine_image 需要它
    coarse = coarse.cpu()

    fine = input_batch["fine"]

    # --------------------------------------------------------
    # B. 時間條件
    # 與第一份程式相同
    # --------------------------------------------------------

    condition_params = torch.stack(
        (
            input_batch["doy"].to(device),
            input_batch["hour"].to(device)
        ),
        dim=1
    )

    # --------------------------------------------------------
    # C. EDM noise schedule
    # --------------------------------------------------------

    sigma_min = max(
        sigma_min,
        model.sigma_min
    )

    sigma_max = min(
        sigma_max,
        model.sigma_max
    )

    init_noise = torch.randn(
        (
            images_input.shape[0],
            1,
            H,
            W
        ),
        dtype=torch.float64,
        device=device
    )

    step_indices = torch.arange(
        num_steps,
        dtype=torch.float64,
        device=device
    )

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (
            sigma_min ** (1 / rho)
            - sigma_max ** (1 / rho)
        )
    ) ** rho

    t_steps = torch.cat([
        model.round_sigma(t_steps),
        torch.zeros_like(t_steps[:1])
    ])

    x_next = (
        init_noise.to(torch.float64)
        * t_steps[0]
    )

    # --------------------------------------------------------
    # D. EDM Heun + Churn Sampling
    # --------------------------------------------------------

    for i, (t_cur, t_next) in enumerate(
        zip(t_steps[:-1], t_steps[1:])
    ):

        x_cur = x_next

        gamma = (
            min(
                S_churn / num_steps,
                np.sqrt(2) - 1
            )
            if S_min <= t_cur <= S_max
            else 0
        )

        t_hat = model.round_sigma(
            t_cur + gamma * t_cur
        )

        x_hat = (
            x_cur
            + (
                t_hat ** 2 - t_cur ** 2
            ).sqrt()
            * S_noise
            * torch.randn_like(x_cur)
        )

        denoised = model(
            x_hat,
            t_hat,
            images_input,
            condition_params
        ).to(torch.float64)

        d_cur = (
            x_hat - denoised
        ) / t_hat

        x_next = (
            x_hat
            + (t_next - t_hat) * d_cur
        )

        if i < num_steps - 1:

            denoised = model(
                x_next,
                t_next,
                images_input,
                condition_params
            ).to(torch.float64)

            d_prime = (
                x_next - denoised
            ) / t_next

            x_next = (
                x_hat
                + (t_next - t_hat)
                * (
                    0.5 * d_cur
                    + 0.5 * d_prime
                )
            )

    # --------------------------------------------------------
    # E. Residual -> Fine Rainfall
    # --------------------------------------------------------

    # datasetTaiwan.py 定義：
    #
    # res_denorm = residual * inv_std + inv_mean
    # predicted = coarse + res_denorm

    predicted = dataset.residual_to_fine_image(
        x_next.detach().cpu(),
        coarse
    )

    return (
        coarse,
        fine,
        predicted,
        input_batch["time"]
    )


# ============================================================
# 8. 繪圖
# 完全沿用第一個專案設定
# ============================================================

def plot_2x2_comparison(
    coarse,
    fine,
    predicted,
    qm_trend_map,
    time_str,
    dataset,
    typhoon_name
):

    lat_fine = dataset.lat[::-1]
    lon_fine = dataset.lon

    lon2d, lat2d = np.meshgrid(
        lon_fine,
        lat_fine
    )

    land_mask = _get_land_mask(
        lon2d,
        lat2d
    )

    # 第一個專案的繪圖處理
    # 注意：這裡是展示用途，不是 DM condition
    img_c_raw = coarse[0:1, 0:1]

    img_c16 = F.interpolate(
        img_c_raw,
        size=(16, 16),
        mode="area"
    )

    img_coarse_320 = F.interpolate(
        img_c16,
        size=(H, W),
        mode="nearest"
    ).squeeze().numpy()

    img_qm = (
        img_coarse_320 * qm_trend_map
    )

    img_fine = (
        fine[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    img_pred = (
        predicted[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        dpi=1000,
        subplot_kw={
            "projection": ccrs.PlateCarree()
        }
    )

    axs = axs.flatten()

    titles = [
        "GraphCast (Coarse Input)",
        "QPESUMS (Fine)",
        "QM",
        "DM"
    ]

    imgs = [
        img_coarse_320,
        img_fine,
        img_qm,
        img_pred
    ]

    taiwan_feat = _get_taiwan_feature()

    for i, (ax, img, title) in enumerate(
        zip(axs, imgs, titles)
    ):

        ax.set_extent(
            [119.5, 122.5, 21.5, 25.5],
            crs=ccrs.PlateCarree()
        )

        if taiwan_feat is not None:
            ax.add_feature(taiwan_feat)

        img_masked = np.where(
            land_mask,
            img,
            np.nan
        )

        cf = ax.pcolormesh(
            lon2d,
            lat2d,
            img_masked,
            cmap=custom_cmap,
            norm=custom_norm,
            shading="auto"
        )

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold"
        )

        gl = ax.gridlines(
            draw_labels=True,
            alpha=0.2
        )

        gl.top_labels = False
        gl.right_labels = False

        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

    plt.subplots_adjust(
        wspace=0.0005,
        hspace=0.12
    )

    cbar_ax = fig.add_axes([
        0.15,
        0.05,
        0.7,
        0.02
    ])

    fig.colorbar(
        cf,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=clevels,
        label="Rainfall (mm)"
    )

    plt.suptitle(
        "Rainfall Verification: "
        "GraphCast vs. QPESUMS vs. QM vs. DM\n"
        f" {time_str} {typhoon_name}",
        fontsize=18,
        fontweight="bold",
        y=0.96
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{typhoon_name}.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"📊 已存檔 (Saved): {save_path}"
    )


# ============================================================
# 9. 主程式
# ============================================================

if __name__ == "__main__":

    set_seed(RANDOM_SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"🚀 使用裝置: {device}")

    typhoon_indices = {
        408: "GAEMI",
        547: "KRATHON",
        601: "KONG-REY",
        631: "USAGI"
    }

    # --------------------------------------------------------
    # A. 使用相同 Dataset statistics
    # --------------------------------------------------------

    (
        train_f,
        test_f,
        m_r,
        s_r,
        m_res,
        s_res,
        _
    ) = get_dataset_and_stats()

    print(f"Mean Raw: {m_r:.6f}")
    print(f"Std Raw: {s_r:.6f}")
    print(f"Mean Residual: {m_res:.6f}")
    print(f"Std Residual: {s_res:.6f}")

    # --------------------------------------------------------
    # B. Train Dataset
    # --------------------------------------------------------

    ds_train = RainOnlyUpscaleDataset(
        data_paths=train_f,
        is_netcdf=False,
        normalize_rawdata_mean=torch.Tensor([m_r]),
        normalize_rawdata_std=torch.Tensor([s_r]),
        normalize_residual_mean=torch.Tensor([m_res]),
        normalize_residual_std=torch.Tensor([s_res])
    )

    # --------------------------------------------------------
    # C. QM Trend
    # --------------------------------------------------------

    historical_trend = build_live_train_trend(
        ds_train
    )

    # --------------------------------------------------------
    # D. Test Dataset
    # --------------------------------------------------------

    ds_test = RainOnlyUpscaleDataset(
        data_paths=test_f,
        is_netcdf=False,
        normalize_rawdata_mean=torch.Tensor([m_r]),
        normalize_rawdata_std=torch.Tensor([s_r]),
        normalize_residual_mean=torch.Tensor([m_res]),
        normalize_residual_std=torch.Tensor([s_res])
    )

    # --------------------------------------------------------
    # E. 相同 EDM Network
    # --------------------------------------------------------

    diff_model = EDMPrecond(
        img_resolution=(320, 320),
        in_channels=1,
        out_channels=1,
        label_dim=2,
        model_channels=64,
        channel_mult=[1, 2, 2],
        num_blocks=1
    ).to(device)

    # --------------------------------------------------------
    # F. 載入相同權重
    # --------------------------------------------------------

    if os.path.exists(MODEL_WEIGHT_PATH):

        print(
            f"🚀 載入模型權重: {MODEL_WEIGHT_PATH}"
        )

        diff_model.load_state_dict(
            torch.load(
                MODEL_WEIGHT_PATH,
                map_location=device,
                weights_only=True
            )
        )

        diff_model.eval()

        # ----------------------------------------------------
        # G. 逐一處理測試颱風
        # ----------------------------------------------------

        for idx, name in typhoon_indices.items():

            if idx >= len(ds_test):

                print(
                    f"⚠️ Index {idx} "
                    f"超出測試集範圍，跳過 {name}"
                )

                continue

            print(
                f"\n🌀 正在處理: {name} "
                f"(Index: {idx})"
            )

            test_loader = DataLoader(
                Subset(ds_test, [idx]),
                batch_size=1,
                shuffle=False
            )

            for batch in test_loader:

                time_val = batch["time"][0]

                # 讀取外部 GraphCast
                external_grap = get_external_grap(
                    time_val
                )

                if external_grap is None:
                    continue

                # 相同初始隨機種子
                set_seed(RANDOM_SEED)

                # EDM inference
                coarse, fine, pred, time_info = (
                    sample_model_EDS(
                        input_batch=batch,
                        external_grap=external_grap,
                        model=diff_model,
                        device=device,
                        dataset=ds_test,
                        mean_raw=m_r,
                        std_raw=s_r
                    )
                )

                # 2x2 Comparison
                plot_2x2_comparison(
                    coarse,
                    fine,
                    pred,
                    historical_trend,
                    time_info[0],
                    ds_test,
                    name
                )

    else:

        print(
            f"⚠️ 找不到權重檔: "
            f"{MODEL_WEIGHT_PATH}"
        )

    print(
        "\n✅ GraphCast 降尺度繪圖完成！"
    )
