# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:11:47 2026

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 01:56:53 2025

@author: user
"""
#lr從-4改-5，以後可能-6

import os
import torch
import Network
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datasetTaiwan import RainOnlyUpscaleDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditional_img=None, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


def training_step(model, loss_fn, optimiser, data_loader, scaler, step, accum=4, writer=None, device="cuda"):
    model.train()
    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {step}")
        epoch_losses = []
        step_loss = 0

        for i, batch in enumerate(data_loader):
            tq.update(1)
            image_input = batch["inputs"].to(device)
            image_output = batch["targets"].to(device)
            day = batch["doy"].to(device)
            hour = batch["hour"].to(device)
            condition_params = torch.stack((day, hour), dim=1)

            from torch import amp
            with amp.autocast("cuda"):
                loss = loss_fn(net=model, images=image_output, conditional_img=image_input, labels=condition_params)
                loss = torch.mean(loss)

            scaler.scale(loss).backward()
            step_loss += loss.item()

            if (i + 1) % accum == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                if writer is not None:
                    writer.add_scalar("Loss/train", step_loss / accum, step * len(data_loader) + i)
                step_loss = 0

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")
        return mean_loss

@torch.no_grad()
def sample_model(model, dataloader, num_steps=40, sigma_min=0.002, sigma_max=80, rho=7,
                 S_churn=40, S_min=0, S_max=float('inf'), S_noise=1,
                 device="cuda", epoch=None):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import os
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.feature import ShapelyFeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import numpy as np
    from datetime import datetime

    # ==== 色階設定 ====
    nws_precip_colors = [
        '#ffffff', '#98ffff', '#00ceff', '#009aff', '#006af7',
        '#2e9c00', '#2bff00', '#fefe08', '#ffcb00', '#ff9c00',
        '#fe0005', '#c90200', '#9d0000', '#9a009d', '#cf00d7',
        '#ff00f7', '#fdcafe'
    ]
    clevel = [0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 600]
    cmap = mcolors.ListedColormap(nws_precip_colors)
    norm = mcolors.BoundaryNorm(clevel, cmap.N)

    # ==== 1. 取 batch 並推論 ====
    batch = next(iter(dataloader))
    images_input = batch["inputs"].to(device)
    coarse, fine = batch["coarse"], batch["fine"]
    condition_params = torch.stack((batch["doy"].to(device), batch["hour"].to(device)), dim=1)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)
    init_noise = torch.randn((images_input.shape[0], 1, images_input.shape[2], images_input.shape[3]),
                             dtype=torch.float64, device=device)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = init_noise * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        denoised = model(x_hat, t_hat, images_input, condition_params).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input, condition_params).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = dataloader.dataset.residual_to_fine_image(x_next.detach().cpu(), coarse)

    # ==== 2. 抓時間與經緯度資訊 ====
    idx = 0
    try:
        timestamp = datetime.strptime(batch["time"][idx], "%Y-%m-%d %H:%M:%S")
        time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        fname_str = timestamp.strftime("%Y-%m-%d_%H%M")
    except:
        time_str = f"index_{idx:03d}"
        fname_str = time_str

    lat = dataloader.dataset.lat[::-1]
    lon = dataloader.dataset.lon
    lon2d, lat2d = np.meshgrid(lon, lat)
    pred_img = predicted[idx, 0].numpy()

    shapefile_path ="/8THDD6/Zrnie/onlytw_shp/taiwan.shp"
    shapes = list(shpreader.Reader(shapefile_path).geometries())
    if not shapes:
        raise ValueError("Shapefile 中沒有任何幾何資料！")
    taiwan_feature = ShapelyFeature(shapes, ccrs.PlateCarree(), edgecolor='black', facecolor='none')

    # ==== 3. 繪製三張圖比較 ====
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    titles = ["Low Resolution (Coarse)", "Ground Truth (Fine)", "Model Prediction"]
    images = [coarse[idx, 0].numpy(), fine[idx, 0].numpy(), pred_img]

    for ax, img, title in zip(axs, images, titles):
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax.add_feature(taiwan_feature)

        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 10, 'color': 'gray'}
        gl.ylabel_style = {'size': 10, 'color': 'gray'}

        cf = ax.pcolormesh(lon2d, lat2d, img, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
        ax.set_title(title, fontsize=12, fontweight="bold")

    cbar = fig.colorbar(cf, ax=axs, orientation='vertical', pad=0.03, shrink=0.75, ticks=clevel)
    cbar.set_label("Rainfall (mm)", fontsize=12)

    if epoch is None:
        epoch = 0

    plt.suptitle(
        f"Spatial Downscaling of Rainfall in Taiwan: From 0.25° to 0.0125° Resolution\n{time_str} (Epoch {epoch})",
        fontsize=18, fontweight="bold", y=0.91
    )
    fname_str = f"{fname_str}_epoch{epoch}"
    os.makedirs("./results", exist_ok=True)
    fig.savefig(f"./results/comparison_{fname_str}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 比較圖已儲存：./results/comparison_{fname_str}.png")

    base_error = torch.mean(torch.abs(fine - coarse))
    pred_error = torch.mean(torch.abs(fine - predicted))
    return (base_error.item(), pred_error.item(), time_str)


def main():
    # ==== 字體設定 (Font Settings) ====
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    # =================================

    from datasetTaiwan import train_files, test_files, mean_raw, std_raw, mean_residual, std_residual

    train_dataset = RainOnlyUpscaleDataset(
        data_paths=train_files,
        is_netcdf=False,
        normalize_rawdata_mean=torch.Tensor([mean_raw]),
        normalize_rawdata_std=torch.Tensor([std_raw]),
        normalize_residual_mean=torch.Tensor([mean_residual]),
        normalize_residual_std=torch.Tensor([std_residual])
    )

    temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    first_batch = next(iter(temp_loader))
    print("Input shape:", first_batch["inputs"].shape)

    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 500
    accum = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = Network.EDMPrecond(
        img_resolution=(320, 320),
        in_channels=1,
        out_channels=1,
        label_dim=2,
        model_channels=64,
        channel_mult=[1, 2, 2],
        num_blocks=1,
        dropout=0.05
    ).to(device)

    test_dataset = RainOnlyUpscaleDataset(
        data_paths=test_files,
        is_netcdf=False,
        normalize_rawdata_mean=torch.Tensor([mean_raw]),
        normalize_rawdata_std=torch.Tensor([std_raw]),
        normalize_residual_mean=torch.Tensor([mean_residual]),
        normalize_residual_std=torch.Tensor([std_residual])
    )

    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    scaler = torch.cuda.amp.GradScaler()
    optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    writer = SummaryWriter("./runs")
    loss_fn = EDMLoss()

    os.makedirs("./Model", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    losses = []
    for step in range(num_epochs):
        epoch_loss = training_step(network, loss_fn, optimiser, dataloader_train, scaler, step, accum, writer)
        losses.append(epoch_loss)

        if step % 1 == 0:
           base_error, pred_error, time_str = sample_model(network, dataloader_test, epoch=step)
           writer.add_scalar("Error/base", base_error, step)
           writer.add_scalar("Error/pred", pred_error, step)


        if losses[-1] == min(losses):
            torch.save(network.state_dict(), f"./Model/{step}.pt")


if __name__ == "__main__":
    main()