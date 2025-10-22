from PIL import Image
import io
from xarray import Dataset
import numpy as np
from matplotlib import pyplot as plt
from s3_upload import save_bytes_to_s3


def save_image_s3(bucket_name: str, image_bytes, object_url: str):
    save_bytes_to_s3(
        bucket_name=bucket_name,
        object_bytes=image_bytes,
        object_key=object_url.partition(bucket_name + "/")[2],
    )


def generate_thumbnail(
    bucket_name: str,
    forecast_netcdf_file_url: str,
    thumbnail_file_urls: dict[str, str],
    forecast_dataset: Dataset,
) -> dict[str, str]:
    try:
        aa = forecast_dataset
        depth = 0
        lead = 9
        var = "zos"
        thetao = aa[var][lead]
        thetao = thetao.rio.write_crs("EPSG:4326")
        thetao_ortho = thetao.rio.reproject("EPSG:4326")
        thetao_min, thetao_max = (
            thetao_ortho.min().item(),
            thetao_ortho.max().item(),
        )
        thetao_normalized = (
            (thetao_ortho - thetao_min) / (thetao_max - thetao_min) * 255
        ).astype(np.uint8)
        cmap = plt.get_cmap("seismic")
        thetao_colored = cmap(thetao_normalized)
        alpha = np.where(np.isnan(thetao_ortho), 0, 255).astype(
            np.uint8
        )  # Transparent for NaNs
        thetao_colored[..., 3] = alpha / 255.0  # Set alpha channel in RGBA
        thetao_rgba = (thetao_colored * 255).astype(np.uint8)
        img = Image.fromarray(thetao_rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        save_image_s3(bucket_name, image_bytes, thumbnail_file_urls["zos"])

        var = "thetao"
        thetao = aa[var][lead][depth]
        thetao = thetao.rio.write_crs("EPSG:4326")
        thetao_ortho = thetao.rio.reproject("EPSG:4326")
        thetao_min, thetao_max = (
            thetao_ortho.min().item(),
            thetao_ortho.max().item(),
        )
        thetao_normalized = (
            (thetao_ortho - thetao_min) / (thetao_max - thetao_min) * 255
        ).astype(np.uint8)
        cmap = plt.get_cmap("viridis")
        thetao_colored = cmap(thetao_normalized)
        alpha = np.where(np.isnan(thetao_ortho), 0, 255).astype(
            np.uint8
        )  # Transparent for NaNs
        thetao_colored[..., 3] = alpha / 255.0  # Set alpha channel in RGBA
        thetao_rgba = (thetao_colored * 255).astype(np.uint8)
        img = Image.fromarray(thetao_rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        save_image_s3(bucket_name, image_bytes, thumbnail_file_urls["thetao"])

        var = "so"
        thetao = aa[var][lead][depth]
        thetao = thetao.rio.write_crs("EPSG:4326")
        thetao_ortho = thetao.rio.reproject("EPSG:4326")
        thetao_min, thetao_max = (
            thetao_ortho.min().item(),
            thetao_ortho.max().item(),
        )
        thetao_normalized = (
            (thetao_ortho - thetao_min) / (thetao_max - thetao_min) * 255
        ).astype(np.uint8)
        cmap = plt.get_cmap("jet")
        thetao_colored = cmap(thetao_normalized)
        alpha = np.where(np.isnan(thetao_ortho), 0, 255).astype(
            np.uint8
        )  # Transparent for NaNs
        thetao_colored[..., 3] = alpha / 255.0  # Set alpha channel in RGBA
        thetao_rgba = (thetao_colored * 255).astype(np.uint8)
        img = Image.fromarray(thetao_rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        save_image_s3(bucket_name, image_bytes, thumbnail_file_urls["so"])

        var = "uo"
        thetao = aa[var][2][0]
        thetao = thetao.rio.write_crs("EPSG:4326")
        thetao_ortho = thetao.rio.reproject("EPSG:4326")
        thetao_min, thetao_max = (
            thetao_ortho.min().item(),
            thetao_ortho.max().item(),
        )
        thetao_normalized = (
            (thetao_ortho - thetao_min) / (thetao_max - thetao_min) * 255
        ).astype(np.uint8)
        cmap = plt.get_cmap("coolwarm")
        thetao_colored = cmap(thetao_normalized)
        alpha = np.where(np.isnan(thetao_ortho), 0, 255).astype(
            np.uint8
        )  # Transparent for NaNs
        thetao_colored[..., 3] = alpha / 255.0  # Set alpha channel in RGBA
        thetao_rgba = (thetao_colored * 255).astype(np.uint8)
        img = Image.fromarray(thetao_rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        save_image_s3(bucket_name, image_bytes, thumbnail_file_urls["uo"])

        var = "vo"
        thetao = aa[var][2][0]
        thetao = thetao.rio.write_crs("EPSG:4326")
        thetao_ortho = thetao.rio.reproject("EPSG:4326")
        thetao_min, thetao_max = (
            thetao_ortho.min().item(),
            thetao_ortho.max().item(),
        )
        thetao_normalized = (
            (thetao_ortho - thetao_min) / (thetao_max - thetao_min) * 255
        ).astype(np.uint8)
        cmap = plt.get_cmap("coolwarm")
        thetao_colored = cmap(thetao_normalized)
        alpha = np.where(np.isnan(thetao_ortho), 0, 255).astype(
            np.uint8
        )  # Transparent for NaNs
        thetao_colored[..., 3] = alpha / 255.0  # Set alpha channel in RGBA
        thetao_rgba = (thetao_colored * 255).astype(np.uint8)
        img = Image.fromarray(thetao_rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        save_image_s3(bucket_name, image_bytes, thumbnail_file_urls["vo"])
    except Exception as exception:
        print(
            f"Failed to generate thumbnails for resource {forecast_netcdf_file_url}: {exception}"
        )
        raise
    return thumbnail_file_urls
