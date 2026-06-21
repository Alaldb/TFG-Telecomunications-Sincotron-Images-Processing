# -*- coding: utf-8 -*-
"""
Script de prueba: Auto Brightness/Contrast
Replica el comportamiento de Image > Adjust > Brightness/Contrast > Auto de ImageJ.

Pipeline:
    1. Carga la imagen (cualquier dtype)
    2. Normaliza a uint8 [0, 255]
    3. Aplica Auto B&C sobre la imagen normalizada

Uso:
    python test_auto_bc.py
    python test_auto_bc.py ruta/a/imagen.tif
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# --- Algoritmo ---

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normaliza cualquier dtype a uint8 [0, 255] por rango min-max."""
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)


def auto_brightness_contrast(
    image: np.ndarray,
    saturated_pct: float = 0.35,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Replica Image > Adjust > Brightness/Contrast > Auto de ImageJ.

    Parametros
    ----------
    image        : imagen uint8 de entrada (2D)
    saturated_pct: porcentaje de pixeles saturados en cada extremo (defecto 0.35)
    roi_mask     : mascara booleana opcional; si se pasa, el calculo usa solo esos pixeles

    Retorna
    -------
    Imagen uint8 con el contraste ajustado
    """
    pixels = image[roi_mask].astype(np.float32) if roi_mask is not None else image.astype(np.float32).ravel()

    p_low  = np.percentile(pixels, saturated_pct)
    p_high = np.percentile(pixels, 100.0 - saturated_pct)

    if p_high == p_low:
        raise ValueError(f"Rango de imagen nulo: p_low = p_high = {p_low}.")

    inicio, final=  histogram_range(image, coverage=0.90)
    stretched = np.clip((image.astype(np.float32) - p_low) / (p_high - p_low) * 255, 0, 50)
    return stretched.astype(np.uint8)


def histogram_range(
    image: np.ndarray,
    coverage: float = 0.80,
    roi_mask: np.ndarray | None = None,
) -> tuple[int, int]:
    """
    Encuentra el intervalo minimo [v_low, v_high] cuya integral del histograma
    representa al menos `coverage` fraccion del total de pixeles.

    Usa ventana deslizante sobre la CDF del histograma (256 bins, imagen uint8).

    Parametros
    ----------
    image    : imagen uint8 2D
    coverage : fraccion objetivo (defecto 0.80 = 80%)
    roi_mask : mascara booleana opcional

    Retorna
    -------
    (v_low, v_high) : valores de pixel que delimitan el tramo minimo
    """
    pixels = image[roi_mask].ravel() if roi_mask is not None else image.ravel()
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    target = hist.sum() * coverage

    best = (0, 255)
    best_width = 256
    window_sum = int(hist[0])
    j = 0

    for i in range(256):
        while window_sum < target and j < 255:
            j += 1
            window_sum += int(hist[j])

        if window_sum >= target:
            width = j - i
            if width < best_width:
                best_width = width
                best = (i, j)

        window_sum -= int(hist[i])

    return i,j


# --- Visualizacion ---

def plot_comparison(
    original: np.ndarray,
    normalized: np.ndarray,
    result: np.ndarray,
    image_path: str,
) -> None:
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"Auto Brightness/Contrast -- {image_path}", fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)

    # 1. Imagen original
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original, cmap="gray")
    ax_orig.set_title(f"1. Original  ({original.dtype})\nmin={original.min():.3f}  max={original.max():.3f}")
    ax_orig.axis("off")

    # 2. Normalizada a uint8
    ax_norm = fig.add_subplot(gs[0, 1])
    ax_norm.imshow(normalized, cmap="gray")
    ax_norm.set_title(f"2. Normalizada (uint8)\nmin={normalized.min()}  max={normalized.max()}")
    ax_norm.axis("off")

    # 3. Auto B&C sobre la normalizada
    ax_res = fig.add_subplot(gs[0, 2])
    ax_res.imshow(result, cmap="gray")
    ax_res.set_title(f"3. Auto B&C  (uint8)\nmin={result.min()}  max={result.max()}")
    ax_res.axis("off")

    # Histograma original
    ax_h1 = fig.add_subplot(gs[1, 0])
    ax_h1.hist(original.astype(np.float32).ravel(), bins=100, color="steelblue", alpha=0.8)
    ax_h1.set_title("Histograma original")
    ax_h1.set_xlabel("Valor de pixel")
    ax_h1.set_ylabel("Frecuencia")
    ax_h1.grid(linestyle="--", alpha=0.5)

    # Histograma normalizada
    ax_h2 = fig.add_subplot(gs[1, 1])
    ax_h2.hist(normalized.ravel(), bins=100, color="seagreen", alpha=0.8)
    ax_h2.set_title("Histograma normalizada")
    ax_h2.set_xlabel("Valor de pixel (0-255)")
    ax_h2.set_ylabel("Frecuencia")
    ax_h2.grid(linestyle="--", alpha=0.5)

    # Histograma resultado
    ax_h3 = fig.add_subplot(gs[1, 2])
    ax_h3.hist(result.ravel(), bins=100, color="darkorange", alpha=0.8)
    ax_h3.set_title("Histograma tras Auto B&C")
    ax_h3.set_xlabel("Valor de pixel (0-255)")
    ax_h3.set_ylabel("Frecuencia")
    ax_h3.grid(linestyle="--", alpha=0.5)

    plt.show()


# --- Entry point ---

if __name__ == "__main__":
    DEFAULT_IMAGE = r"Images ALBA - Sample PyHM004/Low temperature/DICHROISM.tif"

    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    print(f"Cargando imagen: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"ERROR: No se pudo cargar la imagen en '{image_path}'")
        sys.exit(1)

    print(f"  dtype : {image.dtype}")
    print(f"  shape : {image.shape}")
    print(f"  rango : [{image.min():.4f}, {image.max():.4f}]")

    normalized = normalize_to_uint8(image)
    print(f"\nNormalizada a uint8:")
    print(f"  dtype : {normalized.dtype}")
    print(f"  rango : [{normalized.min()}, {normalized.max()}]")

    result = auto_brightness_contrast(normalized, saturated_pct=0.35)
    print(f"\nResultado tras Auto B&C:")
    print(f"  dtype : {result.dtype}")
    print(f"  rango : [{result.min()}, {result.max()}]")

    plot_comparison(image, normalized, result, image_path)
