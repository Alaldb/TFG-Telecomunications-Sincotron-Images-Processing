"""
IsingPlotService
================
Extrae datos de un objeto Ising y los devuelve en un diccionario estructurado.
No tiene dependencia de matplotlib: la visualización vive en las funciones plot_*
y en el bloque __main__.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from processing.isingMethodService import Ising
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PHYSICAL_LABELS = ["Black", "Gray", "White"]
PHYSICAL_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


class IsingPlotService:
    def get_plot_data(self, ising: "Ising") -> dict:
        state_map = self.build_state_map(ising)
        histogram_data = self.build_histogram_data(ising, state_map)
        images_data = self.build_images_data(ising, state_map)
        return {
            "histogram": histogram_data,
            "images": images_data,
        }

    def build_state_map(self, ising: "Ising") -> dict[int, dict]:
        sorted_states = sorted(
            range(ising.num_states),
            key=lambda state: ising.parameters[state]["mean"],
        )
        state_map: dict[int, dict] = {}
        for rank, state in enumerate(sorted_states):
            state_map[state] = {
                "label": PHYSICAL_LABELS[rank],
                "color": PHYSICAL_COLORS[rank],
            }
        return state_map

    def build_histogram_data(self, ising: "Ising", state_map: dict) -> dict:

        mask = ising.mask
        total_pixels = int(mask.sum())

        pixel_values = ising.original_image[mask].astype(np.float32)
        state_per_pixel = ising.final_image[mask].astype(np.int32)

        state_stats: dict[int, dict] = {}
        for state in range(ising.num_states):
            area_px = int((state_per_pixel == state).sum())
            percent = 100.0 * area_px / total_pixels if total_pixels > 0 else 0.0

            state_stats[state] = {
                "label":   state_map[state]["label"],
                "percent": percent,
                "area_px": area_px,
                "mean":    ising.parameters[state]["mean"],
                "std":     ising.parameters[state]["std"],
            }

        return {
            "pixel_values":    pixel_values,
            "state_per_pixel": state_per_pixel,
            "state_stats":     state_stats,
        }

    def build_images_data(self, ising: "Ising", state_map: dict) -> dict:

        height, width = ising.original_image.shape
        segmented_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        for state, info in state_map.items():
            pixels_mask = (ising.final_image == state) & ising.mask
            segmented_rgb[pixels_mask] = np.array(info["color"], dtype=np.uint8)

        state_colors: dict[int, tuple] = {
            state: (info["label"], info["color"])
            for state, info in state_map.items()
        }

        return {
            "original":      ising.original_image,
            "segmented_rgb": segmented_rgb,
            "state_colors":  state_colors,
        }


# ---------------------------------------------------------------------------
# Funciones de plot (dependen de matplotlib; no pertenecen a la clase)
# ---------------------------------------------------------------------------

def plot_window_1(data: dict) -> None:
    h_data  = data["histogram"]
    stats   = h_data["state_stats"]
    colors  = data["images"]["state_colors"]   # estado → (label, (R,G,B))

    pixel_values    = h_data["pixel_values"]
    state_per_pixel = h_data["state_per_pixel"]

    fig, (ax_hist, ax_stats) = plt.subplots(
        1, 2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle("Distribución de intensidades y estadísticas por estado",
                 fontsize=13, fontweight="bold")

    # ── Histograma solapado por estado ───────────────────────────────────
    for state, (label, rgb) in colors.items():
        state_intensities = pixel_values[state_per_pixel == state]
        if len(state_intensities) == 0:
            continue
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)
        sns.histplot(
            state_intensities,
            bins=50,
            kde=True,
            ax=ax_hist,
            color=hex_color,
            label=label,
            alpha=0.6,
            element="step",
            stat="density",
        )

    ax_hist.set_xlabel("Valor de gris original (0 - 255)")
    ax_hist.set_ylabel("Densidad de frecuencia")
    ax_hist.set_title("Distribución de Intensidad de Píxeles")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(linestyle="--", alpha=0.5)
    ax_hist.set_xlim(0, 255)

    # ── Panel de estadísticas ─────────────────────────────────────────────
    ax_stats.axis("off")
    y_pos = 0.95
    for state in sorted(stats, key=lambda s: stats[s]["mean"], reverse=True):
        info = stats[state]
        _, rgb = colors[state]
        patch_color = tuple(c / 255 for c in rgb)

        # Cuadro de color
        ax_stats.add_patch(plt.Rectangle((0.02, y_pos - 0.06), 0.10, 0.08,
                                          transform=ax_stats.transAxes,
                                          color=patch_color, clip_on=False))
        # Texto
        text = (
            f"{info['label']}   {info['percent']:.1f} %\n"
            f"  μ = {info['mean']:.1f}   σ = {info['std']:.2f}\n"
            f"  Área = {info['area_px']:,} px"
        )
        ax_stats.text(0.16, y_pos - 0.01, text,
                      transform=ax_stats.transAxes,
                      fontsize=10, verticalalignment="top",
                      fontfamily="monospace")
        y_pos -= 0.28

    ax_stats.set_title("Estadísticas finales")

    plt.tight_layout()
    plt.show()


def plot_window_2(data: dict) -> None:

    img_data = data["images"]
    colors   = img_data["state_colors"]   # estado → (label, (R,G,B))

    fig, (ax_orig, ax_seg) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Comparación imagen original vs segmentación Ising",
                 fontsize=13, fontweight="bold")

    ax_orig.imshow(img_data["original"], cmap="gray")
    ax_orig.set_title("Imagen original")
    ax_orig.axis("off")

    ax_seg.imshow(img_data["segmented_rgb"])
    ax_seg.set_title("Segmentación Ising")
    ax_seg.axis("off")

    # Leyenda con significado físico
    patches = [
        mpatches.Patch(
            color=tuple(c / 255 for c in rgb),
            label=label,
        )
        for _, (label, rgb) in sorted(colors.items(),
                                       key=lambda x: x[1][0])  # orden alfabético
    ]
    ax_seg.legend(handles=patches, loc="lower right",
                  fontsize=10, framealpha=0.85)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    # Permite ejecutar desde TFG Teleco apuntando al módulo Ising en New_Tool
    new_tool_path = os.path.join(
        os.path.dirname(__file__), "..", "TFG-Teleco", "New_Tool"
    )
    sys.path.insert(0, os.path.abspath(new_tool_path))


    print("Iniciando modelo de Ising...")
    ising = Ising(beta=3, max_iterations=100, num_states=3)
    print("Modelo completado. Extrayendo datos...")

    service = IsingPlotService()
    data = service.get_plot_data(ising)

    print("Mostrando ventana 1 (histograma + estadísticas)...")
    plot_window_1(data)

    print("Mostrando ventana 2 (comparación de imágenes)...")
    plot_window_2(data)
