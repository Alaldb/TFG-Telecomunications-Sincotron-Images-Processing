import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.isingMethodService import Ising
from processing.graphCutsService import GraphCutsService

# ── Parámetros del test ───────────────────────────────────────────────────────
SIZE        = 400
NOISE_RATIO = 0.05
NUM_STATES  = 4
SEED        = 42

np.random.seed(SEED)

# ── Crear imagen sintética ────────────────────────────────────────────────────
half  = SIZE // 2
image = np.zeros((SIZE, SIZE), dtype=np.float32)
image[:half, :half] = 50
image[:half, half:] = 200
image[half:, :half] = 150
image[half:, half:] = 100

# Guardar máscara de píxeles ruidosos antes de añadirlos
noisy_mask = np.zeros((SIZE, SIZE), dtype=bool)
n_noisy    = int(SIZE * SIZE * NOISE_RATIO)
noise_rows = np.random.randint(0, SIZE, n_noisy)
noise_cols = np.random.randint(0, SIZE, n_noisy)
noise_vals = np.random.choice([50, 100, 150, 200], n_noisy)
image[noise_rows, noise_cols] = noise_vals
noisy_mask[noise_rows, noise_cols] = True

# ── Helper: porcentaje de ruido no eliminado ──────────────────────────────────
def noise_remaining(segmentation: np.ndarray) -> float:
    """
    Para cada cuadrante encuentra la etiqueta dominante en los píxeles limpios.
    Luego cuenta qué porcentaje de píxeles ruidosos tienen una etiqueta distinta
    a la dominante de su cuadrante → ruido no eliminado.
    """
    quads = {
        0: (0,    half, 0,    half),
        1: (0,    half, half, SIZE),
        2: (half, SIZE, 0,    half),
        3: (half, SIZE, half, SIZE),
    }

    # Etiqueta dominante por cuadrante (solo píxeles limpios)
    quad_label = {}
    for q, (r0, r1, c0, c1) in quads.items():
        region       = segmentation[r0:r1, c0:c1]
        region_noisy = noisy_mask[r0:r1, c0:c1]
        clean_vals   = region[~region_noisy & (region >= 0)]
        if len(clean_vals) > 0:
            quad_label[q] = np.bincount(clean_vals.astype(int)).argmax()

    # Cuadrante esperado para cada píxel ruidoso
    wrong = 0
    for r, c in zip(noise_rows, noise_cols):
        q = 0 if (r < half and c < half) else \
            1 if (r < half) else \
            2 if (c < half) else 3
        if segmentation[r, c] != quad_label.get(q, -1):
            wrong += 1

    total = noisy_mask.sum()
    return 100 * wrong / total if total > 0 else 0.0

# ── Correr modelos ────────────────────────────────────────────────────────────
ising = Ising(beta=2, max_iterations=20, num_states=NUM_STATES)
t0 = time.perf_counter()
ising.run(image)
t_ising = time.perf_counter() - t0
noise_ising = noise_remaining(ising.final_image)

gc = GraphCutsService(
    num_states=NUM_STATES,
    lambda_value=100,
    sigma=None,
    num_iterations=-1,
    number_gaussians_per_state=1
)
t0 = time.perf_counter()
gc.run(image)
t_gc = time.perf_counter() - t0
noise_gc = noise_remaining(gc.final_image)

print(f"ICM        ->  tiempo: {t_ising:.3f}s")
print(f"             ruido restante: {noise_ising:.1f}%")
print(f"Graph Cuts ->  tiempo: {t_gc:.3f}s")
print(f"             ruido restante: {noise_gc:.1f}%")

# ── Visualizar ────────────────────────────────────────────────────────────────
colors = ["black"] + list(mcolors.TABLEAU_COLORS.values())[:NUM_STATES]
cmap   = mcolors.ListedColormap(colors[:NUM_STATES + 1])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"ICM vs Graph Cuts  —  ruido inicial {NOISE_RATIO*100:.0f}%", fontsize=14)

axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis("off")

im1 = axes[1].imshow(ising.iterative_matrix, cmap=cmap, vmin=-1, vmax=NUM_STATES - 1)
axes[1].set_title("KMeans inicial")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], ticks=range(-1, NUM_STATES))

im2 = axes[2].imshow(ising.final_image, cmap=cmap, vmin=-1, vmax=NUM_STATES - 1)
axes[2].set_title(f"ICM  ({t_ising:.2f}s)\n  —  ruido restante: {noise_ising:.1f}%")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], ticks=range(-1, NUM_STATES))

im3 = axes[3].imshow(gc.final_image, cmap=cmap, vmin=-1, vmax=NUM_STATES - 1)
axes[3].set_title(f"Graph Cuts  ({t_gc:.2f}s)\n  —  ruido restante: {noise_gc:.1f}%")
axes[3].axis("off")
plt.colorbar(im3, ax=axes[3], ticks=range(-1, NUM_STATES))

plt.tight_layout()
plt.show()