import numpy as np
from sklearn import cluster
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Ising:
    def __init__(self, beta=2, max_iterations=20, num_states=3):
        self.beta = beta
        self.max_iterations = max_iterations
        self.num_states = num_states

    def run(self, image: np.ndarray):
        self.original_image = image
        self.mask = self.create_mask()
        self.iterative_matrix, self.parameters = self.initialize_ising_model()
        self.final_image = self.apply_ising_model_icm(self.beta, self.max_iterations)

    def search_circles(self):
        img = cv2.normalize(self.original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.medianBlur(img, 5)

        height = self.original_image.shape[0]
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=height, #Only 1 circle
            param1=100,
            param2=100,
            minRadius=int(height * 0.4),
            maxRadius=int(height * 0.55),
        )

        if circles is not None:
            circles = [(x, y, r) for x, y, r in np.round(circles[0]).astype(int)]
            return circles
        else:
            print("No se detectó ningún círculo.")

    def create_mask(self):
        circles=self.search_circles()
        if circles is None:
            return np.ones(self.original_image.shape, dtype=bool)

        mask = np.zeros(self.original_image.shape, dtype=np.uint8)
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, 1, thickness=-1)
        return mask.astype(bool)
    
    def calculate_statistical_variables(self):
        parameters = {}
        for state in range(self.num_states):
            if self.mask is not None: 
                pixel_state_values = self.original_image[(self.iterative_matrix==state) & self.mask]
            else: 
                pixel_state_values = self.original_image[(self.iterative_matrix==state)]
            if len(pixel_state_values) > 0:
                parameters[state] = {
                    'mean': np.mean(pixel_state_values),
                    'std': np.std(pixel_state_values)+1e-6
                }
            else:
                print(f"Advertencia: No se encontraron píxeles para el estado {state}. Se asignarán valores por defecto.")
                parameters[state] = {
                    'mean': 0,
                    'std': 1e-6
                }
        return parameters
    
    def initialize_ising_model(self):
        original_image_for_Kmeans = self.original_image[self.mask].reshape(-1, 1)
        kmeans = cluster.KMeans(n_clusters=self.num_states, n_init=10, random_state=64)
        clustered_image = kmeans.fit_predict(original_image_for_Kmeans)
        masked_clustered_image = np.full(self.original_image.shape, -1, dtype=np.int32)
        masked_clustered_image[self.mask] = clustered_image
        self.iterative_matrix = masked_clustered_image
        parameters = self.calculate_statistical_variables()
        return masked_clustered_image, parameters
    
    def calculate_neighbour_sum(self, state):
        match = (self.iterative_matrix == state).astype(np.float32)
        neighbour_sum = (
            np.roll(match, 1, axis=0) +   # arriba
            np.roll(match, -1, axis=0) +  # abajo
            np.roll(match, 1, axis=1) +   # izquierda
            np.roll(match, -1, axis=1)    # derecha
        )
        return neighbour_sum
    
    def apply_ising_model_icm(self, beta, max_iterations):
        matrix_iteration = self.iterative_matrix.copy()
        matrix_background = self.iterative_matrix.copy()
        for iteration in range(max_iterations):

            prev_matrix_iteration = matrix_iteration.copy()
            energy_maps = np.full((self.num_states, *self.original_image.shape), np.inf)

            for state in range(self.num_states):
                mu = self.parameters[state]['mean']
                sigma = self.parameters[state]['std']

                stat_energy = ((self.original_image - mu) / sigma) ** 2
                neigh_energy = -beta * self.calculate_neighbour_sum(state)
                energy_maps[state] = stat_energy + neigh_energy
            new_states = np.argmin(energy_maps, axis=0).astype(np.int32)
            
            matrix_iteration = np.where(self.mask, new_states, matrix_background)
            
            if np.array_equal(matrix_iteration, prev_matrix_iteration):
                print(f"Converged at iteration {iteration}")
                break
            
            self.iterative_matrix = matrix_iteration
            self.parameters = self.calculate_statistical_variables()
        
        return matrix_iteration

def plot_results(ising):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n = ising.num_states
    base_colors = ["black"] + list(mcolors.TABLEAU_COLORS.values())[:n]
    cmap_states = mcolors.ListedColormap(base_colors[:n + 1])

    # ── Cálculo de diferencia ─────────────────────────────────────────
    # Píxeles dentro de la máscara que cambiaron de estado entre KMeans e ICM
    diff = (ising.iterative_matrix != ising.final_image) & ising.mask
    pixels_totales   = int(ising.mask.sum())
    pixels_cambiados = int(diff.sum())
    porcentaje       = 100 * pixels_cambiados / pixels_totales if pixels_totales > 0 else 0

    print(f"\n── Diferencia KMeans vs ICM ──────────────────")
    print(f"  Píxeles en ROI:      {pixels_totales}")
    print(f"  Píxeles cambiados:   {pixels_cambiados}")
    print(f"  Porcentaje cambiado: {porcentaje:.2f} %")
    print(f"──────────────────────────────────────────────\n")

    # ── Figura ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle("Modelo de Ising — resultados", fontsize=14, fontweight="bold")

    axes[0].imshow(ising.original_image, cmap="gray")
    axes[0].set_title("1. Imagen original")
    axes[0].axis("off")

    axes[1].imshow(ising.mask, cmap="gray")
    axes[1].set_title("2. Máscara (ROI)")
    axes[1].axis("off")

    im3 = axes[2].imshow(ising.iterative_matrix, cmap=cmap_states, vmin=-1, vmax=n - 1)
    axes[2].set_title("3. Segmentación inicial\n(KMeans)")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], ticks=range(-1, n), label="Estado")

    im4 = axes[3].imshow(ising.final_image, cmap=cmap_states, vmin=-1, vmax=n - 1)
    axes[3].set_title("4. Segmentación final\n(ICM)")
    axes[3].axis("off")
    plt.colorbar(im4, ax=axes[3], ticks=range(-1, n), label="Estado")

    # Panel 5: diferencia (blanco = cambió, negro = igual)
    axes[4].imshow(diff, cmap="hot")
    axes[4].set_title(f"5. Diferencia KMeans vs ICM\n{pixels_cambiados} px cambiados ({porcentaje:.1f} %)")
    axes[4].axis("off")

    # Tabla de parámetros finales
    param_text = "\n".join(
        f"Estado {s}:  μ = {ising.parameters[s]['mean']:.1f}   σ = {ising.parameters[s]['std']:.2f}"
        for s in range(n)
    )
    fig.text(0.82, 0.01, param_text, ha="center", va="bottom",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    plt.tight_layout()
    plt.show()
    n = ising.num_states
    base_colors = ["black"] + list(mcolors.TABLEAU_COLORS.values())[:n]
    cmap_states = mcolors.ListedColormap(base_colors[:n + 1])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Modelo de Ising — resultados", fontsize=14, fontweight="bold")

    axes[0].imshow(ising.original_image, cmap="gray")
    axes[0].set_title("1. Imagen original")
    axes[0].axis("off")

    axes[1].imshow(ising.mask, cmap="gray")
    axes[1].set_title("2. Máscara (ROI)")
    axes[1].axis("off")

    im3 = axes[2].imshow(ising.iterative_matrix, cmap=cmap_states, vmin=-1, vmax=n - 1)
    axes[2].set_title("3. Segmentación inicial\n(KMeans)")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], ticks=range(-1, n), label="Estado")

    im4 = axes[3].imshow(ising.final_image, cmap=cmap_states, vmin=-1, vmax=n - 1)
    axes[3].set_title("4. Segmentación final\n(ICM)")
    axes[3].axis("off")
    plt.colorbar(im4, ax=axes[3], ticks=range(-1, n), label="Estado")

    param_text = "\n".join(
        f"Estado {s}:  μ = {ising.parameters[s]['mean']:.1f}   σ = {ising.parameters[s]['std']:.2f}"
        for s in range(n)
    )
    fig.text(0.75, 0.01, param_text, ha="center", va="bottom",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Iniciando modelo de Ising...")
    ising = Ising(beta=100, max_iterations=100, num_states=2)
    print("Modelo completado. Mostrando resultados...")
    plot_results(ising)

