from __future__ import annotations

import numpy as np
from skimage.measure import label
from core.segmentationContainer import SegmentationContainer


class DomainService:

    def __init__(self, segmentation_container: SegmentationContainer, seed = 42):
        self.ising = segmentation_container
        self.seed = seed
        self.binary_images= self.extract_state_images()
        self.labeled_images= self.label_domains()
        self.colored_images= self.color_domains()
        self.domain_data= self.extract_domain_data()

    def extract_state_images(self):
        result = {}
        for state in range(self.ising.num_states):
            binary = np.zeros(self.ising.final_image.shape, dtype=np.uint8)
            binary[(self.ising.final_image == state) & self.ising.mask] = 1
            result[state] = binary
        return result

    def label_domains(self):
        result = {}
        for state, binary in self.binary_images.items():
            result[state] = label(binary, connectivity=2)
        return result

    def color_domains(self):
        rng = np.random.default_rng(self.seed)
        result = {}
        for state, labeled in self.labeled_images.items():
            height, width = labeled.shape
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            num_domains = labeled.max()
            colors = rng.integers(60, 255, size=(num_domains + 1, 3), dtype=np.uint8)
            colors[0] = (0, 0, 0)  #background
            for domain_id in range(1, num_domains + 1):
                rgb[labeled == domain_id] = colors[domain_id]
            result[state] = rgb
        return result
    
    def extract_domain_data(self) -> dict[int, dict]:
        result = {}
        for state, labeled in self.labeled_images.items():
            num_domains = labeled.max()
            result[state] = {}
            for domain_id in range(1, num_domains + 1):
                mask = labeled == domain_id
                coords = np.argwhere(mask)
                values = self.ising.original_image[mask]
                result[state][domain_id] = {
                    "coords": coords,
                    "values": values,
                }
        return result

    def get_data(self) -> dict:
        return {
            "original":      self.ising.original_image,
            "binary_images": self.binary_images,
            "labeled_images": self.labeled_images,
            "colored_images": self.colored_images,
            "num_domains":   {
                state: int(labeled.max())
                for state, labeled in self.labeled_images.items()
            },
            "domain_data": self.domain_data
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Iniciando modelo de Ising...")
    ising = Ising(beta=3, max_iterations=100, num_states=3)
    print("Modelo completado. Calculando dominios...")

    service = DomainService(ising)
    data = service.get_data()

    # Ventana: imagen original
    fig0, ax0 = plt.subplots(figsize=(6, 5))
    fig0.suptitle("Imagen original", fontsize=13, fontweight="bold")
    ax0.imshow(data["original"], cmap="gray")
    ax0.axis("off")
    plt.tight_layout()

    # Ventana por estado
    for state in range(ising.num_states):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.suptitle(
            f"Estado {state} — {data['num_domains'][state]} dominios (8-conectividad)",
            fontsize=13, fontweight="bold"
        )
        ax.imshow(data["colored_images"][state])
        ax.axis("off")
        plt.tight_layout()

    plt.show()
