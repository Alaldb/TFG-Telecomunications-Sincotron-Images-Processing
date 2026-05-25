from matplotlib import pyplot as plt
import numpy as np
import cv2


class Corrector:
    def __init__(self, coverage: float = 0.80):
        self.coverage = coverage

    def apply_correction(
        self,
        image: np.ndarray,
        coverage: float | None = None,
        v_low: int | None = None,
        v_high: int | None = None,
        contrast: float | None = 0.3,
        brightness: float | None = 1,
        correct_illumination: bool = True 
    ) -> np.ndarray:
        coverage = coverage if coverage is not None else self.coverage
        mask = self.create_mask(image)
        if correct_illumination:
            image = self.correct_illumination(image, mask)
        normalized = self.normalize_to_uint8(image)
        if v_low is None and v_high is None:
            v_low, v_high = self.histogram_range(normalized, coverage, mask)
        streched=self.linear_stretch(normalized, v_low, v_high)
        return self.adjust_brightness_contrast(streched,brightness=brightness,contrast=contrast)

    def search_circles(self, image: np.ndarray):
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.medianBlur(img, 5)
        height = image.shape[0]
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=height,
            param1=100,
            param2=100,
            minRadius=int(height * 0.4),
            maxRadius=int(height * 0.55),
        )
        if circles is not None:
            return [(x, y, r) for x, y, r in np.round(circles[0]).astype(int)]
        print("No se detecto ningun circulo.")
        return None

    def create_mask(self, image: np.ndarray) -> np.ndarray:
        circles = self.search_circles(image)
        if circles is None:
            return np.ones(image.shape, dtype=bool)
        mask = np.zeros(image.shape, dtype=np.uint8)
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, 1, thickness=-1)
        return mask.astype(bool)

    def normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    def histogram_range(
        self, image: np.ndarray, coverage: float, mask=None
    ) -> tuple[int, int]:
        pixels = image[mask].ravel() if mask is not None else image.ravel()
        hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
        cumsum = np.cumsum(hist)
        v_low  = int(np.searchsorted(cumsum, (1 - coverage) / 2 * cumsum[-1]))
        v_high = int(np.searchsorted(cumsum, (1 - (1 - coverage) / 2) * cumsum[-1]))
        return v_low, v_high

    def linear_stretch(self, image: np.ndarray, min_intensity: int, max_intensity: int) -> np.ndarray:
        if max_intensity == min_intensity:
            raise ValueError(f"Rango nulo: v_low = v_high = {min_intensity}.")
        stretched = np.clip(
            (image.astype(np.float32) - min_intensity) / (max_intensity - min_intensity) * 255, 0, 255
        )
        return stretched.astype(np.uint8)
    
    def adjust_brightness_contrast(self, image: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
        if contrast <= 0:
            raise ValueError("El contraste debe ser mayor que 0.")
        center = (255/2) - brightness
        window_width = 255.0 / contrast
        v_low = int(center - (window_width / 2.0))
        v_high = int(center + (window_width / 2.0))
        return self.linear_stretch(image, v_low, v_high)
    
    def correct_illumination(self, image: np.ndarray, mask: np.ndarray, kernel_ratio: float = 0.15) -> np.ndarray:
        size = max(image.shape)
        ksize = int(size * kernel_ratio)
        if ksize % 2 == 0:
            ksize += 1

        information_mean = float(image[mask].mean())
        filled = image.copy()
        filled[~mask] = information_mean

        background = cv2.GaussianBlur(filled, (ksize, ksize), sigmaX=0)
        corrected = image - background + float(background[mask].mean())

        result = image.copy()
        result[mask] = corrected[mask]

        return result


if __name__ == "__main__":
    image_path = r"Images ALBA - Sample PyHM004/Low temperature/primera.tif"

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"ERROR: No se pudo cargar '{image_path}'")
        raise SystemExit(1)

    corrector = Corrector(coverage=0.80)
    result = corrector.apply_correction(image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Original ({image.dtype})")
    axes[0].axis("off")
    axes[1].imshow(result, cmap="gray")
    axes[1].set_title("Corregida (uint8)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
