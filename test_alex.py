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
    ) -> np.ndarray:
        coverage = coverage if coverage is not None else self.coverage
        mask = self.create_mask(image)
        normalized = self.normalize_to_uint8(image)
        if v_low is None and v_high is None:
            v_low, v_high = self.histogram_range(normalized, coverage, mask)
        return self.linear_stretch(normalized, v_low, v_high)

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

    def create_mask(self,image):
        img = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        _, mask = cv2.threshold(
            img,
            20,
            255,
            cv2.THRESH_BINARY
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((15,15), np.uint8)
        )

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
