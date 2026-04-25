from pathlib import Path
import cv2
import tifffile as tiff
import numpy as np

class Directory:
    def __init__(self, path, method="ising"):
        enum_methods = ["ising", "thresholding"]
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"El archivo {self.path} no existe.")
        if method not in enum_methods:
            raise ValueError(f"El método {method} no es válido.")
        self.img_array=self.create_img_array()
        self.method = method
        
    def create_img_array(self):
        img = []
        if self.path.is_file():
            img.append(self.path)
        elif self.path.is_dir():
            for image in self.path.iterdir():
                if image.is_file() and image.suffix.lower() in ['.png', '.tif', '.tiff']:
                    img.append(image)
        return img
        