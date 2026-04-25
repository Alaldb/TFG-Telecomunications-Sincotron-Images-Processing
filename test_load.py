import unittest, load
from pathlib import Path
import numpy as np
import tifffile as tiff

class TestClass(unittest.TestCase):
    def test_case(self):
        self.assertEqual(1, 1)

    def test_reach_correct_path(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png"
        image = load.Directory(path=path)
        self.assertIsInstance(image, load.Directory)
    
    def test_path_not_exist(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\archivo_inexistente.png"
        with self.assertRaises(FileNotFoundError):
            load.Directory(path=path)
    
    def test_load_folder(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica"
        directory = load.Directory(path=path)
        self.assertIsInstance(directory, load.Directory)
    
    def test_load_incorrect_folder(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\carpeta_inexistente"
        with self.assertRaises(FileNotFoundError):
            load.Directory(path=path)
    
    def test_array_len_equals_1_if_path_is_file(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png"
        image = load.Directory(path=path)
        self.assertEqual(len(image.img_array), 1)
    
    def test_array_len_equals_number_of_images_if_path_is_folder(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature"
        directory = load.Directory(path=path)
        expected_num_images = len([file for file in Path(path).iterdir() if file.is_file() and file.suffix.lower() in ['.png', '.tif', '.tiff']])
        self.assertEqual(len(directory.img_array), expected_num_images)
    
    def test_default_processing_method(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png"
        image = load.Directory(path=path)
        self.assertEqual(image.method, "ising")
    
    def test_other_valid_method(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png"
        image = load.Directory(path=path, method="thresholding")
        self.assertEqual(image.method, "thresholding")
    
    def test_invalid_method(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png"
        with self.assertRaises(ValueError):
            load.Directory(path=path, method="invalid_method")
    
    

    

if __name__ == "__main__":
    unittest.main(verbosity=2)