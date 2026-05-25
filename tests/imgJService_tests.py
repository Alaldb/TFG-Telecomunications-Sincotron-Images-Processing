import unittest
import subprocess
import os
import cv2
import numpy as np
from processing.imgJService import ImageJ


class TestImageJService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Se instancia ImageJ una sola vez para todos los tests.
        El usuario interactuará normalmente con el explorador de archivos.
        """
        print("\n[SETUP] Iniciando ImageJ — selecciona una imagen cuando aparezca el explorador...")
        cls.herramienta = ImageJ()


    # ------------------------------------------------------------------ #
    #  TEST 1 — search_fiji                                                #
    # ------------------------------------------------------------------ #

    def test_01_search_fiji_devuelve_ruta(self):
        """search_fiji() debe devolver una ruta no vacía."""
        self.assertIsNotNone(self.herramienta.fiji_path)
        self.assertNotEqual(self.herramienta.fiji_path, "")

    def test_02_search_fiji_ejecutable_existe_en_disco(self):
        """La ruta de Fiji devuelta debe existir físicamente en el disco."""
        self.assertTrue(os.path.exists(self.herramienta.fiji_path))

    def test_03_search_fiji_crea_archivo_txt(self):
        """search_fiji() debe haber generado el archivo fiji_path.txt en Elements."""
        txt_path = os.path.join(self.herramienta.folder_path, 'fiji_path.txt')
        self.assertTrue(os.path.exists(txt_path))

    def test_04_search_fiji_txt_contiene_ruta_correcta(self):
        """El contenido del TXT debe coincidir exactamente con fiji_path."""
        txt_path = os.path.join(self.herramienta.folder_path, 'fiji_path.txt')
        with open(txt_path, 'r', encoding='utf-8') as f:
            contenido = f.read()
        self.assertEqual(contenido, self.herramienta.fiji_path)


    # ------------------------------------------------------------------ #
    #  TEST 2 — search_fiji_image_path                                     #
    # ------------------------------------------------------------------ #

    def test_05_image_path_no_es_vacio(self):
        """El usuario debe haber seleccionado una imagen válida."""
        self.assertIsNotNone(self.herramienta.image_path)
        self.assertNotEqual(self.herramienta.image_path, "")

    def test_06_image_path_existe_en_disco(self):
        """La imagen seleccionada debe existir en el disco."""
        self.assertTrue(os.path.exists(self.herramienta.image_path))

    def test_07_image_path_tiene_extension_compatible(self):
        """La imagen seleccionada debe tener una extensión admitida por Fiji."""
        extensiones_validas = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
        self.assertTrue(self.herramienta.image_path.lower().endswith(extensiones_validas))


    # ------------------------------------------------------------------ #
    #  TEST 3 — normalize_image_for_fiji                                   #
    # ------------------------------------------------------------------ #

    def test_08_normalize_genera_archivo_temporal(self):
        """normalize_image_for_fiji() debe haber generado el archivo temporal."""
        self.assertTrue(os.path.exists(self.herramienta.temporal_image_path))

    def test_09_normalize_archivo_temporal_no_esta_vacio(self):
        """El archivo temporal no debe estar vacío."""
        self.assertGreater(os.path.getsize(self.herramienta.temporal_image_path), 0)

    def test_10_normalize_imagen_es_legible(self):
        """cv2 debe poder leer la imagen temporal sin errores."""
        img = cv2.imread(self.herramienta.temporal_image_path, cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(img)

    def test_11_normalize_imagen_conserva_dimensiones(self):
        """La imagen temporal debe tener las mismas dimensiones que la original."""
        original = cv2.imread(self.herramienta.image_path, cv2.IMREAD_UNCHANGED)
        temporal = cv2.imread(self.herramienta.temporal_image_path, cv2.IMREAD_UNCHANGED)
        self.assertEqual(original.shape[:2], temporal.shape[:2])

    def test_12_normalize_float32_se_convierte_a_uint8(self):
        """Si la imagen original es float32, la temporal debe ser uint8."""
        original = cv2.imread(self.herramienta.image_path, cv2.IMREAD_UNCHANGED)
        temporal = cv2.imread(self.herramienta.temporal_image_path, cv2.IMREAD_UNCHANGED)
        if original.dtype == np.float32:
            self.assertEqual(temporal.dtype, np.uint8)
            self.assertGreaterEqual(temporal.min(), 0)
            self.assertLessEqual(temporal.max(), 255)
        else:
            self.skipTest("La imagen original no es float32, rama de normalización no aplicable.")


    # ------------------------------------------------------------------ #
    #  TEST 4 — apertura en Fiji                                           #
    # ------------------------------------------------------------------ #

    def test_13_fiji_se_lanza_sin_errores(self):
        """Fiji debe abrirse con la imagen temporal y devolver un PID válido."""
        proceso = subprocess.Popen(
            [self.herramienta.fiji_path, self.herramienta.temporal_image_path]
        )
        self.assertIsNotNone(proceso.pid)
        self.assertGreater(proceso.pid, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)