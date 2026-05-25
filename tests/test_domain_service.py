import unittest
from unittest.mock import MagicMock
import numpy as np

from processing.domainService import DomainService


# =============================================================================
# Helper
# =============================================================================

def make_ising(num_states=2, final_image=None, original_image=None, mask=None):
    """
    Mock de Ising con dos estados y dos dominios desconectados por estado.

    Layout 6x6 (bloques separados por 2 filas/columnas para evitar 8-conectividad):
      Estado 0 → bloque sup-izq (0:2, 0:2) y bloque inf-der (4:6, 4:6)
      Estado 1 → resto

    Con 8-conectividad los dos bloques de estado 0 están a 3 pasos de distancia
    diagonal, por lo que son 2 dominios distintos.
    """
    if final_image is None:
        final_image = np.array([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ], dtype=np.int32)

    if original_image is None:
        original_image = np.array([
            [ 50,  50, 200, 200, 200, 200],
            [ 50,  50, 200, 200, 200, 200],
            [200, 200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200, 200],
            [200, 200, 200, 200,  50,  50],
            [200, 200, 200, 200,  50,  50],
        ], dtype=np.uint8)

    if mask is None:
        mask = np.ones((6, 6), dtype=bool)

    ising = MagicMock()
    ising.num_states = num_states
    ising.final_image = final_image
    ising.original_image = original_image
    ising.mask = mask
    return ising


# =============================================================================
# 1. extract_state_images
# =============================================================================
class TestExtractStateImages(unittest.TestCase):

    def setUp(self):
        self.ising = make_ising()
        self.service = DomainService(self.ising)

    def test_devuelve_dict_con_todos_los_estados(self):
        for state in range(self.ising.num_states):
            self.assertIn(state, self.service.binary_images)

    def test_dtype_uint8(self):
        for binary in self.service.binary_images.values():
            self.assertEqual(binary.dtype, np.uint8)

    def test_valores_solo_0_y_1(self):
        for binary in self.service.binary_images.values():
            self.assertTrue(set(np.unique(binary)).issubset({0, 1}))

    def test_pixeles_del_estado_son_1(self):
        binary_0 = self.service.binary_images[0]
        expected = ((self.ising.final_image == 0) & self.ising.mask).astype(np.uint8)
        np.testing.assert_array_equal(binary_0, expected)

    def test_mascara_excluye_pixeles(self):
        mask = np.ones((6, 6), dtype=bool)
        mask[0, 0] = False
        ising = make_ising(mask=mask)
        service = DomainService(ising)
        self.assertEqual(service.binary_images[0][0, 0], 0)


# =============================================================================
# 2. label_domains
# =============================================================================
class TestLabelDomains(unittest.TestCase):

    def setUp(self):
        self.ising = make_ising()
        self.service = DomainService(self.ising)

    def test_devuelve_dict_con_todos_los_estados(self):
        for state in range(self.ising.num_states):
            self.assertIn(state, self.service.labeled_images)

    def test_misma_forma_que_imagen(self):
        for labeled in self.service.labeled_images.values():
            self.assertEqual(labeled.shape, self.ising.final_image.shape)

    def test_etiquetas_son_enteros_no_negativos(self):
        for labeled in self.service.labeled_images.values():
            self.assertTrue((labeled >= 0).all())

    def test_num_dominios_correcto(self):
        # Estado 0 tiene 2 regiones desconectadas en el layout del helper
        self.assertEqual(self.service.labeled_images[0].max(), 2)


# =============================================================================
# 3. color_domains
# =============================================================================
class TestColorDomains(unittest.TestCase):

    def setUp(self):
        self.ising = make_ising()
        self.service = DomainService(self.ising)

    def test_devuelve_dict_con_todos_los_estados(self):
        for state in range(self.ising.num_states):
            self.assertIn(state, self.service.colored_images)

    def test_imagen_rgb_tiene_3_canales(self):
        for rgb in self.service.colored_images.values():
            self.assertEqual(rgb.shape[2], 3)

    def test_imagen_rgb_misma_forma_espacial(self):
        h, w = self.ising.final_image.shape
        for rgb in self.service.colored_images.values():
            self.assertEqual(rgb.shape[:2], (h, w))

    def test_fondo_es_negro(self):
        for state, rgb in self.service.colored_images.items():
            background = rgb[self.service.binary_images[state] == 0]
            np.testing.assert_array_equal(background, np.zeros_like(background))

    def test_misma_semilla_da_mismo_resultado(self):
        s1 = DomainService(self.ising, seed=42)
        s2 = DomainService(self.ising, seed=42)
        for state in range(self.ising.num_states):
            np.testing.assert_array_equal(
                s1.colored_images[state], s2.colored_images[state]
            )

    def test_diferente_semilla_da_diferente_resultado(self):
        s1 = DomainService(self.ising, seed=42)
        s2 = DomainService(self.ising, seed=99)
        differs = any(
            not np.array_equal(s1.colored_images[s], s2.colored_images[s])
            for s in range(self.ising.num_states)
        )
        self.assertTrue(differs)


# =============================================================================
# 4. extract_domain_data
# =============================================================================
class TestExtractDomainData(unittest.TestCase):

    def setUp(self):
        self.ising = make_ising()
        self.service = DomainService(self.ising)

    def test_devuelve_dict_con_todos_los_estados(self):
        for state in range(self.ising.num_states):
            self.assertIn(state, self.service.domain_data)

    def test_cada_dominio_tiene_coords_y_values(self):
        for domains in self.service.domain_data.values():
            for data in domains.values():
                self.assertIn("coords", data)
                self.assertIn("values", data)

    def test_coords_son_array_2d_con_dos_columnas(self):
        for domains in self.service.domain_data.values():
            for data in domains.values():
                self.assertEqual(data["coords"].ndim, 2)
                self.assertEqual(data["coords"].shape[1], 2)

    def test_values_corresponden_a_imagen_original(self):
        for domains in self.service.domain_data.values():
            for data in domains.values():
                for coord in data["coords"]:
                    expected = self.ising.original_image[coord[0], coord[1]]
                    self.assertIn(expected, data["values"])


# =============================================================================
# 5. get_data
# =============================================================================
class TestGetData(unittest.TestCase):

    def setUp(self):
        self.ising = make_ising()
        self.service = DomainService(self.ising)
        self.data = self.service.get_data()

    def test_claves_principales_presentes(self):
        for key in ["original", "binary_images", "colored_images",
                    "num_domains", "domain_data"]:
            self.assertIn(key, self.data)

    def test_original_es_la_imagen_del_ising(self):
        np.testing.assert_array_equal(self.data["original"], self.ising.original_image)

    def test_num_domains_es_entero_por_estado(self):
        for n in self.data["num_domains"].values():
            self.assertIsInstance(n, int)

    def test_num_domains_coincide_con_labeled(self):
        for state, n in self.data["num_domains"].items():
            self.assertEqual(n, int(self.service.labeled_images[state].max()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
