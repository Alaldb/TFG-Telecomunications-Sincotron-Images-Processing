import unittest
import numpy as np
from unittest.mock import MagicMock

from processing.isingPlotService import IsingPlotService


# =============================================================================
# Helper
# =============================================================================

def make_ising(
    original_image=None,
    final_image=None,
    mask=None,
    parameters=None,
    num_states=3,
):
    """
    Mock de Ising con layout 4x4:
      - Estado 0 (media 30)  → Black → Rojo   (255,0,0)
      - Estado 1 (media 128) → Gray  → Verde  (0,255,0)
      - Estado 2 (media 220) → White → Azul   (0,0,255)
    """
    if original_image is None:
        original_image = np.array([
            [ 30,  30, 128, 128],
            [ 30,  30, 128, 128],
            [220, 220,  30,  30],
            [220, 220,  30,  30],
        ], dtype=np.uint8)

    if final_image is None:
        final_image = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ], dtype=np.int32)

    if mask is None:
        mask = np.ones((4, 4), dtype=bool)

    if parameters is None:
        parameters = {
            0: {"mean":  30.0, "std": 5.0 + 1e-6},
            1: {"mean": 128.0, "std": 8.0 + 1e-6},
            2: {"mean": 220.0, "std": 6.0 + 1e-6},
        }

    ising = MagicMock()
    ising.original_image = original_image
    ising.final_image = final_image
    ising.mask = mask
    ising.parameters = parameters
    ising.num_states = num_states
    return ising


# =============================================================================
# 1. Estructura del diccionario devuelto
# =============================================================================
class TestGetPlotDataStructure(unittest.TestCase):

    def setUp(self):
        self.service = IsingPlotService()
        self.ising = make_ising()
        self.data = self.service.get_plot_data(self.ising)

    def test_claves_principales_presentes(self):
        self.assertIn("histogram", self.data)
        self.assertIn("images", self.data)

    def test_histogram_tiene_todas_las_subclaves(self):
        h = self.data["histogram"]
        self.assertIn("pixel_values",    h)
        self.assertIn("state_per_pixel", h)
        self.assertIn("state_stats",     h)

    def test_images_tiene_todas_las_subclaves(self):
        img = self.data["images"]
        self.assertIn("original",      img)
        self.assertIn("segmented_rgb", img)
        self.assertIn("state_colors",  img)

    def test_state_stats_tiene_entrada_por_cada_estado(self):
        stats = self.data["histogram"]["state_stats"]
        for state in range(self.ising.num_states):
            self.assertIn(state, stats)

    def test_cada_estado_tiene_campos_requeridos(self):
        stats = self.data["histogram"]["state_stats"]
        required = {"label", "percent", "area_px", "mean", "std"}
        for state, info in stats.items():
            self.assertTrue(
                required.issubset(info.keys()),
                msg=f"Estado {state} falta campos: {required - info.keys()}",
            )

    def test_state_colors_tiene_entrada_por_cada_estado(self):
        colors = self.data["images"]["state_colors"]
        for state in range(self.ising.num_states):
            self.assertIn(state, colors)

    def test_cada_state_color_tiene_label_y_rgb(self):
        colors = self.data["images"]["state_colors"]
        for state, (label, rgb) in colors.items():
            self.assertIsInstance(label, str)
            self.assertEqual(len(rgb), 3)


# =============================================================================
# 2. Porcentajes y conteos
# =============================================================================
class TestGetPlotDataStats(unittest.TestCase):

    def setUp(self):
        self.service = IsingPlotService()

    def test_porcentajes_suman_100(self):
        data = self.service.get_plot_data(make_ising())
        total = sum(v["percent"] for v in data["histogram"]["state_stats"].values())
        self.assertAlmostEqual(total, 100.0, places=6)

    def test_areas_suman_total_de_pixeles_en_mascara(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        total_area = sum(v["area_px"] for v in data["histogram"]["state_stats"].values())
        self.assertEqual(total_area, int(ising.mask.sum()))

    def test_porcentajes_con_mascara_parcial(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        data = self.service.get_plot_data(make_ising(mask=mask))
        total = sum(v["percent"] for v in data["histogram"]["state_stats"].values())
        self.assertAlmostEqual(total, 100.0, places=6)

    def test_pixel_values_solo_dentro_de_mascara(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        self.assertEqual(len(data["histogram"]["pixel_values"]), int(ising.mask.sum()))

    def test_state_per_pixel_mismo_tamanio_que_pixel_values(self):
        data = self.service.get_plot_data(make_ising())
        self.assertEqual(
            len(data["histogram"]["pixel_values"]),
            len(data["histogram"]["state_per_pixel"]),
        )

    def test_mean_y_std_provienen_de_ising_parameters(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        stats = data["histogram"]["state_stats"]
        for state in range(ising.num_states):
            self.assertAlmostEqual(stats[state]["mean"], ising.parameters[state]["mean"])
            self.assertAlmostEqual(stats[state]["std"],  ising.parameters[state]["std"])


# =============================================================================
# 3. Mapeo de estados → etiquetas y colores físicos
# =============================================================================
class TestGetPlotDataColorMapping(unittest.TestCase):

    def setUp(self):
        self.service = IsingPlotService()

    def _color_of_label(self, label: str, data: dict) -> tuple:
        for _, (lbl, rgb) in data["images"]["state_colors"].items():
            if lbl == label:
                return rgb
        raise KeyError(f"Label '{label}' no encontrado en state_colors")

    def test_estado_mayor_media_es_white(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        stats = data["histogram"]["state_stats"]
        white_state = max(ising.parameters, key=lambda s: ising.parameters[s]["mean"])
        self.assertEqual(stats[white_state]["label"], "White")

    def test_estado_menor_media_es_black(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        stats = data["histogram"]["state_stats"]
        black_state = min(ising.parameters, key=lambda s: ising.parameters[s]["mean"])
        self.assertEqual(stats[black_state]["label"], "Black")

    def test_estado_intermedio_es_gray(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        stats = data["histogram"]["state_stats"]
        sorted_states = sorted(ising.parameters, key=lambda s: ising.parameters[s]["mean"])
        self.assertEqual(stats[sorted_states[1]]["label"], "Gray")

    def test_white_tiene_color_azul(self):
        data = self.service.get_plot_data(make_ising())
        self.assertEqual(self._color_of_label("White", data), (0, 0, 255))

    def test_gray_tiene_color_verde(self):
        data = self.service.get_plot_data(make_ising())
        self.assertEqual(self._color_of_label("Gray", data), (0, 255, 0))

    def test_black_tiene_color_rojo(self):
        data = self.service.get_plot_data(make_ising())
        self.assertEqual(self._color_of_label("Black", data), (255, 0, 0))

    def test_segmented_rgb_tiene_forma_correcta(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        h, w = ising.original_image.shape
        self.assertEqual(data["images"]["segmented_rgb"].shape, (h, w, 3))

    def test_segmented_rgb_dtype_uint8(self):
        data = self.service.get_plot_data(make_ising())
        self.assertEqual(data["images"]["segmented_rgb"].dtype, np.uint8)

    def test_segmented_rgb_colores_corresponden_a_estados(self):
        ising = make_ising()
        data = self.service.get_plot_data(ising)
        rgb_img = data["images"]["segmented_rgb"]
        colors  = data["images"]["state_colors"]
        for state, (_, rgb) in colors.items():
            px_mask = (ising.final_image == state) & ising.mask
            if px_mask.any():
                expected = np.tile(np.array(rgb, dtype=np.uint8), (px_mask.sum(), 1))
                np.testing.assert_array_equal(rgb_img[px_mask], expected)

    def test_mapeo_robusto_con_orden_kmeans_invertido(self):
        """Si KMeans asigna estado 0 al cluster más brillante el mapeo sigue correcto."""
        parameters = {
            0: {"mean": 220.0, "std": 6.0 + 1e-6},
            1: {"mean": 128.0, "std": 8.0 + 1e-6},
            2: {"mean":  30.0, "std": 5.0 + 1e-6},
        }
        data = self.service.get_plot_data(make_ising(parameters=parameters))
        stats = data["histogram"]["state_stats"]
        self.assertEqual(stats[0]["label"], "White")
        self.assertEqual(stats[2]["label"], "Black")


if __name__ == "__main__":
    unittest.main(verbosity=2)
