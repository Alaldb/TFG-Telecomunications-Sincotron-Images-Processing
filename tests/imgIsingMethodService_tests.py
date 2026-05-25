import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ── Mocks de dependencias externas antes del import ──────────────────────────
cv2_mock = MagicMock()
cv2_mock.IMREAD_GRAYSCALE = 0
cv2_mock.NORM_MINMAX = 32
cv2_mock.HOUGH_GRADIENT = 3

sklearn_mock = MagicMock()

sys.modules["cv2"] = cv2_mock
sys.modules["sklearn"] = sklearn_mock
sys.modules["sklearn.cluster"] = sklearn_mock.cluster

from processing.isingMethodService import Ising  # noqa: E402


def make_ising(
    num_states=3,
    original_image=None,
    mask=None,
    iterative_matrix=None,
    parameters=None,
    beta=2,
    max_iterations=20,
):
    """Construye un objeto Ising sin ejecutar __init__ ni run."""
    obj = object.__new__(Ising)
    obj.num_states = num_states
    obj.beta = beta
    obj.max_iterations = max_iterations
    obj.original_image = (
        original_image if original_image is not None
        else np.zeros((10, 10), dtype=np.uint8)
    )
    obj.mask = (
        mask if mask is not None
        else np.ones((10, 10), dtype=bool)
    )
    obj.iterative_matrix = (
        iterative_matrix if iterative_matrix is not None
        else np.zeros((10, 10), dtype=np.int32)
    )
    obj.parameters = parameters or {
        0: {"mean": 50.0,  "std": 10.0 + 1e-6},
        1: {"mean": 150.0, "std": 10.0 + 1e-6},
        2: {"mean": 200.0, "std": 10.0 + 1e-6},
    }
    return obj


# =============================================================================
# 1. __init__ — solo guarda parámetros, no ejecuta nada
# =============================================================================
class TestInit(unittest.TestCase):
    def test_guarda_beta(self):
        ising = Ising(beta=5)
        self.assertEqual(ising.beta, 5)

    def test_guarda_max_iterations(self):
        ising = Ising(max_iterations=50)
        self.assertEqual(ising.max_iterations, 50)

    def test_guarda_num_states(self):
        ising = Ising(num_states=2)
        self.assertEqual(ising.num_states, 2)

    def test_no_carga_imagen_en_init(self):
        """__init__ no debe asignar original_image."""
        ising = Ising()
        self.assertFalse(hasattr(ising, "original_image"))

    def test_valores_por_defecto(self):
        ising = Ising()
        self.assertEqual(ising.beta, 2)
        self.assertEqual(ising.max_iterations, 20)
        self.assertEqual(ising.num_states, 3)


# =============================================================================
# 2. run — puebla los atributos del objeto
# =============================================================================
class TestRun(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((10, 10), dtype=np.uint8)
        self.image[:5, :] = 50
        self.image[5:, :] = 200
        self.mask = np.ones((10, 10), dtype=bool)
        self.imatrix = np.zeros((10, 10), dtype=np.int32)
        self.params = {
            0: {"mean": 50.0,  "std": 1.0 + 1e-6},
            1: {"mean": 200.0, "std": 1.0 + 1e-6},
        }
        self.final = np.ones((10, 10), dtype=np.int32)

    def _patched_run(self, ising):
        with patch.object(ising, "create_mask", return_value=self.mask), \
             patch.object(ising, "initialize_ising_model",
                          return_value=(self.imatrix, self.params)), \
             patch.object(ising, "apply_ising_model_icm",
                          return_value=self.final):
            ising.run(self.image)
        return ising

    def test_asigna_original_image(self):
        ising = self._patched_run(Ising(num_states=2))
        np.testing.assert_array_equal(ising.original_image, self.image)

    def test_asigna_mask(self):
        ising = self._patched_run(Ising(num_states=2))
        np.testing.assert_array_equal(ising.mask, self.mask)

    def test_asigna_final_image(self):
        ising = self._patched_run(Ising(num_states=2))
        np.testing.assert_array_equal(ising.final_image, self.final)

    def test_llama_create_mask(self):
        ising = Ising(num_states=2)
        with patch.object(ising, "create_mask", return_value=self.mask) as mock_mask, \
             patch.object(ising, "initialize_ising_model",
                          return_value=(self.imatrix, self.params)), \
             patch.object(ising, "apply_ising_model_icm", return_value=self.final):
            ising.run(self.image)
        mock_mask.assert_called_once()

    def test_llama_icm_con_beta_y_max_iterations(self):
        ising = Ising(beta=7, max_iterations=15, num_states=2)
        with patch.object(ising, "create_mask", return_value=self.mask), \
             patch.object(ising, "initialize_ising_model",
                          return_value=(self.imatrix, self.params)), \
             patch.object(ising, "apply_ising_model_icm",
                          return_value=self.final) as mock_icm:
            ising.run(self.image)
        mock_icm.assert_called_once_with(7, 15)


# =============================================================================
# 3. search_circles
# =============================================================================
class TestSearchCircles(unittest.TestCase):
    def setUp(self):
        image = np.zeros((200, 200), dtype=np.uint8)
        self.ising = make_ising(original_image=image)
        cv2_mock.normalize.return_value = image
        cv2_mock.medianBlur.return_value = image

    def test_devuelve_lista_de_tuplas_cuando_hay_circulos(self):
        cv2_mock.HoughCircles.return_value = np.array([[[100.0, 100.0, 80.0]]])
        result = self.ising.search_circles()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 3)

    def test_valores_del_circulo_son_redondeados(self):
        cv2_mock.HoughCircles.return_value = np.array([[[99.6, 100.4, 80.6]]])
        result = self.ising.search_circles()
        x, y, r = result[0]
        self.assertEqual(x, 100)
        self.assertEqual(y, 100)
        self.assertEqual(r, 81)

    def test_devuelve_none_cuando_no_hay_circulos(self):
        cv2_mock.HoughCircles.return_value = None
        result = self.ising.search_circles()
        self.assertIsNone(result)

    def test_usa_altura_como_min_dist(self):
        cv2_mock.HoughCircles.return_value = None
        self.ising.search_circles()
        args, kwargs = cv2_mock.HoughCircles.call_args
        min_dist = kwargs.get("minDist", args[3] if len(args) > 3 else None)
        self.assertEqual(min_dist, 200)


# =============================================================================
# 4. create_mask
# =============================================================================
class TestCreateMask(unittest.TestCase):
    def setUp(self):
        self.ising = make_ising(original_image=np.zeros((100, 100), dtype=np.uint8))

    def test_mascara_completa_cuando_no_hay_circulos(self):
        with patch.object(self.ising, "search_circles", return_value=None):
            result = self.ising.create_mask()
        self.assertTrue(result.all())
        self.assertEqual(result.dtype, bool)

    def test_forma_igual_a_imagen(self):
        with patch.object(self.ising, "search_circles", return_value=None):
            result = self.ising.create_mask()
        self.assertEqual(result.shape, self.ising.original_image.shape)

    def test_tipo_bool_cuando_hay_circulos(self):
        with patch.object(self.ising, "search_circles", return_value=[(50, 50, 30)]):
            result = self.ising.create_mask()
        self.assertEqual(result.dtype, bool)

    def test_llama_a_cv2_circle_por_cada_circulo(self):
        circles = [(50, 50, 30), (80, 80, 20)]
        cv2_mock.circle.reset_mock()
        with patch.object(self.ising, "search_circles", return_value=circles):
            self.ising.create_mask()
        self.assertEqual(cv2_mock.circle.call_count, len(circles))


# =============================================================================
# 5. calculate_statistical_variables
# =============================================================================
class TestCalculateStatisticalVariables(unittest.TestCase):
    def setUp(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[0:4, :] = 50
        image[4:7, :] = 150
        image[7:10, :] = 200

        imatrix = np.zeros((10, 10), dtype=np.int32)
        imatrix[0:4, :] = 0
        imatrix[4:7, :] = 1
        imatrix[7:10, :] = 2

        self.ising = make_ising(
            original_image=image,
            iterative_matrix=imatrix,
            mask=np.ones((10, 10), dtype=bool),
            num_states=3,
        )

    def test_devuelve_dict_con_todos_los_estados(self):
        result = self.ising.calculate_statistical_variables()
        for state in range(self.ising.num_states):
            self.assertIn(state, result)

    def test_cada_estado_tiene_mean_y_std(self):
        result = self.ising.calculate_statistical_variables()
        for state in range(self.ising.num_states):
            self.assertIn("mean", result[state])
            self.assertIn("std", result[state])

    def test_media_correcta_por_estado(self):
        result = self.ising.calculate_statistical_variables()
        self.assertAlmostEqual(result[0]["mean"], 50.0)
        self.assertAlmostEqual(result[1]["mean"], 150.0)
        self.assertAlmostEqual(result[2]["mean"], 200.0)

    def test_std_siempre_positivo(self):
        result = self.ising.calculate_statistical_variables()
        for state in range(self.ising.num_states):
            self.assertGreater(result[state]["std"], 0)

    def test_estado_vacio_recibe_valores_por_defecto(self):
        self.ising.iterative_matrix = np.zeros((10, 10), dtype=np.int32)
        result = self.ising.calculate_statistical_variables()
        self.assertEqual(result[1]["mean"], 0)
        self.assertEqual(result[1]["std"], 1e-6)


# =============================================================================
# 6. initialize_ising_model
# =============================================================================
class TestInitializeIsingModel(unittest.TestCase):
    def setUp(self):
        image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        self.ising = make_ising(original_image=image, mask=mask, num_states=3)

        self.kmeans_mock = MagicMock()
        self.kmeans_mock.fit_predict.return_value = np.zeros(400, dtype=np.int32)
        sklearn_mock.cluster.KMeans.return_value = self.kmeans_mock

    def test_devuelve_tupla_matriz_y_parametros(self):
        result = self.ising.initialize_ising_model()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_matriz_tiene_forma_correcta(self):
        matrix, _ = self.ising.initialize_ising_model()
        self.assertEqual(matrix.shape, self.ising.original_image.shape)

    def test_pixeles_fuera_de_mascara_son_menos_uno(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        self.ising.mask = mask
        self.kmeans_mock.fit_predict.return_value = np.ones(100, dtype=np.int32)
        matrix, _ = self.ising.initialize_ising_model()
        self.assertTrue((matrix[~mask] == -1).all())

    def test_kmeans_se_llama_con_num_states(self):
        self.ising.initialize_ising_model()
        sklearn_mock.cluster.KMeans.assert_called_with(
            n_clusters=self.ising.num_states, n_init=10, random_state=64
        )


# =============================================================================
# 7. calculate_neighbour_sum
# =============================================================================
class TestCalculateNeighbourSum(unittest.TestCase):
    def setUp(self):
        imatrix = np.zeros((5, 5), dtype=np.int32)
        imatrix[2, 2] = 1
        self.ising = make_ising(iterative_matrix=imatrix)

    def test_devuelve_array_con_forma_correcta(self):
        result = self.ising.calculate_neighbour_sum(state=1)
        self.assertEqual(result.shape, self.ising.iterative_matrix.shape)

    def test_pixel_central_tiene_cero_vecinos_del_mismo_estado(self):
        result = self.ising.calculate_neighbour_sum(state=1)
        self.assertEqual(result[2, 2], 0)

    def test_vecinos_directos_cuentan_uno(self):
        result = self.ising.calculate_neighbour_sum(state=1)
        self.assertEqual(result[1, 2], 1)
        self.assertEqual(result[3, 2], 1)
        self.assertEqual(result[2, 1], 1)
        self.assertEqual(result[2, 3], 1)

    def test_diagonales_no_cuentan(self):
        result = self.ising.calculate_neighbour_sum(state=1)
        self.assertEqual(result[1, 1], 0)
        self.assertEqual(result[1, 3], 0)
        self.assertEqual(result[3, 1], 0)
        self.assertEqual(result[3, 3], 0)

    def test_estado_ausente_devuelve_ceros(self):
        result = self.ising.calculate_neighbour_sum(state=2)
        np.testing.assert_array_equal(result, np.zeros((5, 5)))


# =============================================================================
# 8. apply_ising_model_icm
# =============================================================================
class TestApplyIsingModelICM(unittest.TestCase):
    def setUp(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:5, :] = 50
        image[5:, :] = 200

        imatrix = np.zeros((10, 10), dtype=np.int32)
        imatrix[:5, :] = 0
        imatrix[5:, :] = 1

        self.ising = make_ising(
            original_image=image,
            iterative_matrix=imatrix,
            mask=np.ones((10, 10), dtype=bool),
            num_states=2,
            parameters={
                0: {"mean": 50.0,  "std": 10.0 + 1e-6},
                1: {"mean": 200.0, "std": 10.0 + 1e-6},
            },
        )

    def test_devuelve_matriz_con_forma_correcta(self):
        result = self.ising.apply_ising_model_icm(beta=2, max_iterations=5)
        self.assertEqual(result.shape, self.ising.original_image.shape)

    def test_devuelve_matriz_de_enteros(self):
        result = self.ising.apply_ising_model_icm(beta=2, max_iterations=5)
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_pixeles_fuera_de_mascara_no_cambian(self):
        mask = np.ones((10, 10), dtype=bool)
        mask[0, 0] = False
        self.ising.mask = mask
        self.ising.iterative_matrix[0, 0] = -1
        result = self.ising.apply_ising_model_icm(beta=2, max_iterations=5)
        self.assertEqual(result[0, 0], -1)

    def test_una_sola_iteracion_no_falla(self):
        try:
            result = self.ising.apply_ising_model_icm(beta=2, max_iterations=1)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Lanzó excepción inesperada: {e}")

    def test_estados_fuera_del_rango_no_aparecen(self):
        result = self.ising.apply_ising_model_icm(beta=2, max_iterations=5)
        valid = (result == -1) | ((result >= 0) & (result < self.ising.num_states))
        self.assertTrue(valid.all())

    def test_converge_con_entrada_ya_optima(self):
        with patch("builtins.print") as mock_print:
            self.ising.apply_ising_model_icm(beta=0, max_iterations=20)
        printed = any("Converged" in str(call) for call in mock_print.call_args_list)
        self.assertTrue(printed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
