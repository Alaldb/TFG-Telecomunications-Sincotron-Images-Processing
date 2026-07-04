import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ── Mocks de dependencias externas antes del import ──────────────────────────
cv2_mock = MagicMock()
cv2_mock.NORM_MINMAX = 32
cv2_mock.HOUGH_GRADIENT = 3

sklearn_mock = MagicMock()
gco_mock = MagicMock()
tifffile_mock = MagicMock()

sys.modules["cv2"] = cv2_mock
sys.modules["sklearn"] = sklearn_mock
sys.modules["sklearn.cluster"] = sklearn_mock.cluster
sys.modules["sklearn.mixture"] = sklearn_mock.mixture
sys.modules["gco"] = gco_mock
sys.modules["tifffile"] = tifffile_mock

from processing.graphCutsService import GraphCutsService  # noqa: E402
from core.segmentationContainer import SegmentationMethod  # noqa: E402


def make_service(
    num_states=3,
    lambda_value=1.0,
    sigma=None,
    num_iterations=-1,
    number_gaussians_per_state=3,
    original_image=None,
    mask=None,
    initial_labels=None,
    final_image=None,
    parameters=None,
):
    """Construye un GraphCutsService sin ejecutar __init__ ni run."""
    obj = object.__new__(GraphCutsService)
    obj.num_states = num_states
    obj.lambda_value = lambda_value
    obj.sigma = sigma
    obj.num_iterations = num_iterations
    obj.number_gaussians_per_state = number_gaussians_per_state
    if original_image is not None:
        obj.original_image = original_image
    if mask is not None:
        obj.mask = mask
    if initial_labels is not None:
        obj.initial_labels = initial_labels
    if final_image is not None:
        obj.final_image = final_image
    if parameters is not None:
        obj.parameters = parameters
    return obj


# =============================================================================
# 1. __init__ — solo guarda parámetros, no ejecuta nada
# =============================================================================
class TestInit(unittest.TestCase):
    def test_valores_por_defecto(self):
        gc = GraphCutsService()
        self.assertEqual(gc.num_states, 3)
        self.assertEqual(gc.lambda_value, 1.0)
        self.assertIsNone(gc.sigma)
        self.assertEqual(gc.num_iterations, -1)
        self.assertEqual(gc.number_gaussians_per_state, 3)

    def test_guarda_num_states(self):
        gc = GraphCutsService(num_states=5)
        self.assertEqual(gc.num_states, 5)

    def test_guarda_lambda_value(self):
        gc = GraphCutsService(lambda_value=0.05)
        self.assertEqual(gc.lambda_value, 0.05)

    def test_guarda_sigma(self):
        gc = GraphCutsService(sigma=12.5)
        self.assertEqual(gc.sigma, 12.5)

    def test_guarda_num_iterations(self):
        gc = GraphCutsService(num_iterations=10)
        self.assertEqual(gc.num_iterations, 10)

    def test_guarda_number_gaussians_per_state(self):
        gc = GraphCutsService(number_gaussians_per_state=7)
        self.assertEqual(gc.number_gaussians_per_state, 7)

    def test_no_carga_imagen_en_init(self):
        """__init__ no debe asignar original_image."""
        gc = GraphCutsService()
        self.assertFalse(hasattr(gc, "original_image"))


# =============================================================================
# 2. run — puebla los atributos del objeto
# =============================================================================
class TestRun(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((10, 10), dtype=np.uint8)
        self.mask = np.ones((10, 10), dtype=bool)
        self.labels = np.zeros((10, 10), dtype=np.int32)
        self.final = np.ones((10, 10), dtype=np.int32)
        self.params = {0: {"mean": 100.0, "std": 1.0}}

    def _patched_run(self, gc):
        with patch.object(gc, "create_mask", return_value=self.mask), \
             patch.object(gc, "initialize_model", return_value=self.labels), \
             patch.object(gc, "apply_graph_cuts", return_value=self.final), \
             patch.object(gc, "calculate_parameters", return_value=self.params):
            gc.run(self.image)
        return gc

    def test_asigna_original_image(self):
        gc = self._patched_run(GraphCutsService())
        np.testing.assert_array_equal(gc.original_image, self.image)

    def test_asigna_mask(self):
        gc = self._patched_run(GraphCutsService())
        np.testing.assert_array_equal(gc.mask, self.mask)

    def test_asigna_initial_labels(self):
        gc = self._patched_run(GraphCutsService())
        np.testing.assert_array_equal(gc.initial_labels, self.labels)

    def test_asigna_final_image(self):
        gc = self._patched_run(GraphCutsService())
        np.testing.assert_array_equal(gc.final_image, self.final)

    def test_asigna_parameters(self):
        gc = self._patched_run(GraphCutsService())
        self.assertEqual(gc.parameters, self.params)

    def test_llama_create_mask(self):
        gc = GraphCutsService()
        with patch.object(gc, "create_mask", return_value=self.mask) as mock_mask, \
             patch.object(gc, "initialize_model", return_value=self.labels), \
             patch.object(gc, "apply_graph_cuts", return_value=self.final), \
             patch.object(gc, "calculate_parameters", return_value=self.params):
            gc.run(self.image)
        mock_mask.assert_called_once()


# =============================================================================
# 3. search_circles
# =============================================================================
class TestSearchCircles(unittest.TestCase):
    def setUp(self):
        image = np.zeros((200, 200), dtype=np.uint8)
        self.gc = make_service(original_image=image)
        cv2_mock.normalize.return_value = image
        cv2_mock.medianBlur.return_value = image

    def test_devuelve_lista_de_tuplas_cuando_hay_circulos(self):
        cv2_mock.HoughCircles.return_value = np.array([[[100.0, 100.0, 80.0]]])
        result = self.gc.search_circles()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 3)

    def test_valores_del_circulo_son_redondeados(self):
        cv2_mock.HoughCircles.return_value = np.array([[[99.6, 100.4, 80.6]]])
        result = self.gc.search_circles()
        x, y, r = result[0]
        self.assertEqual(x, 100)
        self.assertEqual(y, 100)
        self.assertEqual(r, 81)

    def test_devuelve_none_cuando_no_hay_circulos(self):
        cv2_mock.HoughCircles.return_value = None
        result = self.gc.search_circles()
        self.assertIsNone(result)

    def test_usa_altura_como_min_dist(self):
        cv2_mock.HoughCircles.return_value = None
        self.gc.search_circles()
        args, kwargs = cv2_mock.HoughCircles.call_args
        min_dist = kwargs.get("minDist", args[3] if len(args) > 3 else None)
        self.assertEqual(min_dist, 200)


# =============================================================================
# 4. create_mask
# =============================================================================
class TestCreateMask(unittest.TestCase):
    def setUp(self):
        self.gc = make_service(original_image=np.zeros((100, 100), dtype=np.uint8))

    def test_mascara_completa_cuando_no_hay_circulos(self):
        with patch.object(self.gc, "search_circles", return_value=None):
            result = self.gc.create_mask()
        self.assertTrue(result.all())
        self.assertEqual(result.dtype, bool)

    def test_forma_igual_a_imagen(self):
        with patch.object(self.gc, "search_circles", return_value=None):
            result = self.gc.create_mask()
        self.assertEqual(result.shape, self.gc.original_image.shape)

    def test_tipo_bool_cuando_hay_circulos(self):
        with patch.object(self.gc, "search_circles", return_value=[(50, 50, 30)]):
            result = self.gc.create_mask()
        self.assertEqual(result.dtype, bool)

    def test_llama_a_cv2_circle_por_cada_circulo(self):
        circles = [(50, 50, 30), (80, 80, 20)]
        cv2_mock.circle.reset_mock()
        with patch.object(self.gc, "search_circles", return_value=circles):
            self.gc.create_mask()
        self.assertEqual(cv2_mock.circle.call_count, len(circles))


# =============================================================================
# 5. initialize_model
# =============================================================================
class TestInitializeModel(unittest.TestCase):
    def setUp(self):
        image = np.random.randint(0, 256, (20, 20)).astype(np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        self.gc = make_service(original_image=image, mask=mask, num_states=3)

        self.kmeans_mock = MagicMock()
        self.kmeans_mock.fit_predict.return_value = np.zeros(400, dtype=np.int32)
        sklearn_mock.cluster.KMeans.return_value = self.kmeans_mock

    def test_forma_correcta(self):
        result = self.gc.initialize_model()
        self.assertEqual(result.shape, self.gc.original_image.shape)

    def test_pixeles_fuera_de_mascara_son_menos_uno(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        self.gc.mask = mask
        self.kmeans_mock.fit_predict.return_value = np.ones(100, dtype=np.int32)
        result = self.gc.initialize_model()
        self.assertTrue((result[~mask] == -1).all())

    def test_kmeans_se_llama_con_num_states(self):
        self.gc.initialize_model()
        sklearn_mock.cluster.KMeans.assert_called_with(
            n_clusters=self.gc.num_states, n_init=10, random_state=64
        )


# =============================================================================
# 6. create_gaussian_mixtures_for_states
# =============================================================================
class TestCreateGaussianMixturesForStates(unittest.TestCase):
    def setUp(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:5, :] = 50
        image[5:, :] = 200

        self.labels = np.zeros((10, 10), dtype=np.int32)
        self.labels[5:, :] = 1

        self.mask = np.ones((10, 10), dtype=bool)

        self.gc = make_service(
            original_image=image, mask=self.mask, num_states=2,
            number_gaussians_per_state=3,
        )

        sklearn_mock.mixture.GaussianMixture.reset_mock()
        self.gm_instance = MagicMock()
        sklearn_mock.mixture.GaussianMixture.return_value = self.gm_instance

    def test_devuelve_entrada_para_cada_estado(self):
        result = self.gc.create_gaussian_mixtures_for_states(self.labels)
        for state in range(self.gc.num_states):
            self.assertIn(state, result)

    def test_llama_fit_para_cada_estado(self):
        self.gc.create_gaussian_mixtures_for_states(self.labels)
        self.assertEqual(self.gm_instance.fit.call_count, self.gc.num_states)

    def test_k_limitado_por_numero_de_pixeles(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, :2] = True  # solo 2 píxeles en estado 0
        gc = make_service(
            original_image=np.zeros((10, 10), dtype=np.uint8),
            mask=mask, num_states=1, number_gaussians_per_state=5,
        )
        labels = np.zeros((10, 10), dtype=np.int32)
        gc.create_gaussian_mixtures_for_states(labels)
        _, kwargs = sklearn_mock.mixture.GaussianMixture.call_args
        self.assertLessEqual(kwargs["n_components"], 2)

    def test_k_limitado_por_valores_unicos(self):
        image = np.full((10, 10), 5, dtype=np.uint8)  # un único valor de píxel
        mask = np.ones((10, 10), dtype=bool)
        labels = np.zeros((10, 10), dtype=np.int32)
        gc = make_service(
            original_image=image, mask=mask, num_states=1,
            number_gaussians_per_state=5,
        )
        gc.create_gaussian_mixtures_for_states(labels)
        _, kwargs = sklearn_mock.mixture.GaussianMixture.call_args
        self.assertEqual(kwargs["n_components"], 1)


# =============================================================================
# 7. compute_data_cost
# =============================================================================
class TestComputeDataCost(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((2, 2), dtype=np.uint8)
        self.mask = np.ones((2, 2), dtype=bool)
        self.gc = make_service(original_image=self.image, mask=self.mask, num_states=2)

        self.gm0 = MagicMock()
        self.gm1 = MagicMock()
        self.gm0.score_samples.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        self.gm1.score_samples.return_value = np.array([4.0, 3.0, 2.0, 1.0])
        self.gaussian_mixtures = {0: self.gm0, 1: self.gm1}

    def test_forma_de_salida(self):
        result = self.gc.compute_data_cost(self.gaussian_mixtures)
        self.assertEqual(result.shape, (2, 2, 2))

    def test_llama_score_samples_para_cada_estado(self):
        self.gc.compute_data_cost(self.gaussian_mixtures)
        self.gm0.score_samples.assert_called_once()
        self.gm1.score_samples.assert_called_once()

    def test_minimo_dentro_de_mascara_es_cero(self):
        result = self.gc.compute_data_cost(self.gaussian_mixtures)
        self.assertAlmostEqual(result[self.mask].min(), 0.0, places=6)

    def test_sin_valores_nan(self):
        result = self.gc.compute_data_cost(self.gaussian_mixtures)
        self.assertFalse(np.isnan(result).any())


# =============================================================================
# 8. estimate_sigma
# =============================================================================
class TestEstimateSigma(unittest.TestCase):
    def test_imagen_uniforme_devuelve_valor_casi_cero(self):
        image = np.full((10, 10), 100, dtype=np.uint8)
        gc = make_service(original_image=image)
        sigma = gc.estimate_sigma()
        self.assertAlmostEqual(sigma, 1e-6, places=8)

    def test_imagen_con_variacion_devuelve_sigma_mayor(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, ::2] = 200
        gc = make_service(original_image=image)
        sigma = gc.estimate_sigma()
        self.assertGreater(sigma, 1.0)

    def test_siempre_devuelve_valor_positivo(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        gc = make_service(original_image=image)
        self.assertGreater(gc.estimate_sigma(), 0)


# =============================================================================
# 9. compute_nlink_weights
# =============================================================================
class TestComputeNlinkWeights(unittest.TestCase):
    def test_forma_de_los_pesos(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        gc = make_service(original_image=image, mask=mask, sigma=5.0, lambda_value=1.0)
        v, h = gc.compute_nlink_weights()
        self.assertEqual(v.shape, (3, 4))
        self.assertEqual(h.shape, (4, 3))

    def test_pixeles_iguales_dan_peso_maximo_lambda(self):
        image = np.full((4, 4), 50, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        gc = make_service(original_image=image, mask=mask, sigma=5.0, lambda_value=2.0)
        v, h = gc.compute_nlink_weights()
        self.assertTrue(np.allclose(v, 2.0))
        self.assertTrue(np.allclose(h, 2.0))

    def test_mascara_desconecta_pesos(self):
        image = np.full((4, 4), 50, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        mask[0, 0] = False
        gc = make_service(original_image=image, mask=mask, sigma=5.0, lambda_value=2.0)
        v, h = gc.compute_nlink_weights()
        self.assertEqual(v[0, 0], 0.0)
        self.assertEqual(h[0, 0], 0.0)

    def test_usa_sigma_estimado_si_no_se_especifica(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        gc = make_service(original_image=image, mask=mask, sigma=None, lambda_value=1.0)
        with patch.object(gc, "estimate_sigma", return_value=10.0) as mock_estimate:
            gc.compute_nlink_weights()
        mock_estimate.assert_called_once()


# =============================================================================
# 10. apply_graph_cuts
# =============================================================================
class TestApplyGraphCuts(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((4, 4), dtype=np.uint8)
        self.mask = np.ones((4, 4), dtype=bool)
        self.mask[0, 0] = False
        self.initial_labels = np.zeros((4, 4), dtype=np.int32)
        self.gc = make_service(
            original_image=self.image, mask=self.mask, num_states=2,
            initial_labels=self.initial_labels,
        )

    def test_forma_y_tipo_de_salida(self):
        gco_mock.cut_grid_graph.return_value = np.zeros(16, dtype=np.int32)
        with patch.object(self.gc, "create_gaussian_mixtures_for_states", return_value={}), \
             patch.object(self.gc, "compute_data_cost",
                          return_value=np.zeros((4, 4, 2))), \
             patch.object(self.gc, "compute_nlink_weights",
                          return_value=(np.zeros((3, 4)), np.zeros((4, 3)))):
            result = self.gc.apply_graph_cuts()
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_pixeles_fuera_de_mascara_son_menos_uno(self):
        gco_mock.cut_grid_graph.return_value = np.ones(16, dtype=np.int32)
        with patch.object(self.gc, "create_gaussian_mixtures_for_states", return_value={}), \
             patch.object(self.gc, "compute_data_cost",
                          return_value=np.zeros((4, 4, 2))), \
             patch.object(self.gc, "compute_nlink_weights",
                          return_value=(np.zeros((3, 4)), np.zeros((4, 3)))):
            result = self.gc.apply_graph_cuts()
        self.assertEqual(result[0, 0], -1)

    def test_llama_cut_grid_graph_con_algoritmo_expansion(self):
        gco_mock.cut_grid_graph.reset_mock()
        gco_mock.cut_grid_graph.return_value = np.zeros(16, dtype=np.int32)
        with patch.object(self.gc, "create_gaussian_mixtures_for_states", return_value={}), \
             patch.object(self.gc, "compute_data_cost",
                          return_value=np.zeros((4, 4, 2))), \
             patch.object(self.gc, "compute_nlink_weights",
                          return_value=(np.zeros((3, 4)), np.zeros((4, 3)))):
            self.gc.apply_graph_cuts()
        _, kwargs = gco_mock.cut_grid_graph.call_args
        self.assertEqual(kwargs["algorithm"], "expansion")


# =============================================================================
# 11. calculate_parameters
# =============================================================================
class TestCalculateParameters(unittest.TestCase):
    def setUp(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:5, :] = 50
        image[5:, :] = 200

        final = np.zeros((10, 10), dtype=np.int32)
        final[:5, :] = 0
        final[5:, :] = 1

        self.gc = make_service(
            original_image=image, mask=np.ones((10, 10), dtype=bool),
            final_image=final, num_states=2,
        )

    def test_devuelve_entrada_para_cada_estado(self):
        result = self.gc.calculate_parameters()
        for state in range(self.gc.num_states):
            self.assertIn(state, result)

    def test_media_correcta_por_estado(self):
        result = self.gc.calculate_parameters()
        self.assertAlmostEqual(result[0]["mean"], 50.0)
        self.assertAlmostEqual(result[1]["mean"], 200.0)

    def test_std_siempre_positivo(self):
        result = self.gc.calculate_parameters()
        for state in range(self.gc.num_states):
            self.assertGreater(result[state]["std"], 0)

    def test_estado_vacio_recibe_valores_por_defecto(self):
        gc = make_service(
            original_image=np.zeros((10, 10), dtype=np.uint8),
            mask=np.ones((10, 10), dtype=bool),
            final_image=np.zeros((10, 10), dtype=np.int32),
            num_states=2,
        )
        result = gc.calculate_parameters()
        self.assertEqual(result[1]["mean"], 0)
        self.assertEqual(result[1]["std"], 0)


# =============================================================================
# 12. getSegmentationContainer
# =============================================================================
class TestGetSegmentationContainer(unittest.TestCase):
    def setUp(self):
        self.gc = make_service(
            original_image=np.zeros((5, 5), dtype=np.uint8),
            mask=np.ones((5, 5), dtype=bool),
            final_image=np.zeros((5, 5), dtype=np.int32),
            initial_labels=np.zeros((5, 5), dtype=np.int32),
            parameters={0: {"mean": 0.0, "std": 1e-6}},
            num_states=1,
            lambda_value=0.5,
            sigma=3.0,
            num_iterations=5,
            number_gaussians_per_state=2,
        )

    def test_metodo_es_graph_cuts(self):
        container = self.gc.getSegmentationContainer()
        self.assertEqual(container.method, SegmentationMethod.GRAPH_CUTS)

    def test_conserva_num_states(self):
        container = self.gc.getSegmentationContainer()
        self.assertEqual(container.num_states, 1)

    def test_method_configuration_contiene_todos_los_campos(self):
        container = self.gc.getSegmentationContainer()
        for field in ("lambda_value", "sigma", "num_iterations", "number_gaussians_per_state"):
            self.assertIn(field, container.method_configuration)

    def test_method_configuration_valores_correctos(self):
        container = self.gc.getSegmentationContainer()
        self.assertEqual(container.method_configuration["lambda_value"], 0.5)
        self.assertEqual(container.method_configuration["sigma"], 3.0)
        self.assertEqual(container.method_configuration["num_iterations"], 5)
        self.assertEqual(container.method_configuration["number_gaussians_per_state"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
