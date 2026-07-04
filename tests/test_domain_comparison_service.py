from __future__ import annotations

import math
import unittest
import numpy as np

from processing.domainComparisonService import DomainComparisonService


# =============================================================================
# Helpers
# =============================================================================

def _make_domain(
    area: float = 100.0,
    perimeter: float = 40.0,
    roughness: float = 1.2,
    coords: np.ndarray | None = None,
    values: np.ndarray | None = None,
) -> dict:
    if coords is None:
        coords = np.array([[i, i] for i in range(int(area))], dtype=float).reshape(-1, 2)
    if values is None:
        values = np.full(len(coords), 128.0)
    return {
        "stats":  {"area": area, "perimeter": perimeter, "roughness": roughness},
        "coords": coords,
        "values": values,
    }


def _make_service(
    domain_a: dict | None = None,
    domain_b: dict | None = None,
    image_shape: tuple[int, int] = (100, 100),
) -> DomainComparisonService:
    if domain_a is None:
        domain_a = _make_domain()
    if domain_b is None:
        domain_b = _make_domain()
    return DomainComparisonService(domain_a, domain_b, image_shape)


# =============================================================================
# 1. Inicialización
# =============================================================================

class TestInit(unittest.TestCase):

    def test_image_diagonal_calculo_correcto(self):
        svc = _make_service(image_shape=(30, 40))
        self.assertAlmostEqual(svc.image_diagonal, 50.0, places=5)

    def test_image_diagonal_imagen_cuadrada(self):
        svc = _make_service(image_shape=(10, 10))
        self.assertAlmostEqual(svc.image_diagonal, math.sqrt(200), places=5)


# =============================================================================
# 2. Estructura del resultado de compare()
# =============================================================================

class TestCompareStructure(unittest.TestCase):

    def test_compare_devuelve_stat_diffs_y_displacement(self):
        result = _make_service().compare()
        self.assertIn("stat_diffs", result)
        self.assertIn("displacement", result)

    def test_stat_diffs_contiene_metricas_de_stats(self):
        result = _make_service().compare()
        for key in ("area", "perimeter", "roughness", "mean_intensity"):
            self.assertIn(key, result["stat_diffs"])

    def test_cada_metrica_tiene_cuatro_campos(self):
        result = _make_service().compare()
        for metric, data in result["stat_diffs"].items():
            for field in ("value_a", "value_b", "absolute", "percentage"):
                self.assertIn(field, data, msg=f"Campo '{field}' ausente en métrica '{metric}'")

    def test_displacement_contiene_todos_los_campos(self):
        result = _make_service().compare()
        disp = result["displacement"]
        for field in ("centroid_a", "centroid_b", "displacement_vector", "vector_module", "normalized_vector"):
            self.assertIn(field, disp)


# =============================================================================
# 3. Diferencias de estadísticas
# =============================================================================

class TestStatDiffs(unittest.TestCase):

    def test_absolute_es_b_menos_a(self):
        a = _make_domain(area=100.0)
        b = _make_domain(area=150.0)
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["stat_diffs"]["area"]["absolute"], 50.0)

    def test_absolute_negativo_cuando_b_menor_que_a(self):
        a = _make_domain(area=200.0)
        b = _make_domain(area=100.0)
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["stat_diffs"]["area"]["absolute"], -100.0)

    def test_percentage_correcto(self):
        a = _make_domain(area=100.0)
        b = _make_domain(area=150.0)
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["stat_diffs"]["area"]["percentage"], 50.0)

    def test_percentage_cero_cuando_dominios_identicos(self):
        a = _make_domain(area=100.0)
        b = _make_domain(area=100.0)
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["stat_diffs"]["area"]["percentage"], 0.0)

    def test_percentage_es_cero_cuando_value_a_es_cero(self):
        a = _make_domain(area=0.0)
        b = _make_domain(area=50.0)
        result = _make_service(a, b).compare()
        self.assertEqual(result["stat_diffs"]["area"]["percentage"], 0)

    def test_value_a_y_value_b_se_guardan_correctamente(self):
        a = _make_domain(area=80.0)
        b = _make_domain(area=120.0)
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["stat_diffs"]["area"]["value_a"], 80.0)
        self.assertAlmostEqual(result["stat_diffs"]["area"]["value_b"], 120.0)

    def test_dinamico_nueva_stat_se_calcula(self):
        """Si stats incluye una clave nueva, el servicio la procesa sin cambios."""
        a = _make_domain()
        b = _make_domain()
        a["stats"]["new_metric"] = 10.0
        b["stats"]["new_metric"] = 20.0
        result = _make_service(a, b).compare()
        self.assertIn("new_metric", result["stat_diffs"])
        self.assertAlmostEqual(result["stat_diffs"]["new_metric"]["absolute"], 10.0)


# =============================================================================
# 4. Intensidad media
# =============================================================================

class TestMeanIntensity(unittest.TestCase):

    def test_mean_intensity_se_calcula_correctamente(self):
        a = _make_domain(values=np.array([100.0, 100.0, 100.0]))
        b = _make_domain(values=np.array([200.0, 200.0, 200.0]))
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["stat_diffs"]["mean_intensity"]["value_a"], 100.0)
        self.assertAlmostEqual(result["stat_diffs"]["mean_intensity"]["value_b"], 200.0)

    def test_mean_intensity_values_vacios_devuelve_cero(self):
        a = _make_domain(values=np.array([]))
        b = _make_domain(values=np.array([]))
        result = _make_service(a, b).compare()
        self.assertEqual(result["stat_diffs"]["mean_intensity"]["value_a"], 0)
        self.assertEqual(result["stat_diffs"]["mean_intensity"]["value_b"], 0)


# =============================================================================
# 5. Desplazamiento
# =============================================================================

class TestDisplacement(unittest.TestCase):

    def test_mismo_centroide_distancia_cero(self):
        coords = np.array([[0, 0], [2, 0], [1, 1]], dtype=float)
        a = _make_domain(coords=coords)
        b = _make_domain(coords=coords.copy())
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["displacement"]["vector_module"], 0.0)

    def test_centroide_a_se_calcula_correctamente(self):
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]])
        a = _make_domain(coords=coords)
        result = _make_service(a, _make_domain()).compare()
        cy, cx = result["displacement"]["centroid_a"]
        self.assertAlmostEqual(cy, 1.0)
        self.assertAlmostEqual(cx, 1.0)

    def test_vector_modulo_correcto(self):
        coords_a = np.array([[0.0, 0.0]])
        coords_b = np.array([[3.0, 4.0]])
        a = _make_domain(coords=coords_a)
        b = _make_domain(coords=coords_b)
        result = _make_service(a, b).compare()
        self.assertAlmostEqual(result["displacement"]["vector_module"], 5.0)

    def test_normalized_entre_cero_y_uno(self):
        coords_a = np.array([[0.0, 0.0]])
        coords_b = np.array([[30.0, 40.0]])
        a = _make_domain(coords=coords_a)
        b = _make_domain(coords=coords_b)
        result = _make_service(a, b, image_shape=(100, 100)).compare()
        norm = result["displacement"]["normalized_vector"]
        self.assertGreaterEqual(norm, 0.0)
        self.assertLessEqual(norm, 1.0)

    def test_normalized_esquinas_opuestas_es_uno(self):
        """Centroides en esquinas opuestas → normalized ≈ 1.0."""
        coords_a = np.array([[0.0, 0.0]])
        coords_b = np.array([[100.0, 100.0]])
        a = _make_domain(coords=coords_a)
        b = _make_domain(coords=coords_b)
        result = _make_service(a, b, image_shape=(100, 100)).compare()
        self.assertAlmostEqual(result["displacement"]["normalized_vector"], 1.0, places=5)

    def test_displacement_vector_direccion_correcta(self):
        coords_a = np.array([[0.0, 0.0]])
        coords_b = np.array([[0.0, 5.0]])  # desplazado solo en X
        a = _make_domain(coords=coords_a)
        b = _make_domain(coords=coords_b)
        result = _make_service(a, b).compare()
        dx, dy = result["displacement"]["displacement_vector"]
        self.assertAlmostEqual(dx, 5.0)
        self.assertAlmostEqual(dy, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)