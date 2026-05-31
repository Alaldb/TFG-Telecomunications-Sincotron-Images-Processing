from __future__ import annotations

import math
import unittest
import numpy as np

from stats.domainStats import computeDomainStats


def _single_domain_labeled(shape: tuple[int, int] = (20, 20)) -> dict[int, np.ndarray]:
    """Estado 0 con un único dominio rectangular 4×4 centrado."""
    labeled = np.zeros(shape, dtype=np.int32)
    labeled[8:12, 8:12] = 1          # dominio 1: 4×4 = 16 píxeles
    return {0: labeled}


def _two_states_labeled() -> dict[int, np.ndarray]:
    """Dos estados, cada uno con un dominio."""
    state0 = np.zeros((20, 20), dtype=np.int32)
    state0[2:5, 2:5] = 1             # 3×3 = 9 píxeles

    state1 = np.zeros((20, 20), dtype=np.int32)
    state1[10:16, 10:16] = 1         # 6×6 = 36 píxeles
    return {0: state0, 1: state1}


def _circle_labeled(radius: int = 15) -> dict[int, np.ndarray]:
    """Estado 0 con un dominio aproximadamente circular."""
    size = radius * 2 + 10
    labeled = np.zeros((size, size), dtype=np.int32)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    labeled[mask] = 1
    return {0: labeled}


# =============================================================================
# 1. Estructura del resultado
# =============================================================================
class TestComputeDomainStatsStructure(unittest.TestCase):

    def test_claves_de_estado_coinciden(self):
        labeled = _two_states_labeled()
        result = computeDomainStats(labeled)
        self.assertEqual(set(result.keys()), {0, 1})

    def test_claves_de_dominio_coinciden_con_labeled(self):
        labeled = _single_domain_labeled()
        result = computeDomainStats(labeled)
        self.assertIn(1, result[0])

    def test_no_incluye_id_cero_background(self):
        labeled = _single_domain_labeled()
        result = computeDomainStats(labeled)
        self.assertNotIn(0, result[0])

    def test_estado_sin_dominios_devuelve_dict_vacio(self):
        labeled = {0: np.zeros((10, 10), dtype=np.int32)}
        result = computeDomainStats(labeled)
        self.assertEqual(result[0], {})

    def test_metricas_minimas_presentes(self):
        result = computeDomainStats(_single_domain_labeled())
        domain = result[0][1]
        for key in ("area", "perimeter", "roughness"):
            self.assertIn(key, domain)


# =============================================================================
# 2. Exactitud de area
# =============================================================================
class TestArea(unittest.TestCase):

    def test_area_rectangulo_4x4(self):
        result = computeDomainStats(_single_domain_labeled())
        self.assertEqual(result[0][1]["area"], 16.0)

    def test_area_rectangulo_3x3(self):
        result = computeDomainStats(_two_states_labeled())
        self.assertEqual(result[0][1]["area"], 9.0)

    def test_area_rectangulo_6x6(self):
        result = computeDomainStats(_two_states_labeled())
        self.assertEqual(result[1][1]["area"], 36.0)

    def test_area_es_float(self):
        result = computeDomainStats(_single_domain_labeled())
        self.assertIsInstance(result[0][1]["area"], float)


# =============================================================================
# 3. Propiedad matemática de roughness
# =============================================================================
class TestRoughness(unittest.TestCase):

    def test_roughness_positivo_rectangulo(self):
        result = computeDomainStats(_single_domain_labeled())
        self.assertGreater(result[0][1]["roughness"], 0.0)

    def test_roughness_positivo_dos_estados(self):
        result = computeDomainStats(_two_states_labeled())
        for state_data in result.values():
            for domain in state_data.values():
                self.assertGreater(domain["roughness"], 0.0)

    def test_roughness_circulo_proximo_a_uno(self):
        """Un círculo discreto debe tener roughness cercano a 1.0."""
        result = computeDomainStats(_circle_labeled(radius=15))
        roughness = result[0][1]["roughness"]
        self.assertLess(roughness, 1.3, msg=f"roughness={roughness:.4f} demasiado alto para un círculo")

    def test_roughness_formula_manual(self):
        """Verificación explícita de la fórmula P²/(4π·A)."""
        result = computeDomainStats(_single_domain_labeled())
        d = result[0][1]
        expected = d["perimeter"] ** 2 / (4 * math.pi * d["area"])
        self.assertAlmostEqual(d["roughness"], expected, places=6)


# =============================================================================
# 4. Extensibilidad
# =============================================================================
class TestExtensibilidad(unittest.TestCase):

    def test_metricas_adicionales_no_rompen_estructura(self):
        """Si se añaden claves extra al dict de dominio, area/perimeter/roughness siguen presentes."""
        result = computeDomainStats(_single_domain_labeled())
        domain = result[0][1]
        for key in ("area", "perimeter", "roughness"):
            self.assertIn(key, domain)

    def test_multiples_dominios_mismo_estado(self):
        labeled = np.zeros((30, 30), dtype=np.int32)
        labeled[2:5, 2:5] = 1
        labeled[20:25, 20:25] = 2
        result = computeDomainStats({0: labeled})
        self.assertIn(1, result[0])
        self.assertIn(2, result[0])
        self.assertEqual(result[0][1]["area"], 9.0)
        self.assertEqual(result[0][2]["area"], 25.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
