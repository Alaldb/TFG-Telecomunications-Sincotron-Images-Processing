import unittest
import numpy as np

from core.session import Session


# =============================================================================
# 1. Creación y valores por defecto
# =============================================================================
class TestSessionCreation(unittest.TestCase):

    def setUp(self):
        self.image = np.zeros((10, 10), dtype=np.uint8)
        self.session = Session(image_name="test.tif", original_image=self.image)

    def test_image_name_guardado(self):
        self.assertEqual(self.session.image_name, "test.tif")

    def test_original_image_guardado(self):
        np.testing.assert_array_equal(self.session.original_image, self.image)

    def test_corrected_image_por_defecto_none(self):
        self.assertIsNone(self.session.corrected_image)

    def test_ising_result_por_defecto_none(self):
        self.assertIsNone(self.session.ising_result)

    def test_domain_data_por_defecto_dict_vacio(self):
        self.assertEqual(self.session.domain_data, {})

    def test_parameters_por_defecto_dict_vacio(self):
        self.assertEqual(self.session.parameters, {})

    def test_stats_por_defecto_dict_vacio(self):
        self.assertEqual(self.session.ising_stats, {})

    def test_timestamp_generado_automaticamente(self):
        self.assertIsNotNone(self.session.timestamp)
        self.assertIsInstance(self.session.timestamp, str)


# =============================================================================
# 2. Aislamiento entre instancias (los dicts mutables no se comparten)
# =============================================================================
class TestSessionIsolation(unittest.TestCase):

    def setUp(self):
        self.img = np.zeros((5, 5), dtype=np.uint8)

    def test_domain_data_no_compartido(self):
        s1 = Session(image_name="a.tif", original_image=self.img)
        s2 = Session(image_name="b.tif", original_image=self.img)
        s1.domain_data["key"] = "value"
        self.assertNotIn("key", s2.domain_data)

    def test_parameters_no_compartido(self):
        s1 = Session(image_name="a.tif", original_image=self.img)
        s2 = Session(image_name="b.tif", original_image=self.img)
        s1.parameters["beta"] = 2
        self.assertNotIn("beta", s2.parameters)

    def test_stats_no_compartido(self):
        s1 = Session(image_name="a.tif", original_image=self.img)
        s2 = Session(image_name="b.tif", original_image=self.img)
        s1.ising_stats["area"] = 100
        self.assertNotIn("area", s2.ising_stats)


# =============================================================================
# 3. Asignación de campos opcionales
# =============================================================================
class TestSessionFieldAssignment(unittest.TestCase):

    def setUp(self):
        self.img = np.zeros((10, 10), dtype=np.uint8)
        self.session = Session(image_name="img.tif", original_image=self.img)

    def test_corrected_image_asignable(self):
        corrected = np.ones((10, 10), dtype=np.uint8) * 128
        self.session.corrected_image = corrected
        np.testing.assert_array_equal(self.session.corrected_image, corrected)

    def test_ising_result_asignable(self):
        result = np.zeros((10, 10), dtype=np.int32)
        self.session.ising_result = result
        np.testing.assert_array_equal(self.session.ising_result, result)

    def test_parameters_asignable(self):
        self.session.parameters = {"beta": 3, "coverage": 0.9}
        self.assertEqual(self.session.parameters["beta"], 3)

    def test_temporal_ising_object_asignable(self):
        mock_obj = object()
        self.session.temporal_ising_object = mock_obj
        self.assertIs(self.session.temporal_ising_object, mock_obj)


if __name__ == "__main__":
    unittest.main(verbosity=2)
