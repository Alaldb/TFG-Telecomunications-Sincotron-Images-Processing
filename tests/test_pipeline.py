import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.pipeline import PipelineDictator
from core.session import Session


# =============================================================================
# 1. __init__ — parámetros
# =============================================================================
class TestPipelineDictatorInit(unittest.TestCase):

    def test_valores_por_defecto(self):
        p = PipelineDictator()
        self.assertEqual(p.coverage, 0.80)
        self.assertEqual(p.beta, 2)
        self.assertEqual(p.max_iterations, 20)
        self.assertEqual(p.num_states, 3)

    def test_parametros_personalizados(self):
        p = PipelineDictator(coverage=0.9, beta=5, max_iterations=50, num_states=2)
        self.assertEqual(p.coverage, 0.9)
        self.assertEqual(p.beta, 5)
        self.assertEqual(p.max_iterations, 50)
        self.assertEqual(p.num_states, 2)


# =============================================================================
# 2. apply_correction
# =============================================================================
class TestApplyCorrection(unittest.TestCase):

    def setUp(self):
        self.image = np.zeros((10, 10), dtype=np.uint8)
        self.corrected = np.ones((10, 10), dtype=np.uint8) * 128

    @patch("core.pipeline.Corrector")
    def test_devuelve_session(self, mock_cls):
        mock_cls.return_value.apply_correction.return_value = self.corrected
        result = PipelineDictator().apply_correction(self.image, "img.tif")
        self.assertIsInstance(result, Session)

    @patch("core.pipeline.Corrector")
    def test_image_name_en_session(self, mock_cls):
        mock_cls.return_value.apply_correction.return_value = self.corrected
        result = PipelineDictator().apply_correction(self.image, "img.tif")
        self.assertEqual(result.image_name, "img.tif")

    @patch("core.pipeline.Corrector")
    def test_corrected_image_no_es_none(self, mock_cls):
        mock_cls.return_value.apply_correction.return_value = self.corrected
        result = PipelineDictator().apply_correction(self.image, "img.tif")
        self.assertIsNotNone(result.corrected_image)

    @patch("core.pipeline.Corrector")
    def test_corrected_image_es_la_devuelta_por_corrector(self, mock_cls):
        mock_cls.return_value.apply_correction.return_value = self.corrected
        result = PipelineDictator().apply_correction(self.image, "img.tif")
        np.testing.assert_array_equal(result.corrected_image, self.corrected)

    @patch("core.pipeline.Corrector")
    def test_parametros_guardados_en_session(self, mock_cls):
        mock_cls.return_value.apply_correction.return_value = self.corrected
        result = PipelineDictator(coverage=0.9, beta=5).apply_correction(self.image, "img.tif")
        self.assertEqual(result.parameters["coverage"], 0.9)
        self.assertEqual(result.parameters["beta"], 5)

    @patch("core.pipeline.Corrector")
    def test_original_image_es_copia(self, mock_cls):
        mock_cls.return_value.apply_correction.return_value = self.corrected
        result = PipelineDictator().apply_correction(self.image, "img.tif")
        self.image[0, 0] = 99
        self.assertEqual(result.original_image[0, 0], 0)

    @patch("core.pipeline.Corrector")
    def test_corrector_instanciado_con_coverage(self, mock_cls):
        mock_cls.return_value.apply_correction.return_value = self.corrected
        PipelineDictator(coverage=0.85).apply_correction(self.image, "img.tif")
        mock_cls.assert_called_once_with(coverage=0.85)


# =============================================================================
# 3. run_ising
# =============================================================================
class TestRunIsing(unittest.TestCase):

    def setUp(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        self.session = Session(
            image_name="img.tif",
            original_image=img,
            corrected_image=np.ones((10, 10), dtype=np.uint8) * 128,
        )
        self.ising_result = np.zeros((10, 10), dtype=np.int32)

    @patch("core.pipeline.Ising")
    def test_ising_result_no_es_none(self, mock_cls):
        mock_cls.return_value.final_image = self.ising_result
        result = PipelineDictator().run_ising(self.session)
        self.assertIsNotNone(result.ising_result)

    @patch("core.pipeline.Ising")
    def test_ising_result_coincide_con_final_image(self, mock_cls):
        mock_cls.return_value.final_image = self.ising_result
        result = PipelineDictator().run_ising(self.session)
        np.testing.assert_array_equal(result.ising_result, self.ising_result)

    @patch("core.pipeline.Ising")
    def test_temporal_ising_object_guardado(self, mock_cls):
        mock_ising = mock_cls.return_value
        mock_ising.final_image = self.ising_result
        result = PipelineDictator().run_ising(self.session)
        self.assertIs(result.temporal_ising_object, mock_ising)

    @patch("core.pipeline.Ising")
    def test_run_llamado_con_corrected_image(self, mock_cls):
        mock_cls.return_value.final_image = self.ising_result
        PipelineDictator().run_ising(self.session)
        mock_cls.return_value.run.assert_called_once_with(self.session.corrected_image)

    @patch("core.pipeline.Ising")
    def test_devuelve_misma_session(self, mock_cls):
        mock_cls.return_value.final_image = self.ising_result
        result = PipelineDictator().run_ising(self.session)
        self.assertIs(result, self.session)

    @patch("core.pipeline.Ising")
    def test_ising_instanciado_con_parametros_correctos(self, mock_cls):
        mock_cls.return_value.final_image = self.ising_result
        PipelineDictator(beta=7, max_iterations=15, num_states=2).run_ising(self.session)
        mock_cls.assert_called_once_with(beta=7, max_iterations=15, num_states=2)


# =============================================================================
# 4. run_domains
# =============================================================================
class TestRunDomains(unittest.TestCase):

    def setUp(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        self.mock_ising = MagicMock()
        self.session = Session(
            image_name="img.tif",
            original_image=img,
            corrected_image=img,
            ising_result=np.zeros((10, 10), dtype=np.int32),
            temporal_ising_object=self.mock_ising,
        )

    def _mock_domain_data(self, extra: dict | None = None) -> dict:
        """Datos mínimos que get_data() debe devolver para que run_domains funcione."""
        base = {"labeled_images": {0: np.zeros((10, 10), dtype=np.int32)}}
        if extra:
            base.update(extra)
        return base

    @patch("core.pipeline.DomainService")
    def test_domain_data_no_esta_vacio(self, mock_cls):
        mock_cls.return_value.get_data.return_value = self._mock_domain_data({"num_domains": {0: 2}})
        result = PipelineDictator().run_domains(self.session)
        self.assertNotEqual(result.domain_data, {})

    @patch("core.pipeline.DomainService")
    def test_domain_service_instanciado_con_ising_object(self, mock_cls):
        mock_cls.return_value.get_data.return_value = self._mock_domain_data()
        PipelineDictator().run_domains(self.session)
        mock_cls.assert_called_once_with(self.mock_ising)

    @patch("core.pipeline.DomainService")
    def test_devuelve_misma_session(self, mock_cls):
        mock_cls.return_value.get_data.return_value = self._mock_domain_data()
        result = PipelineDictator().run_domains(self.session)
        self.assertIs(result, self.session)

    @patch("core.pipeline.DomainService")
    def test_domain_data_es_el_devuelto_por_get_data(self, mock_cls):
        expected = self._mock_domain_data({"num_domains": {0: 3, 1: 1}})
        mock_cls.return_value.get_data.return_value = expected
        result = PipelineDictator().run_domains(self.session)
        self.assertEqual(result.domain_data, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
