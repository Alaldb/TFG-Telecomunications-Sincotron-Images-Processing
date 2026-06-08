import os
import tempfile
import unittest

import numpy as np

from core.session import Session
from persistence.session_io import loadSession, saveSession, exportCorrectedImage


class TestSessionIO(unittest.TestCase):

    def setUp(self):
        self.session = Session(
            image_name="test_image",
            original_image=np.array([[1, 2], [3, 4]], dtype=np.uint16),
            corrected_image=np.array([[5, 6], [7, 8]], dtype=np.float32),
            ising_result=np.array([[0, 1], [1, 0]], dtype=np.int32),
            domain_data={
                "labeled_images": {
                    0: np.array([[1, 0], [0, 2]], dtype=np.int32),
                    1: np.array([[0, 1], [1, 0]], dtype=np.int32),
                }
            },
            parameters={"beta": 2, "num_states": 2},
            domain_stats={
                0: {1: {"area": 10.0}, 2: {"area": 5.0}},
                1: {1: {"area": 8.0}},
            },
        )
        self.tmp = tempfile.NamedTemporaryFile(suffix=".session", delete=False)
        self.path = self.tmp.name
        self.tmp.close()
        saveSession(self.session, self.path)
        self.loaded = loadSession(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_image_name(self):
        self.assertEqual(self.loaded.image_name, self.session.image_name)

    def test_original_image(self):
        np.testing.assert_array_equal(self.loaded.original_image, self.session.original_image)

    def test_corrected_image(self):
        np.testing.assert_array_equal(self.loaded.corrected_image, self.session.corrected_image)

    def test_ising_result(self):
        np.testing.assert_array_equal(self.loaded.ising_result, self.session.ising_result)

    def test_parameters(self):
        self.assertEqual(self.loaded.parameters, self.session.parameters)

    def test_domain_stats(self):
        self.assertEqual(self.loaded.domain_stats, self.session.domain_stats)

    def test_labeled_images(self):
        for state in self.session.domain_data["labeled_images"]:
            np.testing.assert_array_equal(
                self.loaded.domain_data["labeled_images"][state],
                self.session.domain_data["labeled_images"][state]
            )
    
    def test_export_corrected_image(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tif_path = f.name
        try:
            exportCorrectedImage(self.session, tif_path)
            import cv2
            result = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(result)
        finally:
            os.unlink(tif_path)