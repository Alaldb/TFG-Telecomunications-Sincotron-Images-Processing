import unittest
import numpy as np
import cv2
from processing.corrector import Corrector


class TestNormalizeToUint8(unittest.TestCase):

    def setUp(self):
        self.corrector = Corrector()
        rng = np.random.default_rng(42)
        img = rng.normal(loc=0.0, scale=0.1, size=(64, 64)).astype(np.float32)
        self.image = np.clip(img, -0.2, 1.0)

    def test_output_dtype_is_uint8(self):
        result = self.corrector.normalize_to_uint8(self.image)
        self.assertEqual(result.dtype, np.uint8)

    def test_output_min_is_0(self):
        result = self.corrector.normalize_to_uint8(self.image)
        self.assertEqual(result.min(), 0)

    def test_output_max_is_255(self):
        result = self.corrector.normalize_to_uint8(self.image)
        self.assertEqual(result.max(), 255)

    def test_shape_preserved(self):
        result = self.corrector.normalize_to_uint8(self.image)
        self.assertEqual(result.shape, self.image.shape)


class TestHistogramRange(unittest.TestCase):

    def setUp(self):
        self.corrector = Corrector()
        rng = np.random.default_rng(42)
        self.image = rng.integers(80, 120, size=(64, 64), dtype=np.uint8)

    def test_returns_two_ints(self):
        v_low, v_high = self.corrector.histogram_range(self.image, coverage=0.80)
        self.assertIsInstance(v_low, int)
        self.assertIsInstance(v_high, int)

    def test_v_low_less_than_v_high(self):
        v_low, v_high = self.corrector.histogram_range(self.image, coverage=0.80)
        self.assertLess(v_low, v_high)

    def test_range_within_0_255(self):
        v_low, v_high = self.corrector.histogram_range(self.image, coverage=0.80)
        self.assertGreaterEqual(v_low, 0)
        self.assertLessEqual(v_high, 255)

    def test_range_contains_coverage_fraction(self):
        coverage = 0.80
        v_low, v_high = self.corrector.histogram_range(self.image, coverage=coverage)
        hist, _ = np.histogram(self.image.ravel(), bins=256, range=(0, 255))
        pixels_in_range = hist[v_low:v_high + 1].sum()
        self.assertGreaterEqual(pixels_in_range / hist.sum(), coverage - 0.01)

    def test_mask_restricts_pixels(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:32, :] = 50
        img[32:, :] = 200
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32, :] = True
        v_low, v_high = self.corrector.histogram_range(img, coverage=0.80, mask=mask)
        self.assertLess(v_low, 100)
        self.assertLess(v_high, 100)


class TestLinearStretch(unittest.TestCase):

    def setUp(self):
        self.corrector = Corrector()

    def test_v_low_maps_to_0(self):
        img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        result = self.corrector.linear_stretch(img, min_intensity=50, max_intensity=200)
        self.assertEqual(result[0, 0], 0)

    def test_v_high_maps_to_255(self):
        img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        result = self.corrector.linear_stretch(img, min_intensity=50, max_intensity=200)
        self.assertEqual(result[1, 1], 255)

    def test_output_dtype_is_uint8(self):
        img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        result = self.corrector.linear_stretch(img, min_intensity=50, max_intensity=200)
        self.assertEqual(result.dtype, np.uint8)

    def test_raises_when_range_is_zero(self):
        img = np.full((10, 10), 100, dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.corrector.linear_stretch(img, min_intensity=100, max_intensity=100)

    def test_pixels_below_v_low_clip_to_0(self):
        img = np.array([[10, 50, 200]], dtype=np.uint8)
        result = self.corrector.linear_stretch(img, min_intensity=50, max_intensity=200)
        self.assertEqual(result[0, 0], 0)


class TestCreateMask(unittest.TestCase):

    def setUp(self):
        self.corrector = Corrector()
        rng = np.random.default_rng(42)
        img = rng.normal(0.0, 0.1, size=(64, 64)).astype(np.float32)
        self.image = np.clip(img, -0.2, 1.0)

    def test_returns_bool_array(self):
        mask = self.corrector.create_mask(self.image)
        self.assertEqual(mask.dtype, bool)

    def test_shape_matches_image(self):
        mask = self.corrector.create_mask(self.image)
        self.assertEqual(mask.shape, self.image.shape)

    def test_uniform_image_returns_all_true(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        mask = self.corrector.create_mask(img)
        self.assertTrue(mask.all())


class TestApplyCorrection(unittest.TestCase):

    def setUp(self):
        self.corrector = Corrector(coverage=0.80)
        rng = np.random.default_rng(42)
        img = rng.normal(0.0, 0.1, size=(64, 64)).astype(np.float32)
        self.image = np.clip(img, -0.2, 1.0)

    def test_output_dtype_is_uint8(self):
        result = self.corrector.apply_correction(self.image)
        self.assertEqual(result.dtype, np.uint8)

    def test_output_shape_matches_input(self):
        result = self.corrector.apply_correction(self.image)
        self.assertEqual(result.shape, self.image.shape)

    def test_output_range_valid(self):
        result = self.corrector.apply_correction(self.image)
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_manual_v_low_v_high_respected(self):
        rng = np.random.default_rng(42)
        img = rng.integers(80, 120, size=(64, 64), dtype=np.uint8)
        result = self.corrector.apply_correction(img, v_low=80, v_high=120)
        self.assertEqual(result.dtype, np.uint8)

    def test_different_coverage_gives_different_result(self):
        rng = np.random.default_rng(42)
        img = rng.integers(80, 120, size=(64, 64), dtype=np.uint8)
        result_80 = self.corrector.apply_correction(img, coverage=0.80)
        result_90 = self.corrector.apply_correction(img, coverage=0.90)
        self.assertFalse(np.array_equal(result_80, result_90))


class TestCorrectIllumination(unittest.TestCase):

    def setUp(self):
        self.corrector = Corrector()

    def test_correct_illumination_removes_gradient(self):
        gradient = np.tile(np.linspace(0, 0.1, 100), (100, 1)).astype(np.float32)
        noise = np.random.default_rng(42).normal(0, 0.01, (100, 100)).astype(np.float32)
        image = gradient + noise
        mask = np.ones((100, 100), dtype=bool)

        corrected = self.corrector.correct_illumination(image, mask)

        diff_before = abs(image[:, 50:].mean() - image[:, :50].mean())
        diff_after  = abs(corrected[:, 50:].mean() - corrected[:, :50].mean())

        self.assertLess(diff_after, diff_before * 0.2)

    def test_correct_illumination_preserves_high_frequency_signal(self):
        rng = np.random.default_rng(42)
        signal   = rng.normal(0, 0.02, (100, 100)).astype(np.float32)
        gradient = np.tile(np.linspace(0, 0.1, 100), (100, 1)).astype(np.float32)
        image    = signal + gradient
        mask     = np.ones((100, 100), dtype=bool)

        corrected = self.corrector.correct_illumination(image, mask)

        std_original  = signal.std()
        std_corrected = (corrected - corrected.mean()).std()

        self.assertLess(abs(std_corrected - std_original), std_original * 0.3)

    def test_correct_illumination_respects_mask(self):
        image = np.random.default_rng(42).normal(0, 0.02, (100, 100)).astype(np.float32)
        mask  = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        corrected = self.corrector.correct_illumination(image, mask)

        np.testing.assert_array_equal(corrected[~mask], image[~mask])


if __name__ == "__main__":
    unittest.main()