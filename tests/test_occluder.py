import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.occluder import apply_block_occlusion

class TestOccluder(unittest.TestCase):
    def setUp(self):
        # Create a mock 480x640 BGR test image
        self.image = np.ones((480, 640, 3), dtype=np.uint8) * 200

    def test_zero_occlusion_returns_identical(self):
        res = apply_block_occlusion(self.image, occlusion_pct=0.0)
        np.testing.assert_array_equal(res, self.image)

    def test_occlusion_shape_and_dtype(self):
        res = apply_block_occlusion(self.image, occlusion_pct=0.30, occlusion_type="black", seed=42)
        self.assertEqual(res.shape, self.image.shape)
        self.assertEqual(res.dtype, np.uint8)
        # Should have pixels modified to black (0)
        self.assertTrue(np.any(res == 0))

    def test_occlusion_reproducibility_with_seed(self):
        res1 = apply_block_occlusion(self.image, occlusion_pct=0.45, seed=123)
        res2 = apply_block_occlusion(self.image, occlusion_pct=0.45, seed=123)
        np.testing.assert_array_equal(res1, res2)

    def test_custom_bounding_box(self):
        bbox = (100, 100, 300, 300)
        res = apply_block_occlusion(self.image, occlusion_pct=0.20, hand_bbox=bbox, seed=99)
        self.assertEqual(res.shape, self.image.shape)

    def test_noise_and_gray_types(self):
        gray_res = apply_block_occlusion(self.image, occlusion_pct=0.20, occlusion_type="gray", seed=1)
        noise_res = apply_block_occlusion(self.image, occlusion_pct=0.20, occlusion_type="noise", seed=1)
        self.assertEqual(gray_res.shape, self.image.shape)
        self.assertEqual(noise_res.shape, self.image.shape)

if __name__ == "__main__":
    unittest.main()
