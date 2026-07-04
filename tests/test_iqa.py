import unittest
from pathlib import Path
from PIL import Image
from app.core.iqa import analyze_image_quality

class TestIQA(unittest.TestCase):
    def setUp(self):
        # Create a temp test image
        self.test_img_path = Path("tests/test_sample.png")
        # Draw a basic white image
        img = Image.new("RGB", (300, 400), color="white")
        img.save(self.test_img_path)

    def tearDown(self):
        if self.test_img_path.exists():
            self.test_img_path.unlink()

    def test_iqa_basic_metrics(self):
        res = analyze_image_quality(self.test_img_path)
        self.assertIn("resolution", res)
        self.assertIn("blur", res)
        self.assertIn("contrast", res)
        self.assertIn("noise", res)
        self.assertIn("perspective", res)
        self.assertIn("exposure", res)
        self.assertIn("difficulty", res)
        self.assertIn("overall_status", res)
        self.assertIn("recommendations", res)
        
        # Verify resolution values
        self.assertEqual(res["resolution"]["value"], "300x400 (0.12 MP)")
        self.assertEqual(res["resolution"]["status"], "red")  # Under 0.4 MP is low resolution

if __name__ == "__main__":
    unittest.main()
