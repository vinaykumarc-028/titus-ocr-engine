import unittest

from fastapi.testclient import TestClient

from app.main import app


class DevValidationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_dev_validation_page_is_isolated(self) -> None:
        response = self.client.get("/dev/ocr-validation")

        self.assertEqual(response.status_code, 200)
        self.assertIn("DEV TOOL ONLY", response.text)
        self.assertIn("/dev/ocr-validation/process", response.text)

    def test_ground_truth_must_be_text_or_pdf_file(self) -> None:
        response = self.client.post(
            "/dev/ocr-validation/process",
            files=[
                ("files", ("bad.txt", b"not an image", "text/plain")),
                ("ground_truth", ("truth.png", b"truth", "image/png")),
            ],
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Ground truth must be a .txt or .pdf file.")


if __name__ == "__main__":
    unittest.main()
