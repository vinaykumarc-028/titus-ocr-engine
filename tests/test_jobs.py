import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.core.db import JSONDatabase
from app.ocr.parser import parse_markdown


class JobsAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_list_jobs(self) -> None:
        response = self.client.get("/api/v1/jobs")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_audit_logs(self) -> None:
        response = self.client.get("/api/v1/audit-logs")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_get_settings(self) -> None:
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        self.assertIn("institution_name", response.json())
        self.assertIn("primary_ocr_engine", response.json())

    def test_download_returns_standalone_html_with_all_pages(self) -> None:
        db = JSONDatabase()
        job_id = "unit-download-html"
        pages = []
        for page_number in range(1, 5):
            parsed = parse_markdown(
                page_number=page_number,
                markdown=f"Question {page_number} Explain page {page_number}. [2 Marks]",
            )
            pages.append(
                {
                    "page_number": page_number,
                    "status": "completed",
                    "markdown": parsed.markdown,
                    "plain_text": parsed.plain_text,
                    "elements": [element.model_dump() for element in parsed.elements],
                }
            )

        db.save_job(
            job_id,
            {
                "id": job_id,
                "name": "Unit HTML Export",
                "metadata": {
                    "title": "Unit HTML Export",
                    "subject": "English",
                    "classGrade": "10",
                    "institutionName": "TITUS School",
                },
                "status": "completed",
                "created_at": 0,
                "pages_count": len(pages),
                "pages_processed": len(pages),
                "pages": pages,
                "failures": [],
                "warnings": [],
            },
        )
        try:
            response = self.client.get(f"/api/v1/jobs/{job_id}/download")
        finally:
            db.delete_job(job_id)

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        body = response.text
        self.assertEqual(body.count('<article class="page"'), 4)
        self.assertIn('data-page="1"', body)
        self.assertIn('data-page="4"', body)
        self.assertIn("Question 4", body)

    def test_complete_review_returns_html_attachment_with_all_pages(self) -> None:
        db = JSONDatabase()
        job_id = "unit-complete-html-all-pages"
        pages = []
        for page_number in range(1, 15):
            parsed = parse_markdown(
                page_number=page_number,
                markdown=f"Question {page_number} Preserve reviewed page {page_number}. [2 Marks]",
            )
            pages.append(
                {
                    "page_number": page_number,
                    "status": "completed" if page_number <= 9 else "pending_review",
                    "markdown": parsed.markdown,
                    "plain_text": parsed.plain_text,
                    "elements": [element.model_dump() for element in parsed.elements],
                }
            )

        db.save_job(
            job_id,
            {
                "id": job_id,
                "name": "Fourteen Page Review",
                "metadata": {
                    "title": "Fourteen Page Review",
                    "subject": "English",
                    "classGrade": "10",
                    "institutionName": "TITUS School",
                },
                "status": "pending_review",
                "created_at": 0,
                "pages_count": len(pages),
                "pages_processed": len(pages),
                "pages": pages,
                "failures": [],
                "warnings": [],
            },
        )
        try:
            response = self.client.post(f"/api/v1/jobs/{job_id}/complete")
        finally:
            db.delete_job(job_id)

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn('attachment; filename="English_10_', response.headers["content-disposition"])
        self.assertTrue(response.headers["content-disposition"].endswith('.html"'))
        body = response.text
        self.assertEqual(body.count('<article class="page"'), 14)
        self.assertIn('data-page="1"', body)
        self.assertIn('data-page="14"', body)
        self.assertIn("Question 14", body)


if __name__ == "__main__":
    unittest.main()
